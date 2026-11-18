"""
Python chatbot web server (Flask).

- Serves the static UI from ./html for every request that does NOT start with /api
- POST /api/chatbot  {"question": "...", "lang": "en|no|uk"}  ->  {"answer": "...", ...}

Answers are produced by a local LLM (Qwen2.5-Coder-1.5B-Instruct, GGUF, CPU) stored in ./data.
Before generation, the question is used to search a small Python knowledge base
(data/python_kb.json); the best matching entries are given to the model as context
(retrieval-augmented generation), which keeps answers short and correct.

Run:  .venv\\Scripts\\python app.py   ->  http://localhost:2008
"""
import json
import math
import os
import re
import threading
import time
from collections import Counter

from flask import Flask, abort, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_DIR = os.path.join(BASE_DIR, "html")
DATA_DIR = os.path.join(BASE_DIR, "data")
KB_FILE = os.path.join(DATA_DIR, "python_kb.json")
PORT = 2008

LANG_NAMES = {"en": "English", "no": "Norwegian (Bokmål)", "uk": "Ukrainian"}
OFF_TOPIC = {
    "en": "Sorry, I can only answer questions about **Python** programming.",
    "no": "Beklager, jeg kan bare svare på spørsmål om **Python**-programmering.",
    "uk": "Вибачте, я можу відповідати лише на питання про програмування на **Python**.",
}

app = Flask(__name__, static_folder=None)


# --------------------------------------------------------------------------- #
#  Knowledge-base search (BM25 over a small set of Python facts)
# --------------------------------------------------------------------------- #
TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
STOP = set("""a an the is are was were be been of to in on for and or with what which how
why do does can could i you it its this that these those my me we our there their
python pythons about tell explain please give list show use using used""".split())


def tokenize(text):
    out = []
    for t in TOKEN_RE.findall(text.lower()):
        if t in STOP or len(t) < 2:
            continue
        # tiny stemmer: types -> type, classes -> class, functions -> function
        if len(t) > 4 and t.endswith("es") and not t.endswith("ses"):
            t = t[:-1]
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class KnowledgeBase:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            self.entries = json.load(f)
        # title and keywords are repeated so they weigh more than the body text
        self.docs = [tokenize(" ".join([e["title"]] * 2 + e.get("keywords", []) * 2 + [e["text"]]))
                     for e in self.entries]
        self.avgdl = sum(len(d) for d in self.docs) / max(1, len(self.docs))
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tfs = [Counter(d) for d in self.docs]

    def search(self, query, k=3, k1=1.5, b=0.75):
        q = tokenize(query)
        scored = []
        for i, tf in enumerate(self.tfs):
            dl = len(self.docs[i])
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / self.avgdl))
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [(round(s, 2), self.entries[i]) for s, i in scored[:k]]


KB = KnowledgeBase(KB_FILE)

# Programming vocabulary (word stems) in English, Norwegian and Ukrainian.
# Used to decide whether a question the knowledge base knows nothing about is still on topic.
# (A 1.5B model is not reliable at this kind of classification, so a deterministic check is used.)
PROGRAMMING_STEMS = """
python pip venv code coding program script function method class object variable loop iterat
list dict tuple string str int float bool array set module package import librar syntax error
exception debug compile interpret def lambda return print input file json csv regex thread async
recurs algorithm sort search index slice type decorator generator yield api http flask django numpy
pandas test unittest pytest oop inherit instance attribute argument parameter operator boolean
kode programm funksjon klasse variabel løkke liste ordbok streng modul pakke bibliotek feil unntak
fil objekt metode tråd sorter
python код програм функці метод клас обʼєкт об'єкт змінн цикл список словник кортеж рядок модул
пакет бібліотек синтаксис помилк виняток файл тип декоратор генератор потік сортув масив
""".split()
CODE_CHARS = re.compile(r"[(){}\[\]=<>:]|\w+\.\w+\(")


def looks_like_programming(question):
    q = question.lower().replace("ʼ", "'")
    if CODE_CHARS.search(q):
        return True
    words = re.findall(r"[\w']+", q)
    return any(w.startswith(stem) for w in words for stem in PROGRAMMING_STEMS)


# --------------------------------------------------------------------------- #
#  Local LLM (llama.cpp, CPU)
# --------------------------------------------------------------------------- #
class LocalLLM:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.llm = None
        self.model_name = None
        self.error = None
        self.lock = threading.Lock()

    def find_model(self):
        if not os.path.isdir(self.data_dir):
            return None
        ggufs = sorted(f for f in os.listdir(self.data_dir) if f.lower().endswith(".gguf"))
        return os.path.join(self.data_dir, ggufs[0]) if ggufs else None

    def load(self):
        if self.llm is not None or self.error:
            return self.llm
        path = self.find_model()
        if not path:
            self.error = "No .gguf model found in ./data (run: python download_model.py)"
            return None
        try:
            from llama_cpp import Llama
            print(f"[llm] loading {path} ...", flush=True)
            t = time.time()
            self.llm = Llama(model_path=path, n_ctx=4096, n_threads=max(2, (os.cpu_count() or 4) - 1),
                             verbose=False)
            self.model_name = os.path.basename(path)
            print(f"[llm] loaded in {time.time() - t:.1f}s", flush=True)
        except Exception as e:  # pragma: no cover
            self.error = f"Could not load model: {e}"
            print("[llm]", self.error, flush=True)
        return self.llm

    def answer(self, question, context_entries, lang):
        llm = self.load()
        if llm is None:
            return None
        context = "\n\n".join(f"[{e['title']}]\n{e['text']}" for _, e in context_entries) or "(nothing found)"
        system = (
            "You are a helpful Python programming assistant. You answer ONLY questions about the "
            "Python programming language, its standard library and Python programming in general. "
            "If the question is not related to Python, politely say that you can only answer "
            "questions about Python. Use the reference notes when they are relevant; they are "
            "correct. Be concise and clear, use Markdown, and give short code examples in "
            f"```python blocks when helpful. Write the answer in {LANG_NAMES.get(lang, 'English')}."
        )
        language = LANG_NAMES.get(lang, "English")
        user = (f"Reference notes:\n{context}\n\nQuestion: {question}\n\n"
                f"Answer in {language}.")
        with self.lock:  # llama.cpp context is not thread-safe
            out = llm.create_chat_completion(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=700, temperature=0.2, top_p=0.9, repeat_penalty=1.1,
            )
        return out["choices"][0]["message"]["content"].strip()


LLM = LocalLLM(DATA_DIR)


# --------------------------------------------------------------------------- #
#  API
# --------------------------------------------------------------------------- #
@app.post("/api/chatbot")
def chatbot():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or request.form.get("question") or "").strip()
    lang = payload.get("lang", "en")
    if lang not in LANG_NAMES:
        lang = "en"
    if not question:
        return jsonify(error="empty question"), 400
    if len(question) > 4000:
        question = question[:4000]

    t0 = time.time()
    hits = KB.search(question, k=3)
    if not hits and not looks_like_programming(question):
        return jsonify(answer=OFF_TOPIC[lang], mode="off-topic", model=LLM.model_name,
                       sources=[], seconds=round(time.time() - t0, 2))
    answer = LLM.answer(question, hits, lang)
    mode = "llm+search"
    if answer is None:  # model unavailable -> fall back to pure knowledge-base search
        mode = "search-only"
        if hits:
            answer = "\n\n".join(f"**{e['title']}**\n\n{e['text']}" for _, e in hits)
        else:
            answer = "Sorry, I found nothing about that in my Python knowledge base."
        if LLM.error:
            answer += f"\n\n_({LLM.error})_"
    return jsonify(
        answer=answer,
        mode=mode,
        model=LLM.model_name,
        sources=[{"title": e["title"], "score": s} for s, e in hits],
        seconds=round(time.time() - t0, 2),
    )


@app.route("/api/<path:_any>", methods=["GET", "POST", "PUT", "DELETE"])
def api_unknown(_any):
    return jsonify(error="unknown API endpoint"), 404


# --------------------------------------------------------------------------- #
#  Static content: everything that does not start with /api
# --------------------------------------------------------------------------- #
@app.get("/", defaults={"path": ""})
@app.get("/<path:path>")
def static_files(path):
    if path.startswith("api"):
        abort(404)
    full = os.path.join(HTML_DIR, path)
    if path and os.path.isfile(full):
        return send_from_directory(HTML_DIR, path)
    if path and os.path.isfile(full + ".html"):
        return send_from_directory(HTML_DIR, path + ".html")
    return send_from_directory(HTML_DIR, "index.html")


if __name__ == "__main__":
    # load the model in the background so the first question is fast
    threading.Thread(target=LLM.load, daemon=True).start()
    print(f"Serving on http://localhost:{PORT}", flush=True)
    app.run(host="0.0.0.0", port=PORT, threaded=True)
