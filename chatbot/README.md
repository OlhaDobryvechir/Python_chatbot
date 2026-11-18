# Python Chatbot + Mini Chatbot (neural-network demo)

## Run
```
python -m venv .venv
.venv\Scripts\python -m pip install flask huggingface_hub
.venv\Scripts\python -m pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
.venv\Scripts\python download_model.py      # ~1 GB model into ./data (already done)
.venv\Scripts\python app.py                 # http://localhost:2008
```

## Structure
| Path | Purpose |
|---|---|
| `app.py` | Flask server, port 2008. `POST /api/chatbot`; every other non-`/api` path serves `html/` |
| `data/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` | Local LLM (Qwen2.5-Coder 1.5B Instruct, 4-bit, runs on CPU) |
| `data/python_kb.json` | Python knowledge base searched (BM25) before each answer; best hits are given to the LLM as context |
| `html/index.html`, `js/app.js` | Chatbot UI (EN / NO / UK) |
| `html/minichat.html`, `js/minichat.js` | Neural-network visualisation (SVG) |
| `html/js/model.js` | Vocabulary (15 + 15 numbered words), layer sizes, pretraining / training / test records |
| `html/js/nn.js` | 5-layer network on math.js: forward pass, back-propagation, Adam |

## Mini chatbot data
Input `animal state situation` → output `sound action condition`, e.g. `dog tired night → barks sleeps dark`.
- Pretraining: 140 records (all single words, all word pairs, 50 phrases)
- Training: 12 new phrases (pretraining data is replayed to avoid forgetting)
- Testing: 20 phrases never seen in any set → 20/20 correct after training
