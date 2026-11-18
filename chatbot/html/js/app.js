/* Python Chatbot – main page logic: i18n, buttons, POST /api/chatbot */
(function () {
  "use strict";

  const I18N = {
    en: {
      mini: "Mini chatbot", title: "Python Chatbot",
      subtitle: "Ask anything about Python. A local language model searches its knowledge and answers.",
      placeholder: "Type your Python question here, e.g. What basic types are supported in Python?",
      clearIn: "Clear input", clearOut: "Clear output", process: "Process",
      empty: "The answers will appear here.", you: "You", bot: "Chatbot",
      thinking: "Thinking…", emptyQ: "Please type a question first.",
      error: "Error", sources: "Sources", seconds: "s",
    },
    no: {
      mini: "Mini-chatbot", title: "Python-chatbot",
      subtitle: "Spør om hva som helst om Python. En lokal språkmodell søker i kunnskapen sin og svarer.",
      placeholder: "Skriv Python-spørsmålet ditt her, f.eks. Hvilke grunnleggende typer støttes i Python?",
      clearIn: "Tøm inndata", clearOut: "Tøm utdata", process: "Behandle",
      empty: "Svarene vises her.", you: "Du", bot: "Chatbot",
      thinking: "Tenker…", emptyQ: "Skriv et spørsmål først.",
      error: "Feil", sources: "Kilder", seconds: "s",
    },
    uk: {
      mini: "Міні-чатбот", title: "Python-чатбот",
      subtitle: "Запитуйте будь-що про Python. Локальна мовна модель шукає у своїх знаннях і відповідає.",
      placeholder: "Введіть питання про Python, напр. Які базові типи підтримуються в Python?",
      clearIn: "Очистити ввід", clearOut: "Очистити вивід", process: "Обробити",
      empty: "Тут з'являться відповіді.", you: "Ви", bot: "Чатбот",
      thinking: "Думаю…", emptyQ: "Спочатку введіть питання.",
      error: "Помилка", sources: "Джерела", seconds: "с",
    },
  };

  const $ = (s) => document.querySelector(s);
  const input = $("#input"), output = $("#output"), status = $("#status"), btnProcess = $("#btnProcess");
  let lang = "en";
  try { lang = localStorage.getItem("lang") || "en"; } catch (e) { /* storage blocked */ }
  if (!I18N[lang]) lang = "en";
  const t = (k) => I18N[lang][k] || I18N.en[k] || k;

  function applyLang(l) {
    lang = l;
    try { localStorage.setItem("lang", l); } catch (e) { /* ignore */ }
    document.documentElement.lang = l === "no" ? "nb" : l;
    document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
    document.querySelectorAll("[data-i18n-ph]").forEach((el) => { el.placeholder = t(el.dataset.i18nPh); });
    document.querySelectorAll(".flag").forEach((f) => f.classList.toggle("active", f.dataset.lang === l));
    document.title = t("title");
  }

  /* ---- very small, safe Markdown renderer (escape first, then format) ---- */
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  function inline(s) {
    return s
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>")
      .replace(/(^|\W)_([^_]+)_(?=\W|$)/g, "$1<em>$2</em>");
  }
  function markdown(md) {
    const parts = md.split(/```/);
    let html = "";
    parts.forEach((part, i) => {
      if (i % 2 === 1) { // code block
        const code = part.replace(/^[a-zA-Z0-9_+-]*\n/, "");
        html += "<pre><code>" + esc(code.replace(/\n$/, "")) + "</code></pre>";
        return;
      }
      let list = null;
      const close = () => { if (list) { html += `</${list}>`; list = null; } };
      esc(part).split("\n").forEach((line) => {
        let m;
        if ((m = line.match(/^\s*#{1,6}\s+(.*)/))) { close(); html += `<h4>${inline(m[1])}</h4>`; }
        else if ((m = line.match(/^\s*[-*+]\s+(.*)/))) {
          if (list !== "ul") { close(); html += "<ul>"; list = "ul"; }
          html += `<li>${inline(m[1])}</li>`;
        } else if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) {
          if (list !== "ol") { close(); html += "<ol>"; list = "ol"; }
          html += `<li>${inline(m[1])}</li>`;
        } else if (line.trim() === "") { close(); }
        else { close(); html += `<p>${inline(line)}</p>`; }
      });
      close();
    });
    return html;
  }

  function addMessage(kind, who, html, meta) {
    const ph = output.querySelector(".placeholder");
    if (ph) ph.remove();
    const div = document.createElement("div");
    div.className = "msg " + kind;
    div.innerHTML = `<div class="who"></div><div class="body">${html}</div>` + (meta ? `<div class="meta"></div>` : "");
    div.querySelector(".who").textContent = who;
    if (meta) div.querySelector(".meta").textContent = meta;
    output.appendChild(div);
    output.scrollTop = output.scrollHeight;
  }

  function clearOutput() {
    output.innerHTML = `<div class="placeholder" data-i18n="empty">${esc(t("empty"))}</div>`;
  }

  async function processQuestion() {
    const question = input.value.trim();
    if (!question) { input.focus(); status.textContent = t("emptyQ"); return; }
    addMessage("q", t("you"), "<p>" + esc(question).replace(/\n/g, "<br>") + "</p>");
    btnProcess.disabled = true;
    status.innerHTML = `<span class="dot"></span>${esc(t("thinking"))}`;
    try {
      const res = await fetch("/api/chatbot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, lang }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      const src = (data.sources || []).map((s) => s.title).join(", ");
      const meta = [data.model || data.mode, `${data.seconds} ${t("seconds")}`, src && `${t("sources")}: ${src}`]
        .filter(Boolean).join("  ·  ");
      addMessage("a", t("bot"), markdown(data.answer || ""), meta);
    } catch (e) {
      addMessage("a err", t("error"), "<p>" + esc(String(e.message || e)) + "</p>");
    } finally {
      btnProcess.disabled = false;
      status.textContent = "";
    }
  }

  $("#btnClearIn").addEventListener("click", () => { input.value = ""; input.focus(); });
  $("#btnClearOut").addEventListener("click", clearOutput);
  btnProcess.addEventListener("click", processQuestion);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) processQuestion();
  });
  document.querySelectorAll(".flag").forEach((f) => f.addEventListener("click", () => applyLang(f.dataset.lang)));

  applyLang(lang);
})();
