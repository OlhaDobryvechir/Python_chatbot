/*
 * minichat.js – visual demonstration of how a neural network learns.
 * Uses: createModel() from model.js, NN.Network from nn.js, math.js.
 */
(function () {
  "use strict";

  const model = createModel();
  const HP = model.hyper;
  const sizes = model.layers.map((l) => l.size);
  const net = new NN.Network(sizes, HP.seed, HP.activation);
  const L = sizes.length;                       // 5 layers
  const NS = "http://www.w3.org/2000/svg";
  const PATH_COLORS = ["#3fb950", "#d29922", "#a371f7", "#ff7b72", "#39c5cf", "#f778ba", "#79c0ff"];
  const NORM_SCALE = 4;                         // weight-norm that fills a node box completely

  const $ = (id) => document.getElementById(id);
  const svg = $("net"), logEl = $("log"), tip = $("tip"), statusEl = $("status");
  const buttons = ["btnClean", "btnPretrain", "btnTrain", "btnTest", "btnCleanLog", "btnProcess"].map($);

  const state = {
    pretrained: false, trained: false,
    acts: null,             // activations of the last forward pass (per layer)
    run: 0,                 // run token – incremented to cancel animations/training
    busy: false,
  };

  // =========================================================================
  //  Log
  // =========================================================================
  function log(text, cls) {
    const d = document.createElement("div");
    if (cls) d.className = cls;
    d.textContent = text;
    logEl.appendChild(d);
    while (logEl.childElementCount > 3000) logEl.firstChild.remove();
    logEl.scrollTop = logEl.scrollHeight;
  }
  const fmt = (v, d = 2) => (v >= 0 ? " " : "") + v.toFixed(d);
  const vec = (a, d = 2) => "[" + a.map((v) => v.toFixed(d)).join(" ") + "]";
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const frame = () => new Promise((r) => requestAnimationFrame(r));
  function setStatus(html) { statusEl.innerHTML = html; }

  // =========================================================================
  //  SVG drawing
  // =========================================================================
  let geo = null;      // geometry of all nodes
  let refs = null;     // references to svg elements for quick updates

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function nodeName(layer, i) {
    if (layer === 0) return model.inputWords[i].word;
    if (layer === L - 1) return model.outputWords[i].word;
    return `H${layer}.${i + 1}`;
  }

  function computeGeometry() {
    const W = svg.clientWidth, H = svg.clientHeight;
    const left = 92, right = 14, top = 26, bottom = 44;
    const rowH = 40;
    const gap = (H - top - bottom - rowH) / (L - 1);
    const rows = [];
    for (let k = 0; k < L; k++) {
      const n = sizes[k];
      const slot = (W - left - right) / n;
      const isWord = k === 0 || k === L - 1;
      const w = Math.max(18, Math.min(isWord ? 92 : 46, slot - 6));
      const h = isWord ? 38 : 30;
      const y = top + k * gap + (rowH - h) / 2;
      const nodes = [];
      for (let i = 0; i < n; i++) {
        const cx = left + slot * (i + 0.5);
        nodes.push({ x: cx - w / 2, y, w, h, cx, top: y, bottom: y + h });
      }
      rows.push({ y, h, nodes, labelY: top + k * gap + rowH / 2 });
    }
    return { W, H, rows };
  }

  function render() {
    geo = computeGeometry();
    svg.innerHTML = "";
    const defs = el("defs", {}, svg);
    const f = el("filter", { id: "glow", x: "-50%", y: "-50%", width: "200%", height: "200%" }, defs);
    el("feGaussianBlur", { stdDeviation: "2.5", result: "b" }, f);
    const m = el("feMerge", {}, f);
    el("feMergeNode", { in: "b" }, m); el("feMergeNode", { in: "SourceGraphic" }, m);
    PATH_COLORS.forEach((c, i) => {
      const mk = el("marker", { id: "ah" + i, viewBox: "0 0 10 10", refX: "9", refY: "5",
        markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" }, defs);
      el("path", { d: "M0,0 L10,5 L0,10 z", fill: c }, mk);
    });

    const gLinks = el("g", {}, svg), gLabels = el("g", {}, svg), gNodes = el("g", {}, svg);
    const gArrows = el("g", {}, svg);
    refs = { links: [], nodes: [], gArrows };

    // layer labels
    model.layers.forEach((ly, k) => {
      const r = geo.rows[k];
      el("text", { x: 10, y: r.labelY - 2, class: "layer-label" }, gLabels).textContent = `L${k + 1} ${ly.name}`;
      el("text", { x: 10, y: r.labelY + 11, class: "layer-sub" }, gLabels).textContent =
        k === 0 || k === L - 1 ? `${ly.size} words` : `${ly.size} nodes`;
      el("text", { x: 10, y: r.labelY + 23, class: "layer-sub" }, gLabels).textContent =
        k === 0 ? "one-hot" : k === L - 1 ? "sigmoid" : net.act.label;
    });

    // connections (every weight is a line)
    for (let k = 0; k < L - 1; k++) {
      const A = geo.rows[k].nodes, B = geo.rows[k + 1].nodes;
      const rows = [];
      for (let i = 0; i < A.length; i++) {
        const row = [];
        for (let j = 0; j < B.length; j++) {
          row.push(el("line", { x1: A[i].cx, y1: A[i].bottom, x2: B[j].cx, y2: B[j].top, class: "link" }, gLinks));
        }
        rows.push(row);
      }
      refs.links.push(rows);
    }

    // nodes
    for (let k = 0; k < L; k++) {
      const layerRefs = [];
      geo.rows[k].nodes.forEach((n, i) => {
        const g = el("g", { class: "node", "data-l": k, "data-i": i }, gNodes);
        const bg = el("rect", { x: n.x, y: n.y, width: n.w, height: n.h, rx: 6, class: "bg" }, g);
        const r = { g, bg };
        if (k === 0 || k === L - 1) {
          const w = k === 0 ? model.inputWords[i] : model.outputWords[i];
          if (k === L - 1) r.bar = el("rect", { x: n.x + 3, y: n.y + n.h - 5, width: 0, height: 3, rx: 1.5, class: "wbar" }, g);
          el("text", { x: n.cx, y: n.y + 16, class: "word" }, g).textContent = n.w < 50 ? w.word.slice(0, 4) : w.word;
          el("text", { x: n.cx, y: n.y + 29, class: "num" }, g).textContent = "#" + w.n;
        } else {
          r.bar = el("rect", { x: n.cx, y: n.y + n.h / 2, width: 0, height: 0, rx: 3, class: "wbar" }, g);
          r.val = el("text", { x: n.cx, y: n.y + n.h + 11, class: "val" }, g);
        }
        layerRefs.push(r);
      });
      refs.nodes.push(layerRefs);
    }
    updateWeights();
    if (state.acts) showActivations(state.acts);
  }

  /** Grow/shrink node rectangles and recolour all connection lines from the current weights. */
  function updateWeights() {
    if (!refs) return;
    for (let k = 1; k < L; k++) {
      refs.nodes[k].forEach((r, j) => {
        const info = net.nodeInfo(k, j);
        const n = geo.rows[k].nodes[j];
        const s = Math.min(1, info.norm / NORM_SCALE);
        const color = info.norm === 0 ? "#30363d" : info.mean >= 0 ? "#2ea043" : "#db6d28";
        if (k === L - 1) {               // output: horizontal bar at the bottom of the word box
          r.bar.setAttribute("width", Math.max(0, (n.w - 6) * s));
          r.bar.setAttribute("fill", color);
        } else {                          // hidden: a square growing from the centre
          const w = Math.max(0, (n.w - 6) * s), h = Math.max(0, (n.h - 6) * s);
          r.bar.setAttribute("x", n.cx - w / 2); r.bar.setAttribute("y", n.y + n.h / 2 - h / 2);
          r.bar.setAttribute("width", w); r.bar.setAttribute("height", h);
          r.bar.setAttribute("fill", color); r.bar.setAttribute("fill-opacity", 0.75);
          r.val.textContent = info.norm.toFixed(2);
        }
      });
    }
    for (let k = 0; k < L - 1; k++) {
      const W = net.W[k];
      refs.links[k].forEach((row, i) => row.forEach((line, j) => {
        const w = W[i][j], a = Math.abs(w);
        if (a < 1e-9) {
          line.setAttribute("stroke", "#30363d"); line.setAttribute("stroke-opacity", 0.12);
          line.setAttribute("stroke-width", 0.5);
        } else {
          line.setAttribute("stroke", w > 0 ? "#58a6ff" : "#f85149");
          line.setAttribute("stroke-opacity", Math.min(0.75, 0.03 + a / 3));
          line.setAttribute("stroke-width", Math.min(2.4, 0.4 + a * 0.6));
        }
      }));
    }
  }

  function clearActivations() {
    state.acts = null;
    if (!refs) return;
    refs.gArrows.innerHTML = "";
    refs.nodes.forEach((layer) => layer.forEach((r) => {
      r.g.classList.remove("on", "out-on");
      r.bg.style.fill = "";
    }));
  }

  /** Colour nodes by activation; `upTo` limits how many layers are lit (for animation). */
  function showActivations(acts, upTo = L - 1) {
    for (let k = 0; k <= Math.min(upTo, L - 1); k++) {
      const a = acts[k];
      const max = Math.max(1e-9, ...a.map(Math.abs));
      refs.nodes[k].forEach((r, i) => {
        const v = a[i];
        if (k === L - 1) {
          r.bg.style.fill = `rgba(46,160,67,${(v * 0.55).toFixed(3)})`;
          r.g.classList.toggle("out-on", v > HP.threshold);
        } else {
          const rel = Math.abs(v) / max;
          r.bg.style.fill = v > 0 ? `rgba(88,166,255,${(rel * 0.6).toFixed(3)})` : `rgba(248,81,73,${(rel * 0.35).toFixed(3)})`;
          r.g.classList.toggle("on", k === 0 ? v > 0 : rel > 0.55 && v > 0);
        }
      });
    }
  }

  /**
   * Trace how the input words reach each predicted output word: for every output
   * word, follow backwards the strongest positive contributions a_i·W_ij.
   * Returns edges grouped by layer step 0..3.
   */
  function tracePaths(acts, outIdx) {
    const steps = [[], [], [], []];
    const seen = new Set();
    outIdx.forEach((o, ci) => {
      let frontier = [o];
      for (let k = L - 1; k >= 1; k--) {
        const next = new Set();
        frontier.forEach((j) => {
          const contrib = acts[k - 1].map((a, i) => ({ i, c: a * net.W[k - 1][i][j] })).filter((x) => x.c > 0);
          contrib.sort((p, q) => q.c - p.c);
          if (!contrib.length) return;
          const keep = contrib.filter((x, idx) => idx < (k === 1 ? 3 : 2) && x.c > contrib[0].c * 0.25);
          keep.forEach(({ i }) => {
            const key = `${k - 1}:${i}:${j}:${ci}`;
            if (!seen.has(key)) { seen.add(key); steps[k - 1].push({ i, j, color: ci % PATH_COLORS.length }); }
            next.add(i);
          });
        });
        frontier = [...next];
      }
    });
    return steps;
  }

  async function animateForward(acts, token, stepDelay = 450) {
    clearActivations();
    state.acts = acts;
    const P = acts[L - 1];
    let outIdx = P.map((p, i) => i).filter((i) => P[i] > HP.threshold);
    if (!outIdx.length) outIdx = [P.indexOf(Math.max(...P))];
    const steps = tracePaths(acts, outIdx);
    showActivations(acts, 0);
    for (let k = 0; k < L - 1; k++) {
      await sleep(stepDelay);
      if (token !== state.run) return;
      steps[k].forEach(({ i, j, color }) => {
        const a = geo.rows[k].nodes[i], b = geo.rows[k + 1].nodes[j];
        const y1 = a.bottom + 1, y2 = b.top - 2, my = (y1 + y2) / 2;
        const p = el("path", {
          d: `M${a.cx},${y1} C${a.cx},${my} ${b.cx},${my} ${b.cx},${y2}`,
          class: "arrow", stroke: PATH_COLORS[color], opacity: 0.9,
        }, refs.gArrows);
        const len = Math.ceil(p.getTotalLength()) + 2;    // dash animation sized to this path
        p.style.strokeDasharray = len; p.style.strokeDashoffset = len;
        // the arrow head appears when the line has finished drawing
        p.addEventListener("animationend", () => p.setAttribute("marker-end", `url(#ah${color})`), { once: true });
      });
      showActivations(acts, k + 1);
    }
  }

  // =========================================================================
  //  Tooltip
  // =========================================================================
  function tipHtml(k, i) {
    const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
    let h = "";
    const act = state.acts ? state.acts[k][i] : null;
    if (k === 0) {
      const w = model.inputWords[i];
      h += `<div class="t">Input word #${w.n}: “${esc(w.word)}”</div>`;
      h += `<div><span class="k">layer:</span> L1 Input · <span class="k">group:</span> ${w.group}</div>`;
      h += `<div><span class="k">one-hot position:</span> x[${i}] = 1 when the word is typed</div>`;
      if (act !== null) h += `<div><span class="k">current value:</span> ${act}</div>`;
      return h;
    }
    const info = net.nodeInfo(k, i);
    if (k === L - 1) {
      const w = model.outputWords[i];
      h += `<div class="t">Output word #${w.n}: “${esc(w.word)}”</div>`;
      h += `<div><span class="k">layer:</span> L5 Output · <span class="k">group:</span> ${w.group}</div>`;
      if (act !== null) h += `<div><span class="k">probability:</span> ${act.toFixed(4)} ${act > HP.threshold ? "✔ chosen" : ""}</div>`;
    } else {
      h += `<div class="t">Node ${nodeName(k, i)} (#${i + 1})</div>`;
      h += `<div><span class="k">layer:</span> L${k + 1} ${model.layers[k].name} · ${net.act.label}</div>`;
      if (act !== null) h += `<div><span class="k">activation:</span> ${act.toFixed(4)}</div>`;
    }
    h += `<div><span class="k">bias:</span> ${info.bias.toFixed(4)} · <span class="k">‖w‖:</span> ${info.norm.toFixed(3)}</div>`;
    const ws = info.incoming.map((v, s) => ({ v, s })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
    h += `<div class="k">incoming weights (${ws.length}, strongest first):</div><table>`;
    ws.slice(0, 10).forEach(({ v, s }) => {
      h += `<tr><td>${esc(nodeName(k - 1, s))}</td><td class="${v >= 0 ? "p" : "n"}">${fmt(v, 3)}</td></tr>`;
    });
    if (ws.length > 10) h += `<tr><td class="k" colspan="2">… ${ws.length - 10} more</td></tr>`;
    return h + "</table>";
  }

  svg.addEventListener("mousemove", (e) => {
    const g = e.target.closest && e.target.closest(".node");
    if (!g) { tip.hidden = true; return; }
    tip.innerHTML = tipHtml(+g.dataset.l, +g.dataset.i);
    tip.hidden = false;
    const r = tip.getBoundingClientRect();
    let x = e.clientX + 14, y = e.clientY + 14;
    if (x + r.width > innerWidth - 4) x = e.clientX - r.width - 14;
    if (y + r.height > innerHeight - 4) y = e.clientY - r.height - 14;
    tip.style.left = Math.max(4, x) + "px"; tip.style.top = Math.max(4, y) + "px";
  });
  svg.addEventListener("mouseleave", () => { tip.hidden = true; });

  // =========================================================================
  //  Actions
  // =========================================================================
  function setBusy(busy, activeBtn) {
    state.busy = busy;
    buttons.forEach((b) => {
      if (b.id === "btnClean" || b.id === "btnCleanLog") return;   // always available
      b.disabled = busy;
    });
    buttons.forEach((b) => b.classList.remove("running"));
    if (busy && activeBtn) { activeBtn.disabled = false; activeBtn.classList.add("running"); }
  }

  const predictedWords = (p) => model.outputWords.filter((w, j) => p[j] > HP.threshold).map((w) => w.word);
  const sameSet = (a, b) => a.slice().sort().join() === b.slice().sort().join();
  const accuracy = (set) => set.filter((r) => sameSet(predictedWords(net.predict(r.x)[L - 1]), r.outWords)).length;

  function cleanUp(silent) {
    state.run++;
    setBusy(false);
    net.zero();
    state.pretrained = state.trained = false;
    clearActivations();
    updateWeights();
    setStatus("<b>clean</b> · all weights = 0");
    if (silent) return;
    log("CLEAN UP", "h");
    log("Hidden layers L2–L4: every weight and bias set to 0.", "op");
    log("Output layer L5: weights and biases set to 0.", "op");
    log(`Input layer L1 initialised with ${model.inputWords.length} words: ` +
      model.inputWords.map((w) => `#${w.n} ${w.word}`).join(", "), "dim");
    log(`Output layer L5 initialised with ${model.outputWords.length} words: ` +
      model.outputWords.map((w) => `#${w.n} ${w.word}`).join(", "), "dim");
  }

  function ensureInitialised() {
    if (!net.isZero()) return;
    net.randomize(HP.seed);
    log("Weights are all 0 → every neuron of a layer would compute the same thing and learn the same thing. " +
      "Breaking the symmetry: Xavier random init W ~ U(−√(6/(in+out)), +√(6/(in+out))), biases = 0.", "warn");
    updateWeights();
  }

  /** Detailed log of the matrix operations of one epoch (shown once per phase). */
  function logOperations(X, lr) {
    const n = X.length;
    log(`Matrix operations of one epoch (n = ${n} records):`, "op");
    for (let k = 1; k < L; k++) {
      const f = k === L - 1 ? "σ (sigmoid)" : net.act.label;
      log(` Z${k} = A${k - 1}·W${k} + b${k}  (${n}×${sizes[k - 1]})·(${sizes[k - 1]}×${sizes[k]}) → ${n}×${sizes[k]}`, "op");
      log(` A${k} = ${f}(Z${k})`, "op");
    }
    log(` Loss = −mean[Y·ln P + (1−Y)·ln(1−P)]  (binary cross-entropy, P = A4)`, "op");
    log(` δ4 = (P − Y)/n`, "op");
    for (let k = L - 1; k >= 1; k--) {
      log(` dW${k} = A${k - 1}ᵀ·δ${k}  (${sizes[k - 1]}×${n})·(${n}×${sizes[k]})   db${k} = Σrows δ${k}`, "op");
      if (k > 1) log(` δ${k - 1} = (δ${k}·W${k}ᵀ) ⊙ f′(Z${k - 1})`, "op");
    }
    log(` Adam: W ← W − ${lr}·m̂/(√v̂+ε) − ${lr}·${HP.weightDecay}·W`, "op");
  }

  async function runTraining(kind, records, epochs, lr, btn) {
    const token = ++state.run;
    setBusy(true, btn);
    clearActivations();
    const X = records.map((r) => r.x), Y = records.map((r) => r.y);
    logOperations(X, lr);

    // show the first record going through the network numerically
    const s = records[0];
    const acts0 = net.predict(s.x);
    log(`Example record: "${s.text}"`, "dim");
    log(` x = ${vec(s.x, 0)}`, "dim");
    log(` A1 = ${vec(acts0[1])}`, "dim");
    log(` P = ${vec(acts0[L - 1])}`, "dim");
    log(` y = ${vec(s.y, 0)}`, "dim");

    const t0 = performance.now();
    const logEvery = Math.max(10, Math.round(epochs / 16));
    let st = null;
    for (let e = 1; e <= epochs;) {
      const tf = performance.now();
      do {                                 // as many epochs as fit into ~25 ms, then redraw
        st = net.trainEpoch(X, Y, lr, HP.weightDecay);
        if (e === 1 || e % logEvery === 0 || e === epochs) {
          const acc = accuracy(records);
          log(`ep ${String(e).padStart(4)}  loss ${st.loss.toFixed(4)}  Δw ${st.change.toFixed(3)}  ok ${acc}/${records.length}`,
            acc === records.length ? "ok" : "");
        }
        e++;
      } while (e <= epochs && performance.now() - tf < 25);
      updateWeights();
      setStatus(`<b>${kind}</b> · epoch ${Math.min(e - 1, epochs)}/${epochs} · loss <b>${st.loss.toFixed(4)}</b>`);
      await frame();
      if (token !== state.run) { log(`${kind} interrupted.`, "warn"); return false; }
    }
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    log(`${kind} finished: ${epochs} epochs in ${secs}s, final loss ${st.loss.toFixed(5)}.`, "ok");
    setBusy(false);
    return true;
  }

  async function pretraining() {
    if (state.busy) { state.run++; setBusy(false); return; }        // click again = stop
    log("PRETRAINING", "h");
    log(`Large set: ${model.pretrain.length} records (all single words, all word pairs, 50 phrases). ` +
      "Like a language model reading lots of text: it learns which words go together.", "dim");
    ensureInitialised();
    const ok = await runTraining("pretraining", model.pretrain, HP.pretrain.epochs, HP.pretrain.lr, $("btnPretrain"));
    if (!ok) return;
    state.pretrained = true;
    log(`Pretraining set: ${accuracy(model.pretrain)}/${model.pretrain.length} correct. ` +
      `Unseen test phrases already: ${accuracy(model.test)}/${model.test.length}.`, "ok");
    setStatus(`<b>pretrained</b> · loss ${net.loss(net.forward(model.pretrain.map((r) => r.x)).A[L - 1], model.pretrain.map((r) => r.y)).toFixed(4)}`);
  }

  async function training() {
    if (state.busy) { state.run++; setBusy(false); return; }
    log("TRAINING", "h");
    log(`Small set: ${model.train.length} full phrases (fine-tuning on the real task).`, "dim");
    if (!state.pretrained) log("Note: no pretraining yet – the network has to learn from only 12 examples, so it will " +
      "memorise them and generalise badly. Try Clean up → Pretraining → Training.", "warn");
    const before = accuracy(model.train);
    log(`Before training: ${before}/${model.train.length} of these phrases already correct.`, "dim");
    ensureInitialised();
    let records = model.train;
    if (HP.train.replay && state.pretrained) {
      records = [...model.train, ...model.pretrain];
      log(`Replay: the ${model.pretrain.length} pretraining records are mixed in, so the network does not ` +
        `"forget" them (catastrophic forgetting).`, "dim");
    }
    const ok = await runTraining("training", records, HP.train.epochs, HP.train.lr, $("btnTrain"));
    if (!ok) return;
    state.trained = true;
    log(`Training set: ${accuracy(model.train)}/${model.train.length} correct.`, "ok");
    setStatus("<b>trained</b> · press Testing or type words and Process");
  }

  async function testing() {
    if (state.busy) { state.run++; setBusy(false); return; }
    const token = ++state.run;
    setBusy(true, $("btnTest"));
    log("TESTING", "h");
    log(`${model.test.length} phrases that never appeared in pretraining or training.`, "dim");
    if (net.isZero()) log("The network is empty (all weights 0): every output is σ(0) = 0.5, nothing is chosen.", "warn");
    let ok = 0;
    for (let t = 0; t < model.test.length; t++) {
      const r = model.test[t];
      const acts = net.predict(r.x);
      const words = predictedWords(acts[L - 1]);
      const good = sameSet(words, r.outWords);
      ok += good;
      setStatus(`<b>testing</b> · ${t + 1}/${model.test.length} · correct ${ok}`);
      log(`${t + 1}. ${r.inWords.join(" ")}`, "");
      log(`   → ${words.join(" ") || "(nothing)"} ${good ? "✔" : "✘ expected: " + r.outWords.join(" ")}`, good ? "ok" : "err");
      await animateForward(acts, token, 260);
      if (token !== state.run) { log("Testing interrupted.", "warn"); return; }
      await sleep(650);
      if (token !== state.run) { log("Testing interrupted.", "warn"); return; }
    }
    const pct = Math.round((100 * ok) / model.test.length);
    log(`Result: ${ok}/${model.test.length} correct (${pct}%) on new combinations.`, ok === model.test.length ? "ok" : "warn");
    if (ok === model.test.length) log("The network learned the meaning of every word – not a lookup table.", "ok");
    setStatus(`<b>tested</b> · ${ok}/${model.test.length} correct`);
    setBusy(false);
  }

  async function processInput() {
    if (state.busy) return;
    const text = $("userInput").value.trim();
    log("PROCESS", "h");
    if (!text) { log("Type some words first, e.g. “cat happy garden”.", "warn"); return; }
    const enc = model.encodeText(text);
    log(`Input: "${text}"`, "");
    if (enc.unknown.length) log(`Unknown words ignored: ${enc.unknown.join(", ")}`, "warn");
    if (!enc.known.length) {
      log("No known words. Vocabulary: " + model.inputWords.map((w) => w.word).join(", "), "err");
      return;
    }
    const token = ++state.run;
    const acts = net.predict(enc.x);
    log(`L1 input (one-hot): ${enc.known.map((w) => `#${model.inputWords.find((x) => x.word === w).n} ${w}`).join(", ")}`, "op");
    log(` x = ${vec(enc.x, 0)}`, "dim");
    for (let k = 1; k < L - 1; k++) {
      const top = acts[k].map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v).slice(0, 4);
      log(`L${k + 1} = f(L${k}·W${k}+b${k}) top: ` + top.map(({ v, i }) => `${nodeName(k, i)}=${v.toFixed(2)}`).join(" "), "op");
    }
    const P = acts[L - 1];
    const ranked = P.map((p, j) => ({ p, w: model.outputWords[j].word })).sort((a, b) => b.p - a.p);
    log(`L5 = σ(L4·W4+b4): ` + ranked.slice(0, 5).map((r) => `${r.w} ${r.p.toFixed(3)}`).join(", "), "op");
    const words = predictedWords(P);
    log(`Output (p > ${HP.threshold}): ${words.join(" ") || "(nothing)"}`, words.length ? "ok" : "warn");
    if (words.length) log(`“${model.sentence(enc.known, words)}”`, "ok");
    if (net.isZero()) log("The network is empty – press Pretraining first.", "warn");
    setStatus(`<b>processed</b> · ${words.join(" ") || "no output"}`);
    await animateForward(acts, token, 450);
  }

  // =========================================================================
  //  Wiring
  // =========================================================================
  $("btnClean").addEventListener("click", () => cleanUp(false));
  $("btnPretrain").addEventListener("click", pretraining);
  $("btnTrain").addEventListener("click", training);
  $("btnTest").addEventListener("click", testing);
  $("btnCleanLog").addEventListener("click", () => { logEl.innerHTML = ""; });
  $("btnProcess").addEventListener("click", processInput);

  // =========================================================================
  //  Download model: everything Process uses, as final_model.json
  //  (no training data, no optimiser state – see final_model.md)
  // =========================================================================
  function buildModelJson() {
    const r8 = (v) => { const x = Math.round(v * 1e8) / 1e8; return x === 0 ? 0 : x; };   // no "-0"
    const labels = (k) => Array.from({ length: sizes[k] }, (_, i) => nodeName(k, i));
    const lname = (k) => `L${k + 1} ${model.layers[k].name}`;
    const lrelu = HP.activation === "lrelu";
    const params = sizes.slice(1).reduce((s, n, k) => s + sizes[k] * n + n, 0);

    return {
      format: "mini-chatbot-final-model",
      version: 1,
      created: new Date().toISOString(),
      status: {
        pretrained: state.pretrained,
        trained: state.trained,
        optimizerSteps: net.t,
        emptyNetwork: net.isZero(),
      },
      architecture: {
        parameterCount: params,
        layers: model.layers.map((ly, k) => ({
          layer: `L${k + 1}`,
          name: ly.name,
          size: ly.size,
          activation: k === 0 ? "none (input vector)" : k === L - 1 ? "sigmoid" : net.act.label,
          nodes: labels(k),
        })),
      },
      processing: {
        tokenizer: {
          lowercase: true,
          splitOn: "[^a-z]+",
          pluralRule: "a token that is not in the vocabulary but ends with 's' is tried without the 's' (cats -> cat)",
          unknownWords: "ignored",
          duplicateWords: "counted once",
        },
        inputEncoding: "x[i] = 1 if input word number i+1 is in the phrase, otherwise 0",
        forwardPass: [
          "a0 = x",
          "z_k = a_(k-1) · W_k + b_k   for k = 1..4",
          "a_k = hiddenActivation(z_k) for k = 1..3",
          "p = a4 = outputActivation(z4)",
        ],
        hiddenActivation: lrelu
          ? { name: "leaky ReLU", formula: "f(z) = z if z > 0, otherwise negativeSlope · z", negativeSlope: 0.01 }
          : { name: "tanh", formula: "f(z) = tanh(z)" },
        outputActivation: { name: "sigmoid", formula: "p = 1 / (1 + e^(-z))" },
        threshold: HP.threshold,
        selectionRule: `an output word is chosen when p > ${HP.threshold}; every word is decided independently`,
        sentence: {
          subject: "'The <animal>' if an animal word was typed, otherwise 'It'",
          order: ["sound", "action", "condition"],
          conditionPhrases: model.conditionPhrases,
          empty: "(no confident output)",
        },
      },
      vocabulary: {
        input: model.inputWords.map(({ n, word, group }) => ({ n, word, group })),
        output: model.outputWords.map(({ n, word, group }) => ({ n, word, group })),
      },
      connections: net.W.map((W, k) => ({
        name: `W${k + 1}, b${k + 1}`,
        from: lname(k),
        to: lname(k + 1),
        shape: [sizes[k], sizes[k + 1]],
        columns: labels(k + 1),
        weights: Object.fromEntries(W.map((row, i) => [nodeName(k, i), row.map(r8)])),
        biases: net.b[k].map(r8),
      })),
    };
  }

  /** JSON with indent 2, but short arrays / small flat objects kept on one line. */
  function prettyJson(obj) {
    const prim = String.raw`(?:-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|"[^"\n]*"|true|false|null)`;
    return JSON.stringify(obj, null, 2)
      .replace(new RegExp(String.raw`\[\n\s*(${prim}(?:,\n\s*${prim})*)\n\s*\]`, "g"),
        (m, inner) => {
          const items = inner.split(/,\n\s*/);
          const one = "[" + items.join(", ") + "]";
          return items.every((x) => !x.startsWith('"') || x.length <= 24) ? one : m;   // keep long text lists
        })
      .replace(new RegExp(String.raw`\{\n\s*("[^"\n]+": ${prim}(?:,\n\s*"[^"\n]+": ${prim})*)\n\s*\}`, "g"),
        (m, inner) => { const one = "{ " + inner.split(/,\n\s*/).join(", ") + " }"; return one.length <= 80 ? one : m; });
  }

  function downloadModel() {
    const json = prettyJson(buildModelJson());
    const url = URL.createObjectURL(new Blob([json + "\n"], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url; a.download = "final_model.json";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    log("DOWNLOAD MODEL", "h");
    log(`final_model.json: ${(json.length / 1024).toFixed(1)} KB – vocabulary, processing parameters, ` +
      `${net.W.length} weight matrices and biases.`, "ok");
    if (net.isZero()) log("Note: the network is empty (all weights 0) – press Pretraining first.", "warn");
  }

  // =========================================================================
  //  Upload model: load final_model.json (as written by Download model)
  // =========================================================================
  const ACT_BY_NAME = { "leaky ReLU": "lrelu", "tanh": "tanh" };

  /** Check the file completely; returns { W, b, act, threshold, status } or throws a readable Error. */
  function parseModelJson(m) {
    const fail = (msg) => { throw new Error(msg); };
    if (!m || typeof m !== "object") fail("the file is not a JSON object");
    if (m.format !== "mini-chatbot-final-model") fail(`unknown format "${m.format}" – expected "mini-chatbot-final-model"`);
    if (m.version !== 1) fail(`unsupported version ${m.version} – expected 1`);

    // vocabulary must match: the meaning of every input/output position is fixed by model.js
    ["input", "output"].forEach((side) => {
      const mine = side === "input" ? model.inputWords : model.outputWords;
      const theirs = m.vocabulary && m.vocabulary[side];
      if (!Array.isArray(theirs) || theirs.length !== mine.length ||
          theirs.some((w, i) => !w || w.word !== mine[i].word)) {
        fail(`the ${side} vocabulary differs from this page's (${mine.map((w) => w.word).join(", ")})`);
      }
    });

    const P = m.processing || {};
    const ha = P.hiddenActivation || {};
    const act = ACT_BY_NAME[ha.name];
    if (!act) fail(`unsupported hidden activation "${ha.name}"`);
    if (act === "lrelu" && ha.negativeSlope !== 0.01) fail(`leaky ReLU slope ${ha.negativeSlope} is not supported (only 0.01)`);
    if (P.outputActivation && P.outputActivation.name !== "sigmoid") fail(`unsupported output activation "${P.outputActivation.name}"`);
    const threshold = P.threshold === undefined ? HP.threshold : P.threshold;
    if (typeof threshold !== "number" || !(threshold > 0 && threshold < 1)) fail(`threshold must be a number between 0 and 1`);

    const C = m.connections;
    if (!Array.isArray(C) || C.length !== L - 1) fail(`expected ${L - 1} connections, found ${Array.isArray(C) ? C.length : 0}`);
    const isNum = (v) => typeof v === "number" && isFinite(v);
    const W = [], b = [];
    C.forEach((c, k) => {
      const where = `connection ${k + 1} (W${k + 1}, ${sizes[k]}×${sizes[k + 1]})`;
      if (!c || !c.weights || typeof c.weights !== "object") fail(`${where}: "weights" is missing`);
      const rows = [];
      for (let i = 0; i < sizes[k]; i++) {
        const name = nodeName(k, i), row = c.weights[name];
        if (!Array.isArray(row)) fail(`${where}: row "${name}" is missing`);
        if (row.length !== sizes[k + 1] || !row.every(isNum)) fail(`${where}: row "${name}" must have ${sizes[k + 1]} numbers`);
        rows.push(row.slice());
      }
      const extra = Object.keys(c.weights).length - sizes[k];
      if (extra > 0) fail(`${where}: ${extra} unexpected row(s)`);
      if (!Array.isArray(c.biases) || c.biases.length !== sizes[k + 1] || !c.biases.every(isNum)) {
        fail(`${where}: "biases" must have ${sizes[k + 1]} numbers`);
      }
      W.push(rows); b.push(c.biases.slice());
    });
    return { W, b, act, threshold, status: m.status || {} };
  }

  function applyModel(p, fileName) {
    state.run++;                         // stop a running training/testing/animation
    setBusy(false);
    net.W = p.W; net.b = p.b;
    net.act = NN.ACT[p.act];
    HP.threshold = p.threshold;
    net.resetOptimizer();                // Adam memory is not stored in the file → starts fresh
    net.initialized = true;
    state.pretrained = !!p.status.pretrained;
    state.trained = !!p.status.trained;
    clearActivations();
    render();                            // redraw everything (activation label may have changed)

    log("UPLOAD MODEL", "h");
    log(`Loaded ${fileName}: ${p.W.length} weight matrices and biases, ` +
      `${net.act.label}, threshold ${p.threshold}.`, "ok");
    log(`Saved state: pretrained ${state.pretrained ? "yes" : "no"}, trained ${state.trained ? "yes" : "no"}` +
      (p.status.created ? `, created ${p.status.created}` : "") + ".", "dim");
    if (net.isZero()) log("This model is empty (all weights 0) – it will not choose any word.", "warn");
    log("Further Pretraining/Training continues from these weights with a fresh Adam optimiser.", "dim");
    setStatus(`<b>loaded</b> · ${fileName.replace(/[&<>]/g, "")}`);
  }

  const fileInput = $("modelFile");
  fileInput.addEventListener("change", async () => {
    const file = fileInput.files[0];
    fileInput.value = "";                // so the same file can be chosen again
    if (!file) return;
    try {
      const m = JSON.parse(await file.text());
      const p = parseModelJson(m);
      p.status.created = m.created;
      applyModel(p, file.name);
    } catch (err) {
      log("UPLOAD MODEL", "h");
      log(`${file.name} was not loaded: ${err instanceof SyntaxError ? "invalid JSON – " : ""}${err.message}`, "err");
      log("The current network is unchanged.", "dim");
    }
  });

  // "⋮" menu: Pretraining / Training / Testing Math popups (mathview.js) + Download / Upload model
  MathView.init({
    net, model, tracePaths, pathColors: PATH_COLORS,
    getText: () => $("userInput").value,
    getState: () => state,
    download: downloadModel,
    upload: () => fileInput.click(),
  });
  $("userInput").addEventListener("keydown", (e) => { if (e.key === "Enter") processInput(); });

  // =========================================================================
  //  Input overlay: on blur, show the phrase with unknown words in orange
  // =========================================================================
  const inputEl = $("userInput"), viewEl = $("userInputView");

  /** A token is known if the input layer would accept it (same rules as model.encodeText). */
  const isKnownWord = (t) => model.encodeText(t).known.length > 0;

  function showInputView() {
    const text = inputEl.value;
    if (!text.trim()) { viewEl.hidden = true; return; }   // keep the placeholder visible

    // place the div exactly over the inner part of the input (inside border, same padding & font)
    const cs = getComputedStyle(inputEl);
    const bl = parseFloat(cs.borderLeftWidth), bt = parseFloat(cs.borderTopWidth);
    Object.assign(viewEl.style, {
      left: inputEl.offsetLeft + bl + "px",
      top: inputEl.offsetTop + bt + "px",
      width: inputEl.clientWidth + "px",
      height: inputEl.clientHeight + "px",
      lineHeight: inputEl.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom) + "px",
      padding: cs.padding,
      font: cs.font, letterSpacing: cs.letterSpacing,
      borderRadius: cs.borderRadius,
      textIndent: -inputEl.scrollLeft + "px",       // same horizontal scroll as the input
    });

    // words (letters only, as the tokenizer sees them) are coloured; separators are kept as typed
    viewEl.innerHTML = "";
    text.split(/([A-Za-z]+)/).forEach((part, i) => {
      if (!part) return;
      if (i % 2 === 0) { viewEl.appendChild(document.createTextNode(part)); return; }
      const s = document.createElement("span");
      s.className = isKnownWord(part) ? "kw" : "uw";
      s.textContent = part;
      viewEl.appendChild(s);
    });
    viewEl.hidden = false;
  }

  inputEl.addEventListener("blur", showInputView);
  inputEl.addEventListener("focus", () => { viewEl.hidden = true; });   // e.g. focused with Tab
  viewEl.addEventListener("click", () => { viewEl.hidden = true; inputEl.focus(); });
  // the input may change width with the window – re-place the overlay
  new ResizeObserver(() => { if (!viewEl.hidden) showInputView(); }).observe(inputEl);

  let resizeTimer = null;
  new ResizeObserver(() => { clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 80); }).observe($("svgWrap"));

  render();
  cleanUp(true);
  log("MINI CHATBOT", "h");
  log("A 5-layer neural network: 15 input words → 3 hidden layers → 15 output words.", "dim");
  log("Steps: Clean up → Pretraining → Training → Testing, or type words + Process.", "dim");
  log("Hover a node to see its name, number and weights.", "dim");
  log("Try: “dog tired night”, “fish happy garden”, “cow scared snow”.", "dim");
})();
