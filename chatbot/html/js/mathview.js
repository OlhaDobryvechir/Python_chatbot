/*
 * mathview.js – the "⋮" menu of the Mini chatbot and its full-screen
 * "Pretraining Math", "Training Math" and "Testing Math" explanations.
 *
 * Every number shown is computed from the current network:
 *   - Pretraining / Training: one optimisation step is executed on a COPY of the
 *     network (the real network is never changed), exactly as nn.js does it.
 *   - Testing: the forward pass of the phrase typed in the input field.
 *
 * Uses: math.js (global `math`), NN from nn.js. Initialised from minichat.js:
 *   MathView.init({ net, model, tracePaths, pathColors, getText, getState })
 */
(function (root) {
  "use strict";

  const M = root.math;
  const B1 = 0.9, B2 = 0.999, EPS = 1e-8;       // Adam constants (same as nn.js)
  const $ = (id) => document.getElementById(id);
  let ctx, model, HP, sizes, L, LABELS;

  // =========================================================================
  //  Formatting helpers
  // =========================================================================
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  function f(v, d = 4) {
    if (v === 0) return "0";
    if (!isFinite(v)) return String(v);
    const a = Math.abs(v);
    if (a < 1e-4 || a >= 1e5) return v.toExponential(2);
    return v.toFixed(d);
  }
  const fp = (v, d) => (v < 0 ? `(${f(v, d)})` : f(v, d));           // value inside a sum
  const sg = (v, d) => (v >= 0 ? "+" : "") + f(v, d);                  // explicit sign
  const pct = (r) => Math.round(r * 100) + "%";
  const pad = (s, n) => String(s).padEnd(n);
  const maxAbs = (a) => Math.max(1e-12, ...a.flat().map(Math.abs));
  const argmax = (v) => v.reduce((bi, x, i) => (x > v[bi] ? i : bi), 0);
  const argmaxAbs = (v) => v.reduce((bi, x, i) => (Math.abs(x) > Math.abs(v[bi]) ? i : bi), 0);
  function argmaxAbs2(A) {
    let bi = 0, bj = 0;
    A.forEach((r, i) => r.forEach((v, j) => { if (Math.abs(v) > Math.abs(A[bi][bj])) { bi = i; bj = j; } }));
    return [bi, bj];
  }
  const colSum = (A) => A[0].map((_, j) => A.reduce((s, r) => s + r[j], 0));
  const sig = (z) => 1 / (1 + Math.exp(-z));
  const lbl = (k, i) => LABELS[k][i];
  const layName = (k) => `L${k + 1} ${model.layers[k].name}`;
  const S = (name, s) => `${name}<sub>${s}</sub>`;

  function tint(v, scale) {
    if (!v) return "";
    const a = (Math.min(1, Math.abs(v) / scale) * 0.5).toFixed(3);
    return ` style="background:${v > 0 ? `rgba(56,139,253,${a})` : `rgba(248,81,73,${a})`}"`;
  }

  /** A matrix as a table: rows × columns with labels, cells tinted by sign and size. */
  function matrix(A, rowL, colL, o = {}) {
    const sc = o.scale || maxAbs(A), hr = o.hlRows || new Set(), hc = o.hlCols || new Set();
    let h = `<div class="mt"><table class="mx"><tr><th class="corner">${esc(o.corner || "")}</th>`;
    colL.forEach((c, j) => { h += `<th${hc.has(j) ? ' class="hl"' : ""}>${esc(c)}</th>`; });
    h += "</tr>";
    A.forEach((row, i) => {
      h += `<tr${hr.has(i) ? ' class="hl"' : ""}><th>${esc(rowL[i])}</th>`;
      row.forEach((v, j) => { h += `<td${hc.has(j) ? ' class="hc"' : ""}${tint(v, sc)}>${f(v, o.d)}</td>`; });
      h += "</tr>";
    });
    return h + "</table></div>";
  }
  const vector = (v, colL, name, o = {}) => matrix([v], [name], colL, o);
  const fold = (summary, inner, open) => `<details${open ? " open" : ""}><summary>${summary}</summary>${inner}</details>`;
  const formula = (s) => `<div class="f">${s}</div>`;
  function table(headers, rows) {
    return `<div class="mt"><table class="nt"><tr>${headers.map((x) => `<th>${x}</th>`).join("")}</tr>` +
      rows.map((r) => `<tr${r.cls ? ` class="${r.cls}"` : ""}>${(r.cells || r).map((c) => `<td>${c}</td>`).join("")}</tr>`).join("") +
      "</table></div>";
  }
  const yes = (b) => (b ? '<span class="yes">✔</span>' : '<span class="no">–</span>');

  function actText() {
    return HP.activation === "lrelu"
      ? { f: "f(z) = z if z > 0, otherwise 0.01·z   (leaky ReLU)", d: "f′(z) = 1 if z > 0, otherwise 0.01",
          how: (z) => (z > 0 ? "z" : "0.01·z") }
      : { f: "f(z) = tanh(z)", d: "f′(z) = 1 − tanh²(z) = 1 − a²", how: () => "tanh(z)" };
  }

  // =========================================================================
  //  Network maths (copies nn.js, but keeps every intermediate result)
  // =========================================================================
  function copyNet(src) {
    const c = new NN.Network(src.sizes, src.seed, HP.activation);
    const dc = (x) => JSON.parse(JSON.stringify(x));
    c.W = dc(src.W); c.b = dc(src.b);
    c.mW = dc(src.mW); c.vW = dc(src.vW); c.mb = dc(src.mb); c.vb = dc(src.vb);
    c.t = src.t; c.initialized = src.initialized; c.epochsDone = src.epochsDone;
    return c;
  }

  /** The network the next Pretraining/Training press would start from (as a copy). */
  function startNet() {
    const net = copyNet(ctx.net);
    const wasZero = net.isZero();
    if (wasZero) net.randomize(HP.seed);           // same as ensureInitialised() in minichat.js
    return { net, wasZero };
  }

  /** Forward + backward pass without updating: A, Z, δ, δ·Wᵀ, f′(Z), gradients, loss. */
  function passInfo(net, X, Y) {
    const K = net.W.length, n = X.length;
    const { A, Z } = net.forward(X);
    const D = new Array(K), Dpre = new Array(K), Fd = new Array(K), gW = new Array(K), gb = new Array(K);
    let delta = A[K].map((r, i) => r.map((p, j) => (p - Y[i][j]) / n));
    for (let k = K - 1; k >= 0; k--) {
      D[k] = delta;
      gW[k] = M.multiply(M.transpose(A[k]), delta);
      gb[k] = colSum(delta);
      if (k > 0) {
        const back = M.multiply(delta, M.transpose(net.W[k]));
        const deriv = Z[k - 1].map((row, i) => row.map((z, j) => net.act.d(z, A[k][i][j])));
        Dpre[k - 1] = back; Fd[k - 1] = deriv;
        delta = M.dotMultiply(back, deriv);
      }
    }
    return { A, Z, D, Dpre, Fd, gW, gb, loss: net.loss(A[K], Y), n };
  }

  /** Worked example of one node: z = Σ a·w + b, then the activation. */
  function workedZ(aPrev, W, b, j, s) {
    const last = s === L - 1, act = actText();
    const terms = aPrev.map((av, i) => ({ i, av, w: W[i][j] })).filter((t) => t.av !== 0);
    const z = terms.reduce((acc, t) => acc + t.av * t.w, 0) + b[j];
    const node = lbl(s, j);
    let txt = `z${s}[${node}] = Σᵢ a${s - 1}[i] · W${s}[i][${node}] + b${s}[${node}]`;
    if (terms.length < aPrev.length) txt += `     (the ${aPrev.length - terms.length} terms with a = 0 are left out)`;
    txt += "\n";
    terms.forEach((t) => { txt += `   ${pad(lbl(s - 1, t.i), 8)} ${f(t.av)} · ${fp(t.w)} = ${f(t.av * t.w)}\n`; });
    txt += `   ${pad("bias", 8)} ${f(b[j])}\n   z = ${f(z)}\n`;
    txt += last ? `p = σ(z) = 1 / (1 + e^(${f(-z)})) = ${f(sig(z))}` : `a = f(z) = ${act.how(z)} = ${f(ctx.net.act.f(z))}`;
    return txt;
  }

  // =========================================================================
  //  One optimisation step, fully explained (Pretraining Math / Training Math)
  // =========================================================================
  function explainStep(o) {
    const net = o.net, n = o.batch.length, R = o.shown, nb = n > R;
    const X = o.batch.map((r) => r.x), Y = o.batch.map((r) => r.y);
    const recs = o.batch.slice(0, R);
    const rl = recs.map((r) => r.inWords.join(" "));
    const pi = passInfo(net, X, Y);
    const top = (A) => A.slice(0, R);
    const act = actText();
    const inL = LABELS[0], outL = LABELS[L - 1];
    const tgt = recs.map((r) => r.y.map((v, j) => (v ? j : -1)).filter((j) => j >= 0));
    const allTgt = new Set(tgt.flat());
    const activeIn = new Set(recs.flatMap((r) => r.x.map((v, i) => (v ? i : -1)).filter((i) => i >= 0)));
    const out = [];
    let no = 0, h;
    const sec = (title, body) => out.push(`<section class="ms"><h3>${++no}. ${title}</h3>${body}</section>`);

    // ---- starting point -------------------------------------------------------
    if (o.wasZero) {
      h = `<p>All weights are 0 right now (after <b>Clean up</b>). If all weights are equal, every neuron of a layer computes ` +
        `the same value and receives the same gradient, so they stay identical forever and the network cannot learn. ` +
        `So the button first sets the weights randomly (Xavier / Glorot uniform, seed ${HP.seed}) and the biases to 0:</p>`;
      h += formula(`W<sub>s</sub>[i][j] ~ U(−limit, +limit),   limit = √(6 / (n<sub>in</sub> + n<sub>out</sub>)),   b<sub>s</sub> = 0\n` +
        sizes.slice(0, -1).map((a, k) => `W${k + 1}: ${a}×${sizes[k + 1]}   limit = √(6 / ${a + sizes[k + 1]}) = ${f(Math.sqrt(6 / (a + sizes[k + 1])))}`).join("\n"));
    } else {
      const st = ctx.getState();
      h = `<p>The network already has weights (pretrained: ${yes(st.pretrained)}, trained: ${yes(st.trained)}, ` +
        `Adam step counter t = ${net.t}). ${o.phase} continues from these weights – nothing is reset, ` +
        `and Adam keeps its memory m, v of earlier steps.</p>`;
    }
    const params = sizes.slice(1).reduce((s, sz, k) => s + sizes[k] * sz + sz, 0);
    h += formula(model.layers.map((ly) => `${ly.name} (${ly.size})`).join("  →  ") +
      `\nparameters: ` + sizes.slice(1).map((sz, k) => `W${k + 1} ${sizes[k]}×${sz} + b${k + 1} ${sz}`).join(" + ") + ` = ${params}`);
    sec("Starting point: the weights", h);

    // ---- vectors ----------------------------------------------------------------
    h = `<p>Every word has a fixed position. The input vector <b>x</b> has a 1 at the position of every word in the phrase ` +
      `(one-hot / multi-hot), the target vector <b>y</b> has a 1 at every expected output word.</p>`;
    if (nb) h += `<p>All ${n} records are processed together as one batch: X is ${n}×${sizes[0]}, Y is ${n}×${sizes[L - 1]}. ` +
      `Only the first ${R} rows are shown. Every number of these rows is exact; numbers that depend on the whole batch ` +
      `(loss, gradients, update) are computed over all ${n} records.</p>`;
    h += matrix(top(X), rl.map((s) => `x  ${s}`), inL, { hlCols: activeIn, d: 0 });
    h += matrix(top(Y), recs.map((r) => `y  ${r.outWords.join(" ")}`), outL, { hlCols: allTgt, d: 0 });
    sec("The records as vectors", h);

    // ---- forward pass -----------------------------------------------------------
    h = `<p>The input passes through ${L - 1} weight matrices. Each layer multiplies the previous activations by its weights, ` +
      `adds the bias and applies the activation function.</p>`;
    h += formula(`Z<sub>s</sub> = A<sub>s−1</sub> · W<sub>s</sub> + b<sub>s</sub>        element: z[r][j] = Σᵢ a[r][i] · W[i][j] + b[j]\n` +
      `A<sub>s</sub> = f(Z<sub>s</sub>)   hidden layers: ${act.f}\n` +
      `A<sub>${L - 1}</sub> = σ(Z<sub>${L - 1}</sub>) = 1 / (1 + e<sup>−z</sup>)   output layer: every word gets its own probability 0…1\n` +
      `A<sub>0</sub> = X`);
    for (let s = 1; s < L; s++) {
      const W = net.W[s - 1], b = net.b[s - 1], Zs = top(pi.Z[s - 1]), As = top(pi.A[s]), Ap = top(pi.A[s - 1]);
      const last = s === L - 1;
      let t = `<h4>${layName(s - 1)} → ${layName(s)}</h4>`;
      t += formula(`${S("Z", s)} = ${S("A", s - 1)} · ${S("W", s)} + ${S("b", s)}     (${n}×${sizes[s - 1]}) · (${sizes[s - 1]}×${sizes[s]}) → ${n}×${sizes[s]}\n` +
        `${S("A", s)} = ${last ? "σ" : "f"}(${S("Z", s)})`);
      t += fold(`${S("W", s)}: ${sizes[s - 1]}×${sizes[s]} weights (rows = ${layName(s - 1)}, columns = ${layName(s)})`,
        matrix(W, LABELS[s - 1], LABELS[s], { hlRows: s === 1 ? activeIn : undefined }));
      t += vector(b, LABELS[s], `b${s}`);
      if (s === 1) t += `<p>Because x contains only 0 and 1, the multiplication just <b>adds rows</b> of W1: ` +
        recs.map((r) => `z1(${r.inWords.join(" ")}) = ${r.inWords.map((w) => `W1[${w}]`).join(" + ")} + b1`).join("; ") +
        `. The other rows are multiplied by 0.</p>`;
      t += matrix(Zs, rl.map((x) => `Z${s}  ${x}`), LABELS[s]);
      t += matrix(As, rl.map((x) => `A${s}  ${x}`), LABELS[s], { hlCols: last ? allTgt : undefined });
      const j = last ? tgt[0][0] : argmax(As[0]);
      t += `<p>Worked example – record “${rl[0]}”, node ${lbl(s, j)}:</p>` + formula(workedZ(Ap[0], W, b, j, s));
      if (!last && HP.activation === "lrelu") {
        t += `<p>Negative sums: ${Zs.map((r, i) => `${r.filter((z) => z <= 0).length} of ${sizes[s]} nodes for “${rl[i]}”`).join(", ")}. ` +
          `Leaky ReLU lets only 1% of a negative value through, so these nodes are almost silent.</p>`;
      }
      h += t;
    }
    sec("Forward pass – layer by layer", h);

    // ---- loss -------------------------------------------------------------------
    const P = top(pi.A[L - 1]);
    const PY = [], PYl = [];
    recs.forEach((r, i) => { PY.push(P[i], Y[i]); PYl.push(`P  ${rl[i]}`, `Y  ${rl[i]}`); });
    h = `<p>The loss measures how far the predicted probabilities P are from the targets Y. For every output word:</p>`;
    h += formula(`ℓ(p, y) = −[ y · ln(p) + (1 − y) · ln(1 − p) ]        (binary cross-entropy)\n` +
      `y = 1 → ℓ = −ln(p)      small when p is close to 1\n` +
      `y = 0 → ℓ = −ln(1 − p)  small when p is close to 0\n` +
      `Loss = 1 / (n · ${sizes[L - 1]}) · Σ<sub>records</sub> Σ<sub>words</sub> ℓ`);
    h += matrix(PY, PYl, outL, { hlCols: allTgt });
    {
      const j = tgt[0][0], p = P[0][j];
      const non = P[0].map((v, k) => ({ v, k })).filter((x) => !Y[0][x.k]).sort((a, c) => c.v - a.v)[0];
      h += formula(`Record “${rl[0]}”:\n` +
        `  target “${lbl(L - 1, j)}”:  y = 1, p = ${f(p)}  →  ℓ = −ln(${f(p)}) = ${f(-Math.log(p))}\n` +
        `  “${lbl(L - 1, non.k)}” (strongest wrong word):  y = 0, p = ${f(non.v)}  →  ℓ = −ln(1 − ${f(non.v)}) = ${f(-Math.log(1 - non.v))}\n` +
        recs.map((r, i) => `  loss of “${rl[i]}” = mean of its ${sizes[L - 1]} ℓ values = ${f(net.loss([P[i]], [Y[i]]))}`).join("\n") +
        `\n  loss of the whole batch (${n} record${n > 1 ? "s" : ""}) = ${f(pi.loss)}`);
    }
    sec("Loss – how wrong is the network?", h);

    // ---- backward pass ----------------------------------------------------------
    h = `<p>Back-propagation uses the chain rule to find, for every weight, how the loss changes when this weight grows ` +
      `a little – the <b>gradient</b>. It starts at the output and goes back layer by layer. ` +
      `δ<sub>s</sub> is the error signal of layer s (∂Loss/∂Z<sub>s</sub>).</p>`;
    h += formula(`δ${L - 1} = (P − Y) / n                       sigmoid + cross-entropy simplify to this;  n = ${n}\n` +
      `dW<sub>s</sub> = A<sub>s−1</sub>ᵀ · δ<sub>s</sub>                    db<sub>s</sub> = Σ<sub>rows</sub> δ<sub>s</sub>\n` +
      `δ<sub>s−1</sub> = (δ<sub>s</sub> · W<sub>s</sub>ᵀ) ⊙ f′(Z<sub>s−1</sub>)       ⊙ = element by element,  ${act.d}`);
    h += `<p class="dim">nn.js divides by n only (not by n·${sizes[L - 1]} as in the printed loss): the gradient of the loss summed ` +
      `over the ${sizes[L - 1]} words. A constant factor does not matter for Adam, which normalises the step size.</p>`;
    for (let s = L - 1; s >= 1; s--) {
      const D = top(pi.D[s - 1]), Ap = top(pi.A[s - 1]), G = pi.gW[s - 1];
      let t = `<h4>${S("W", s)} and ${S("b", s)} (${layName(s - 1)} → ${layName(s)})</h4>`;
      t += matrix(D, rl.map((x) => `δ${s}  ${x}`), LABELS[s]);
      if (s === L - 1) {
        const j = tgt[0][0];
        t += formula(`δ${s}[“${rl[0]}”][${lbl(s, j)}] = (p − y) / n = (${f(P[0][j])} − 1) / ${n} = ${f(D[0][j])}\n` +
          `negative δ → increasing z of “${lbl(s, j)}” would lower the loss, so the weights into it will grow`);
      }
      const C = W0(sizes[s - 1], sizes[s]);
      for (let r = 0; r < R; r++) for (let i = 0; i < sizes[s - 1]; i++) for (let j = 0; j < sizes[s]; j++) C[i][j] += Ap[r][i] * D[r][j];
      const [gi, gj] = argmaxAbs2(C);
      const terms = Ap.map((a, r) => `${fp(a[gi])}·${fp(D[r][gj])}`).join(" + ");
      const shownSum = C[gi][gj];
      t += formula(`dW${s} = A${s - 1}ᵀ · δ${s}     (${sizes[s - 1]}×${n}) · (${n}×${sizes[s]}) → ${sizes[s - 1]}×${sizes[s]}\n` +
        `example dW${s}[${lbl(s - 1, gi)}][${lbl(s, gj)}] = Σ<sub>r</sub> a${s - 1}[r][${lbl(s - 1, gi)}] · δ${s}[r][${lbl(s, gj)}]\n` +
        `   = ${terms}${nb ? `   (the ${R} records shown) = ${f(shownSum)}\n   + ${f(G[gi][gj] - shownSum)}   (the other ${n - R} records)` : ""}\n` +
        `   = ${f(G[gi][gj])}`);
      if (nb) t += fold(`Contribution of the ${R} records shown to dW${s}`, matrix(C, LABELS[s - 1], LABELS[s], { hlRows: s === 1 ? activeIn : undefined }));
      t += fold(`dW${s}: full gradient${nb ? ` over all ${n} records` : ""} (${sizes[s - 1]}×${sizes[s]})`,
        matrix(G, LABELS[s - 1], LABELS[s], { hlRows: s === 1 ? activeIn : undefined }));
      t += vector(pi.gb[s - 1], LABELS[s], `db${s}`);
      if (s === 1) {
        t += `<p>Rows of words that are not in a record get no gradient from it (their x = 0). A record only changes the ` +
          `input weights of its own words. δ1 is not propagated further – the input is data, not a parameter.</p>`;
      } else {
        const back = top(pi.Dpre[s - 2]), der = top(pi.Fd[s - 2]), Dn = top(pi.D[s - 2]), W = net.W[s - 1];
        t += `<p>Error signal for the layer below:</p>`;
        t += formula(`δ${s - 1} = (δ${s} · W${s}ᵀ) ⊙ f′(Z${s - 1})     (${n}×${sizes[s]}) · (${sizes[s]}×${sizes[s - 1]}) → ${n}×${sizes[s - 1]}`);
        t += matrix(back, rl.map((x) => `δ${s}·W${s}ᵀ  ${x}`), LABELS[s - 1]);
        t += matrix(der, rl.map((x) => `f′(Z${s - 1})  ${x}`), LABELS[s - 1], { d: 2 });
        t += matrix(Dn, rl.map((x) => `δ${s - 1}  ${x}`), LABELS[s - 1]);
        const i = argmaxAbs(Dn[0]);
        let wt = `δ${s - 1}[“${rl[0]}”][${lbl(s - 1, i)}] = ( Σ<sub>j</sub> δ${s}[j] · W${s}[${lbl(s - 1, i)}][j] ) · f′(z)\n`;
        D[0].forEach((dv, j) => { if (dv !== 0) wt += `   ${pad(lbl(s, j), 8)} ${fp(dv)} · ${fp(W[i][j])} = ${f(dv * W[i][j])}\n`; });
        wt += `   sum = ${f(back[0][i])},  z = ${f(pi.Z[s - 2][0][i])} → f′(z) = ${f(der[0][i], 2)}\n   δ = ${f(back[0][i])} · ${f(der[0][i], 2)} = ${f(Dn[0][i])}`;
        t += `<p>Worked example (the node with the largest error signal):</p>` + formula(wt);
      }
      h += t;
    }
    sec("Backward pass – gradients", h);

    // ---- Adam update ----------------------------------------------------------------
    const tt = net.t + 1, c1 = 1 - Math.pow(B1, tt), c2 = 1 - Math.pow(B2, tt);
    const after = copyNet(net);
    const st = after.trainEpoch(X, Y, o.lr, o.wd);
    h = `<p>The gradient says in which direction each weight should move. Adam (adaptive moment estimation) keeps for every ` +
      `parameter a running average of the gradient (m) and of its square (v), and moves each weight by about the learning ` +
      `rate, in the direction that lowers the loss. Weight decay (AdamW) also shrinks every weight a little, which favours ` +
      `simple, general rules over memorising.</p>`;
    h += formula(`m ← β1·m + (1 − β1)·g             β1 = ${B1}\n` +
      `v ← β2·v + (1 − β2)·g²            β2 = ${B2}\n` +
      `m̂ = m / (1 − β1<sup>t</sup>),   v̂ = v / (1 − β2<sup>t</sup>)          bias correction\n` +
      `w ← w − lr · m̂ / (√v̂ + ε) − lr · λ · w     lr = ${o.lr},  λ = ${o.wd},  ε = ${EPS}\n` +
      `b ← b − lr · m̂ / (√v̂ + ε)                (no decay for biases)\n` +
      `this step: t = ${tt},  1 − β1<sup>t</sup> = ${f(c1)},  1 − β2<sup>t</sup> = ${f(c2)}`);
    if (tt === 1) h += `<p>At the very first step m̂ = g and v̂ = g², so the step is lr·g/|g| = ±${o.lr}: every weight moves ` +
      `by almost exactly the learning rate, whatever the size of its gradient.</p>`;
    const picks = [];
    for (let s = 1; s < L; s++) {
      const [i, j] = argmaxAbs2(pi.gW[s - 1]);
      picks.push({ name: `W${s}[${lbl(s - 1, i)}][${lbl(s, j)}]`, w: net.W[s - 1][i][j], g: pi.gW[s - 1][i][j],
        m0: net.mW[s - 1][i][j], v0: net.vW[s - 1][i][j], nw: after.W[s - 1][i][j], decay: true });
    }
    { const j = tgt[0][0];
      picks.push({ name: `b${L - 1}[${lbl(L - 1, j)}]`, w: net.b[L - 2][j], g: pi.gb[L - 2][j],
        m0: net.mb[L - 2][j], v0: net.vb[L - 2][j], nw: after.b[L - 2][j], decay: false }); }
    h += `<p>The parameter with the largest gradient of every matrix, and the output bias of “${lbl(L - 1, tgt[0][0])}”:</p>`;
    h += table(["parameter", "w", "g", "m old", "v old", "m new", "v new", "m̂", "v̂", "step", "decay lr·λ·w", "w new"],
      picks.map((p) => {
        const m = B1 * p.m0 + (1 - B1) * p.g, v = B2 * p.v0 + (1 - B2) * p.g * p.g;
        const mh = m / c1, vh = v / c2, step = o.lr * mh / (Math.sqrt(vh) + EPS), dec = p.decay ? o.lr * o.wd * p.w : 0;
        return [p.name, f(p.w), f(p.g), f(p.m0), f(p.v0), f(m), f(v), f(mh), f(vh), f(step), f(dec), `<b>${f(p.nw)}</b>`];
      }));
    h += `<p>All ${params} parameters are updated this way at the same time. The sum of all |step| values is ` +
      `<b>${f(st.change)}</b> – this is the “Δw” number printed in the log.</p>`;
    sec("Update – Adam with weight decay", h);

    // ---- effect -----------------------------------------------------------------
    const pa = after.forward(X).A[L - 1];
    const lossAfter = after.loss(pa, Y);
    const rows = [];
    recs.forEach((r, i) => {
      const words = [...tgt[i]];
      const wrong = P[i].map((v, k) => ({ v, k })).filter((x) => !r.y[x.k]).sort((a, c) => c.v - a.v)[0];
      words.push(wrong.k);
      words.forEach((j) => rows.push({ cls: r.y[j] ? "" : "dimr",
        cells: [esc(rl[i]), esc(lbl(L - 1, j)) + (r.y[j] ? " (target)" : " (wrong)"), f(P[i][j]), f(pa[i][j]), sg(pa[i][j] - P[i][j])] }));
    });
    h = `<p>The same records after this one update:</p>`;
    h += table(["record", "output word", "p before", "p after", "change"], rows);
    h += formula(recs.map((r, i) => `loss “${rl[i]}”:  ${f(net.loss([P[i]], [Y[i]]))} → ${f(after.loss([pa[i]], [Y[i]]))}`).join("\n") +
      `\nbatch loss (${n} record${n > 1 ? "s" : ""}):  ${f(pi.loss)} → ${f(lossAfter)}`);
    sec("Result of one step", h);

    if (o.repeat) out.push(o.repeat(net, X, Y, ++no));
    out.push(`<section class="ms"><h3>${++no}. The whole ${o.phase.toLowerCase()} loop</h3>${o.loop}</section>`);
    return out.join("");
  }

  const W0 = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));

  // =========================================================================
  //  Pretraining Math
  // =========================================================================
  function explainPretrain() {
    const { net, wasZero } = startNet();
    const ep = HP.pretrain.epochs, lr = HP.pretrain.lr;
    const first = model.pretrain.slice(0, 2).map((r) => `“${esc(r.text)}”`).join(" and ");
    let h = `<section class="ms"><p class="lead"><b>Pretraining</b> teaches the network general word associations from a ` +
      `large set of ${model.pretrain.length} records: 15 single words, 75 word pairs and 50 full phrases ` +
      `(like a language model reading a lot of text). It runs ${ep} epochs; each epoch processes all records at once ` +
      `(full batch) with learning rate ${lr} and weight decay ${HP.weightDecay}.</p>` +
      `<p>Below, the very first epoch is computed exactly as it would run if you pressed <b>Pretraining</b> now – on a copy ` +
      `of the network, the real one is not changed. The first 2 records, ${first}, are followed number by number.</p></section>`;
    h += explainStep({
      net, wasZero, batch: model.pretrain, shown: 2, lr, wd: HP.weightDecay, phase: "Pretraining",
      loop: `<p>The same four steps – forward pass, loss, backward pass, Adam update – are repeated ${ep} times on all ` +
        `${model.pretrain.length} records. The log prints every few epochs: <b>loss</b> (should fall), <b>Δw</b> (sum of all ` +
        `update steps, gets smaller as the weights settle) and <b>ok</b> = number of records whose predicted word set ` +
        `(every word with p &gt; ${HP.threshold}) equals the expected set.</p>` +
        `<p>Because single words and pairs are in the set, the network learns what each word means on its own ` +
        `(cat → meows, rain → wet …), which is why it can later answer combinations it has never seen.</p>`,
    });
    return h;
  }

  // =========================================================================
  //  Training Math
  // =========================================================================
  function explainTrain() {
    const ex = model.mathExample;
    const { net, wasZero } = startNet();
    const lr = HP.train.lr, ep = HP.train.epochs;
    const st = ctx.getState();
    const before = net.predict(ex.x)[L - 1];
    const pred = model.outputWords.filter((w, j) => before[j] > HP.threshold).map((w) => w.word);
    const ok = pred.slice().sort().join() === ex.outWords.slice().sort().join();
    let h = `<section class="ms"><p class="lead"><b>Training</b> is fine-tuning on the real task: ${model.train.length} full ` +
      `three-word phrases, ${ep} epochs, learning rate ${lr} (smaller than in pretraining, so that what was learned is ` +
      `not destroyed). With <i>replay</i> the ${model.pretrain.length} pretraining records are mixed in, so one batch has ` +
      `n = ${model.train.length + (HP.train.replay ? model.pretrain.length : 0)} records.</p>` +
      `<p>To show the maths on a fresh example, this page uses a new record that is in none of the sets ` +
      `(<code>mathExample</code> in model.js):</p>` +
      formula(`“${esc(ex.text)}”     input: ${ex.inWords.join(", ")}   →   expected output: ${ex.outWords.join(", ")}`) +
      `<p>One training step is computed with a batch of just this record (n = 1), starting from the current network ` +
      `(on a copy – the real network is not changed).</p>` +
      `<p>Before the step the network ${net.isZero() ? "is empty" : `predicts <b>${esc(pred.join(" ") || "(nothing)")}</b> ` +
        `${ok ? '<span class="yes">✔ already correct – it generalises from what it learned</span>' : '<span class="no">✘ not yet correct</span>'}`}.` +
      `${!st.pretrained ? " (The network is not pretrained yet.)" : ""}</p></section>`;
    h += explainStep({
      net, wasZero, batch: [ex], shown: 1, lr, wd: HP.weightDecay, phase: "Training",
      repeat: (n0, X, Y, num) => {
        const c = copyNet(n0), marks = new Set([1, 2, 3, 5, 10, 20, 50, 100]), rows = [];
        const tj = ex.y.map((v, j) => (v ? j : -1)).filter((j) => j >= 0);
        for (let i = 1; i <= 100; i++) {
          c.trainEpoch(X, Y, lr, HP.weightDecay);
          if (!marks.has(i)) continue;
          const p = c.predict(ex.x)[L - 1];
          const words = model.outputWords.filter((w, j) => p[j] > HP.threshold).map((w) => w.word);
          rows.push([i, f(c.loss([p], Y)), ...tj.map((j) => f(p[j])), esc(words.join(" ") || "(nothing)")]);
        }
        return `<section class="ms"><h3>${num}. Repeating the step</h3><p>The same step repeated on this one record ` +
          `(still on a copy):</p>` + table(["steps", "loss", ...tj.map((j) => `p(${lbl(L - 1, j)})`), "predicted words"], rows) +
          `<p class="dim">Training on one record alone would make the network over-fit it; that is why the real Training ` +
          `uses all ${model.train.length} phrases and replays the pretraining set.</p></section>`;
      },
      loop: `<p>The real Training runs ${ep} epochs. Every epoch: forward pass of all ${model.train.length}` +
        `${HP.train.replay ? ` + ${model.pretrain.length} replayed` : ""} records, loss, backward pass with δ = (P − Y)/n, ` +
        `Adam update with lr = ${lr}. Adam’s m, v and the step counter t continue from pretraining.</p>`,
    });
    return h;
  }

  // =========================================================================
  //  Testing Math – the forward pass of the phrase in the input field
  // =========================================================================
  function explainTest() {
    const net = ctx.net, thr = HP.threshold, act = actText();
    let text = ctx.getText().trim(), note = "";
    let enc = model.encodeText(text);
    if (!enc.known.length) {
      note = `<p class="warn">${text ? `“${esc(text)}” contains no known word` : "The input field is empty"}, so the example ` +
        `phrase “dog tired night” is used. Type your own phrase and open this window again.</p>`;
      text = "dog tired night"; enc = model.encodeText(text);
    }
    const fw = net.forward([enc.x]);
    const a = fw.A.map((r) => r[0]), z = fw.Z.map((r) => r[0]);      // a[0..L-1]; z[s-1] = Z_s
    const P = a[L - 1];
    const chosen = P.map((p, j) => j).filter((j) => P[j] > thr);
    const outIdx = chosen.length ? chosen : [argmax(P)];
    const steps = ctx.tracePaths(a, outIdx);
    const onPath = (k) => (k === L - 1 ? new Set(outIdx) : new Set(steps[k].map((e) => e.i)));
    const lit = (k, i) => {
      if (k === 0) return a[0][i] > 0;
      if (k === L - 1) return P[i] > thr;
      const mx = Math.max(1e-9, ...a[k].map(Math.abs));
      return a[k][i] > 0 && Math.abs(a[k][i]) / mx > 0.55;
    };
    const out = [];
    let no = 0, h;
    const sec = (title, body) => out.push(`<section class="ms"><h3>${++no}. ${title}</h3>${body}</section>`);

    out.push(`<section class="ms"><p class="lead"><b>Testing</b> only runs the network forward: the phrase is turned into a ` +
      `vector, multiplied through the ${L - 1} weight matrices, and every output word with probability &gt; ${thr} is ` +
      `chosen. Nothing is learned. Phrase: <b>“${esc(text)}”</b></p>${note}` +
      (net.isZero() ? `<p class="warn">The network is empty (all weights 0): every sum is 0 and every output is σ(0) = 0.5, ` +
        `which is not &gt; 0.5, so nothing is chosen. Press Pretraining first.</p>` : "") + `</section>`);

    // ---- tokens -----------------------------------------------------------------
    const seen = new Set();
    const tokRows = text.toLowerCase().split(/[^a-z]+/).filter(Boolean).map((t) => {
      const w = model.encodeText(t).known[0];
      if (!w) return { cls: "dimr", cells: [esc(t), '<span class="no">unknown – ignored</span>'] };
      const iw = model.inputWords.find((x) => x.word === w);
      const how = w === t ? "" : ` (plural “s” removed → “${w}”)`;
      if (seen.has(w)) return { cls: "dimr", cells: [esc(t), `“${w}” already counted – x stays 1`] };
      seen.add(w);
      return [esc(t), `<span class="yes">known</span>: input word #${iw.n} “${w}”, group ${iw.group}${how} → x[${iw.n - 1}] = 1`];
    });
    h = `<p>The text is lower-cased and split at every non-letter. Every token is looked up in the vocabulary of the ` +
      `${model.inputWords.length} input words.</p>` + table(["token", "result"], tokRows);
    const activeIn = new Set(enc.x.map((v, i) => (v ? i : -1)).filter((i) => i >= 0));
    h += vector(enc.x, LABELS[0], "x = a0", { hlCols: activeIn, d: 0 });
    sec("Input → vector x", h);

    // ---- node table + reasons ----------------------------------------------------
    function reason(k, i) {
      const ai = a[k][i], W = net.W[k], limit = k === 0 ? 3 : 2;
      const next = [...onPath(k + 1)];
      if (onPath(k).has(i)) {
        const feeds = [...new Set(steps[k].filter((e) => e.i === i).map((e) => e.j))];
        return `<span class="yes">used</span>: feeds ${feeds.map((j) => `${lbl(k + 1, j)} (a·w = ${f(ai * W[i][j])})`).join(", ")}`;
      }
      if (ai === 0) return k === 0 ? "word not in the phrase (x = 0)" : "a = 0 – contributes nothing";
      if (!next.length) return "no arrow path reaches the next layer";
      let best = null;
      next.forEach((j) => {
        const c = ai * W[i][j];
        if (c <= 0) return;
        const list = a[k].map((av, i2) => ({ i2, c: av * W[i2][j] })).filter((x) => x.c > 0).sort((p, q) => q.c - p.c);
        const cand = { j, c, rank: list.findIndex((x) => x.i2 === i) + 1, top: list[0].c };
        cand.ratio = c / cand.top;
        if (!best || cand.ratio > best.ratio) best = cand;
      });
      if (!best) return `a·w ≤ 0 for every path node of ${layName(k + 1)} (${next.map((j) => lbl(k + 1, j)).join(", ")}): ` +
        `it pushes them down or does not help them, so it is not followed`;
      return `best help: ${lbl(k + 1, best.j)} gets a·w = ${f(best.c)} from it – rank #${best.rank}, ${pct(best.ratio)} of the ` +
        `strongest (${f(best.top)}); ` + (best.rank > limit ? `only the ${limit} strongest are followed` : `needs more than 25%`);
    }
    function nodeTable(k) {
      const mx = Math.max(1e-9, ...a[k].map(Math.abs)), on = onPath(k);
      const idx = k === 0 ? [...activeIn] : a[k].map((_, i) => i).sort((p, q) => a[k][q] - a[k][p]);
      const rows = idx.map((i) => ({
        cls: on.has(i) ? "pathr" : lit(k, i) ? "" : "dimr",
        cells: [lbl(k, i), ...(k ? [f(z[k - 1][i])] : []), f(a[k][i]), pct(Math.abs(a[k][i]) / mx), yes(lit(k, i)), yes(on.has(i)), reason(k, i)],
      }));
      let t = table(["node", ...(k ? ["z"] : []), k ? "a = f(z)" : "x", "% of strongest", "glows", "on arrow path", "why"], rows);
      const strongUnused = idx.filter((i) => lit(k, i) && !on.has(i));
      if (strongUnused.length && k > 0) {
        t += `<p><b>Strong but not used:</b> ${strongUnused.map((i) => lbl(k, i)).join(", ")} glow (high activation) but carry ` +
          `no arrow. A high activation alone is not enough: what matters is a·w, the activation times the weight ` +
          `<i>to a node on the path</i>. These nodes have small or negative weights to the path nodes – they work for ` +
          `other output words (often pushing them down).</p>`;
      }
      return t;
    }

    sec(`${layName(0)} – which words take part`, `<p>On screen an input node glows when x = 1. It gets an arrow if it is one ` +
      `of the main contributors to an arrow node of L2 (rules in the “Arrow paths” section).</p>` + nodeTable(0));

    // ---- layers -----------------------------------------------------------------
    for (let s = 1; s < L; s++) {
      const W = net.W[s - 1], b = net.b[s - 1], last = s === L - 1;
      h = formula(`z${s} = a${s - 1} · W${s} + b${s}      (1×${sizes[s - 1]}) · (${sizes[s - 1]}×${sizes[s]}) + (1×${sizes[s]}) → 1×${sizes[s]}\n` +
        (last ? `p = a${s} = σ(z${s}) = 1 / (1 + e<sup>−z</sup>)` : `a${s} = f(z${s}),   ${act.f}`));
      h += fold(`W${s}: ${sizes[s - 1]}×${sizes[s]} (rows = ${layName(s - 1)}, columns = ${layName(s)}; highlighted = on the arrow path)`,
        matrix(W, LABELS[s - 1], LABELS[s], { hlRows: onPath(s - 1), hlCols: onPath(s) }));
      h += vector(b, LABELS[s], `b${s}`);
      if (s === 1) {
        const ids = [...activeIn];
        h += `<p>x has a 1 only for the ${ids.length} known word${ids.length > 1 ? "s" : ""}, so z1 is the sum of ` +
          `${ids.length > 1 ? "their rows" : "its row"} of W1 plus b1:</p>`;
        h += matrix([...ids.map((i) => W[i]), b, z[0]], [...ids.map((i) => `W1[${lbl(0, i)}]`), "+ b1", "= z1"], LABELS[1]);
      }
      h += vector(z[s - 1], LABELS[s], `z${s}`, { hlCols: onPath(s) });
      h += vector(a[s], LABELS[s], last ? `p = a${s}` : `a${s}`, { hlCols: onPath(s) });
      const j = last ? outIdx[0] : argmax(a[s]);
      h += `<p>Worked example – ${last ? "the first chosen output word" : "the strongest node"} ${lbl(s, j)}:</p>` +
        formula(workedZ(a[s - 1], W, b, j, s));
      if (!last) {
        h += `<p>Glow on screen: a &gt; 0 and |a| &gt; 55% of the strongest |a| in this layer. Arrow path: see section ` +
          `“Arrow paths”. Nodes sorted by activation:</p>` + nodeTable(s);
      }
      sec(`${layName(s - 1)} → ${layName(s)}`, h);
    }

    // ---- output selection --------------------------------------------------------
    const zo = z[L - 2], W4 = net.W[L - 2], b4 = net.b[L - 2];
    const ranked = P.map((p, j) => j).sort((p, q) => P[q] - P[p]);
    h = `<p><b>Selection criterion:</b> an output word is chosen when its probability p &gt; ${thr}, which is the same as ` +
      `z &gt; 0. Every word has its own sigmoid – the words do not compete with each other, so any number of them can be ` +
      `chosen, and a high rank alone means nothing. A well-trained network chooses one word of each group (sound, action, condition).</p>`;
    h += table(["rank", "word", "group", "z", "p", "chosen", "why"], ranked.map((j, r) => ({
      cls: P[j] > thr ? "pathr" : r < 5 ? "" : "dimr",
      cells: [r + 1, lbl(L - 1, j), model.outputWords[j].group, f(zo[j]), f(P[j]), yes(P[j] > thr),
        P[j] > thr ? `p = ${f(P[j])} &gt; ${thr}` :
          r < 5 ? `high rank #${r + 1}, but p ≤ ${thr} (z ≤ 0): the sum of its inputs is not positive` : `p ≤ ${thr}`],
    })));
    const explainWords = [...chosen];
    const bestNo = ranked.find((j) => P[j] <= thr);
    if (bestNo !== undefined) explainWords.push(bestNo);
    let bt = "";
    explainWords.forEach((j) => {
      const cs = a[L - 2].map((av, i) => ({ i, c: av * W4[i][j] }));
      const pos = cs.filter((x) => x.c > 0).sort((p, q) => q.c - p.c), neg = cs.filter((x) => x.c < 0).sort((p, q) => p.c - q.c);
      const sp = pos.reduce((s2, x) => s2 + x.c, 0), sn = neg.reduce((s2, x) => s2 + x.c, 0);
      bt += `${lbl(L - 1, j)}${P[j] > thr ? "  (chosen)" : "  (highest not chosen)"}:\n` +
        `   z = ${f(sp)} (all positive a·w) + ${fp(sn)} (all negative a·w) + ${fp(b4[j])} (bias) = ${f(zo[j])}  →  p = ${f(P[j])}\n` +
        `   strongest push up:    ${pos.slice(0, 4).map((x) => `${lbl(L - 2, x.i)} ${sg(x.c)}`).join(",  ") || "–"}\n` +
        `   strongest pull down:  ${neg.slice(0, 4).map((x) => `${lbl(L - 2, x.i)} ${sg(x.c)}`).join(",  ") || "–"}\n\n`;
    });
    h += `<p>Why the words are in or out – their sum split into positive and negative parts:</p>` + formula(bt.trimEnd());
    sec("Output – which words are chosen", h);

    // ---- arrow paths ---------------------------------------------------------------
    const colors = ctx.pathColors;
    h = `<p>The arrows on screen show where the decision came from. For every chosen output word (or the most probable ` +
      `one if none is chosen) the trace goes backwards, layer by layer:</p>` +
      formula(`for every node j already on the path, and every node i of the layer below:\n` +
        `   contribution c = a[i] · W[i][j]          (how much node i raises z[j])\n` +
        `keep only c &gt; 0, sort them from the largest,\n` +
        `follow at most 3 (from the input words into L2) or 2 (between hidden layers / into the output),\n` +
        `and only those with c &gt; 25% of the strongest contribution to that node j.`) +
      `<p>That is why a node with a high activation can still be without an arrow: its weight to the path nodes is small ` +
      `or negative, or two other nodes contribute more.</p>`;
    outIdx.forEach((o, ci) => {
      const col = ci % colors.length;
      let t = `<h4><i class="sw" style="background:${colors[col]}"></i>${lbl(L - 1, o)}${chosen.length ? "" : " (most probable, not chosen)"}</h4>`;
      const lines = [];
      for (let k = L - 2; k >= 0; k--) {
        steps[k].filter((e) => e.color === col).forEach((e) => {
          lines.push(`${pad(layName(k + 1), 12)} ${pad(lbl(k + 1, e.j), 8)} ← ${pad(lbl(k, e.i), 8)} c = ${f(a[k][e.i])} · ${fp(net.W[k][e.i][e.j])} = ${f(a[k][e.i] * net.W[k][e.i][e.j])}`);
        });
      }
      t += formula(lines.join("\n") || "no positive contribution – no arrows");
      h += t;
    });
    sec("Arrow paths – how the nodes are selected", h);

    const words = chosen.map((j) => lbl(L - 1, j));
    sec("Result", formula(`“${esc(text)}”  →  ${words.join(" ") || "(nothing)"}` +
      (words.length ? `\n${esc(model.sentence(enc.known, words))}` : "")));
    return out.join("");
  }

  // =========================================================================
  //  Menu and popup
  // =========================================================================
  const TITLES = { pretrain: "Pretraining Math", train: "Training Math", test: "Testing Math" };
  let lastFocus = null;

  function openMath(kind) {
    const pop = $("mathPopup"), body = $("mathBody");
    $("mathTitle").textContent = TITLES[kind];
    let html;
    try {
      html = kind === "pretrain" ? explainPretrain() : kind === "train" ? explainTrain() : explainTest();
    } catch (err) {
      console.error(err);
      html = `<p class="warn">Could not build the explanation: ${esc(err.message)}</p>`;
    }
    body.innerHTML = `<div class="math-doc">${html}</div>`;
    body.scrollTop = 0;
    lastFocus = document.activeElement;
    pop.hidden = false;
    $("mathClose").focus();
  }

  function closeMath() {
    $("mathPopup").hidden = true;
    $("mathBody").innerHTML = "";
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }

  function wire() {
    const btn = $("btnMenu"), menu = $("dotsMenu"), pop = $("mathPopup");
    const closeMenu = () => { menu.hidden = true; btn.setAttribute("aria-expanded", "false"); };
    const openMenu = () => {
      menu.hidden = false;
      const r = btn.getBoundingClientRect(), mw = menu.offsetWidth;
      menu.style.top = r.bottom + 4 + "px";
      menu.style.left = Math.max(4, Math.min(innerWidth - mw - 4, r.right - mw)) + "px";
      btn.setAttribute("aria-expanded", "true");
      menu.querySelector("button").focus();
    };
    btn.addEventListener("click", (e) => { e.stopPropagation(); menu.hidden ? openMenu() : closeMenu(); });
    menu.addEventListener("click", (e) => {
      const item = e.target.closest("[data-math], [data-action]");
      if (!item) return;
      closeMenu();
      if (item.dataset.action === "download") ctx.download();
      else if (item.dataset.action === "upload") ctx.upload();
      else openMath(item.dataset.math);
    });
    document.addEventListener("click", (e) => { if (!menu.hidden && !menu.contains(e.target)) closeMenu(); });
    document.addEventListener("keydown", (e) => {
      if (e.key !== "Escape") return;
      if (!pop.hidden) closeMath(); else if (!menu.hidden) { closeMenu(); btn.focus(); }
    });
    window.addEventListener("resize", closeMenu);
    $("mathBack").addEventListener("click", closeMath);
    $("mathClose").addEventListener("click", closeMath);
  }

  function init(c) {
    ctx = c; model = c.model; HP = model.hyper;
    sizes = model.layers.map((l) => l.size); L = sizes.length;
    LABELS = sizes.map((n, k) => (k === 0 ? model.inputWords.map((w) => w.word)
      : k === L - 1 ? model.outputWords.map((w) => w.word)
        : Array.from({ length: n }, (_, i) => `H${k}.${i + 1}`)));
    wire();
  }

  root.MathView = { init };
})(window);
