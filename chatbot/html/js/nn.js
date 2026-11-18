/*
 * nn.js – a small fully-connected neural network built on math.js matrices.
 *
 *   A0 = X                                   (input, one-hot words)
 *   Zk = A(k-1) · Wk + bk                    (matrix multiplication)
 *   Ak = f(Zk)             for hidden layers (leaky ReLU or tanh)
 *   A4 = sigmoid(Z4)       for the output layer (every word independently 0..1)
 *   loss = binary cross-entropy(A4, Y)
 *   back-propagation + Adam optimiser with weight decay
 */
(function (root) {
  "use strict";
  const M = root.math || (typeof require !== "undefined" ? require("./lib/math.min.js") : null);

  // deterministic random numbers so the demo is repeatable
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const sigmoid = (z) => 1 / (1 + Math.exp(-z));
  const addRow = (A, b) => A.map((row) => row.map((v, j) => v + b[j]));     // broadcast bias
  const colSum = (A) => A[0].map((_, j) => A.reduce((s, r) => s + r[j], 0));
  const apply = (A, f) => M.map(A, f);

  // hidden-layer activation functions and their derivatives (from z and a)
  const ACT = {
    tanh:  { f: Math.tanh, d: (z, a) => 1 - a * a, label: "tanh" },
    lrelu: { f: (z) => (z > 0 ? z : 0.01 * z), d: (z) => (z > 0 ? 1 : 0.01), label: "leaky ReLU" },
  };

  class Network {
    constructor(sizes, seed, activation) {
      this.sizes = sizes.slice();
      this.seed = seed || 1;
      this.act = ACT[activation] || ACT.tanh;
      this.zero();
    }

    /** All weights and biases = 0 ("Clean up"). */
    zero() {
      this.W = []; this.b = [];
      for (let k = 1; k < this.sizes.length; k++) {
        this.W.push(M.zeros(this.sizes[k - 1], this.sizes[k]).valueOf());
        this.b.push(new Array(this.sizes[k]).fill(0));
      }
      this.resetOptimizer();
      this.initialized = false;
      this.epochsDone = 0;
    }

    /** Xavier/Glorot random initialisation – breaks the symmetry between neurons. */
    randomize(seed) {
      const rnd = mulberry32(seed || this.seed);
      this.W = this.W.map((W, k) => {
        const limit = Math.sqrt(6 / (this.sizes[k] + this.sizes[k + 1]));
        return apply(W, () => (rnd() * 2 - 1) * limit);
      });
      this.b = this.b.map((b) => b.map(() => 0));
      this.resetOptimizer();
      this.initialized = true;
    }

    resetOptimizer() {
      this.t = 0;
      this.mW = this.W.map((W) => apply(W, () => 0)); this.vW = this.W.map((W) => apply(W, () => 0));
      this.mb = this.b.map((b) => b.map(() => 0));    this.vb = this.b.map((b) => b.map(() => 0));
    }

    isZero() { return this.W.every((W) => W.every((r) => r.every((v) => v === 0))); }

    /** Forward pass for a batch X (n × 15). Returns activations of all layers. */
    forward(X) {
      const A = [X], Z = [];
      const L = this.W.length;
      for (let k = 0; k < L; k++) {
        const z = addRow(M.multiply(A[k], this.W[k]), this.b[k]);   // Z = A·W + b
        Z.push(z);
        A.push(apply(z, k === L - 1 ? sigmoid : this.act.f));        // A = f(Z)
      }
      return { A, Z };
    }

    predict(x) { return this.forward([x]).A.map((a) => a[0]); }

    loss(P, Y) {
      const eps = 1e-7;
      let s = 0;
      for (let i = 0; i < P.length; i++)
        for (let j = 0; j < P[i].length; j++) {
          const p = Math.min(1 - eps, Math.max(eps, P[i][j]));
          s -= Y[i][j] * Math.log(p) + (1 - Y[i][j]) * Math.log(1 - p);
        }
      return s / (P.length * P[0].length);
    }

    /** One epoch of full-batch gradient descent with Adam.
     *  wd = decoupled weight decay (AdamW): every step shrinks weights slightly,
     *  which favours simple, general rules over memorising combinations. */
    trainEpoch(X, Y, lr, wd) {
      wd = wd || 0;
      const n = X.length, L = this.W.length;
      const { A, Z } = this.forward(X);
      const loss = this.loss(A[L], Y);

      // output layer: dL/dZ = (P - Y) / n   (sigmoid + cross-entropy)
      let delta = M.divide(M.subtract(A[L], Y), n);
      const gW = new Array(L), gb = new Array(L);
      for (let k = L - 1; k >= 0; k--) {
        gW[k] = M.multiply(M.transpose(A[k]), delta);       // dW = Aᵀ · δ
        gb[k] = colSum(delta);                               // db = Σ δ
        if (k > 0) {                                         // δ(k-1) = (δ · Wᵀ) ⊙ f'(Z)
          const zk = Z[k - 1], ak = A[k];
          const deriv = zk.map((row, i) => row.map((z, j) => this.act.d(z, ak[i][j])));
          delta = M.dotMultiply(M.multiply(delta, M.transpose(this.W[k])), deriv);
        }
      }

      // Adam update
      const b1 = 0.9, b2 = 0.999, eps = 1e-8;
      this.t++;
      const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
      let change = 0;
      for (let k = 0; k < L; k++) {
        const W = this.W[k], mW = this.mW[k], vW = this.vW[k], g = gW[k];
        for (let i = 0; i < W.length; i++)
          for (let j = 0; j < W[i].length; j++) {
            mW[i][j] = b1 * mW[i][j] + (1 - b1) * g[i][j];
            vW[i][j] = b2 * vW[i][j] + (1 - b2) * g[i][j] * g[i][j];
            const step = lr * (mW[i][j] / c1) / (Math.sqrt(vW[i][j] / c2) + eps);
            W[i][j] -= step + lr * wd * W[i][j]; change += Math.abs(step);
          }
        const bb = this.b[k], mb = this.mb[k], vb = this.vb[k];
        for (let j = 0; j < bb.length; j++) {
          mb[j] = b1 * mb[j] + (1 - b1) * gb[k][j];
          vb[j] = b2 * vb[j] + (1 - b2) * gb[k][j] * gb[k][j];
          bb[j] -= lr * (mb[j] / c1) / (Math.sqrt(vb[j] / c2) + eps);
        }
      }
      this.epochsDone++;
      return { loss, P: A[L], change, gradNorm: gW.map((g) => M.norm(M.flatten(g))) };
    }

    /** Weight summary of a node: incoming weights column, its L2 norm and its bias. */
    nodeInfo(layer, j) {
      if (layer === 0) return null;
      const W = this.W[layer - 1];
      const incoming = W.map((row) => row[j]);
      const norm = Math.sqrt(incoming.reduce((s, v) => s + v * v, 0));
      const mean = incoming.reduce((s, v) => s + v, 0) / incoming.length;
      return { incoming, norm, mean, bias: this.b[layer - 1][j] };
    }
  }

  const api = { Network, mulberry32, ACT };
  if (typeof module !== "undefined") module.exports = api;
  root.NN = api;
})(typeof window !== "undefined" ? window : globalThis);
