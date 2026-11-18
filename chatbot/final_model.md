# final_model.json – the final Mini chatbot model

`final_model.json` is created in **minichat.html** with **⋮ → Download model**.
It contains everything the **Process** button uses to turn a phrase into output words, and nothing else:

* **included:** vocabulary, network shape, all weights and biases, and the rules and parameters of processing
  (tokenizer, activation functions, threshold, sentence building);
* **not included:** the pretraining, training and test records, and training-only settings
  (epochs, learning rates, weight decay, Adam memory `m`/`v`, random seed).

The file is plain JSON, indented with 2 spaces. To keep it readable, short arrays of numbers or words
and small flat objects are written on one line, so every row of a weight matrix is exactly one line.

Example size: about 25 KB, 240 lines.

---

## Top level

```json
{
  "format": "mini-chatbot-final-model",
  "version": 1,
  "created": "2026-11-18T21:53:00.125Z",
  "status": { ... },
  "architecture": { ... },
  "processing": { ... },
  "vocabulary": { ... },
  "connections": [ ... ]
}
```

| key            | meaning |
|----------------|---------|
| `format`       | always `"mini-chatbot-final-model"` – identifies the file type |
| `version`      | version of this file layout (currently `1`) |
| `created`      | when the file was downloaded (ISO 8601, UTC) |
| `status`       | in what state the network was when it was saved |
| `architecture` | layers, their sizes, activation functions and node names |
| `processing`   | how a typed phrase is turned into output words |
| `vocabulary`   | the 15 input words and the 15 output words |
| `connections`  | the learned numbers: 4 weight matrices and 4 bias vectors |

---

## `status`

| key              | type    | meaning |
|------------------|---------|---------|
| `pretrained`     | boolean | Pretraining has finished since the last Clean up |
| `trained`        | boolean | Training has finished since the last Clean up |
| `optimizerSteps` | number  | how many update steps (epochs) the weights have received, e.g. 800 after pretraining and 1000 after pretraining + training |
| `emptyNetwork`   | boolean | `true` when every weight is 0 (after Clean up). Such a model answers nothing: every probability is 0.5 |

---

## `architecture`

```json
"architecture": {
  "parameterCount": 1475,
  "layers": [
    { "layer": "L1", "name": "Input",    "size": 15, "activation": "none (input vector)", "nodes": ["cat", "dog", ...] },
    { "layer": "L2", "name": "Hidden 1", "size": 20, "activation": "leaky ReLU",          "nodes": ["H1.1", ..., "H1.20"] },
    { "layer": "L3", "name": "Hidden 2", "size": 20, "activation": "leaky ReLU",          "nodes": ["H2.1", ..., "H2.20"] },
    { "layer": "L4", "name": "Hidden 3", "size": 20, "activation": "leaky ReLU",          "nodes": ["H3.1", ..., "H3.20"] },
    { "layer": "L5", "name": "Output",   "size": 15, "activation": "sigmoid",             "nodes": ["meows", "barks", ...] }
  ]
}
```

* `parameterCount` – number of weights and biases: 15·20+20 + 20·20+20 + 20·20+20 + 20·15+15 = **1475**.
* `nodes` – the names of the nodes in order. The position in this list is the index used in all vectors
  and matrices. Input and output nodes are words; hidden nodes are named `H<hidden layer>.<number>`, as
  on the screen.

---

## `processing` – the parameters Process uses

| key                | meaning |
|--------------------|---------|
| `tokenizer`        | how the typed text is split into words (see below) |
| `inputEncoding`    | `x[i] = 1` if input word number i+1 is in the phrase, otherwise `0` |
| `forwardPass`      | the formulas of the calculation, in order |
| `hiddenActivation` | function of the hidden layers L2–L4: leaky ReLU, `f(z) = z` if `z > 0`, otherwise `negativeSlope · z` with `negativeSlope = 0.01` (if the model is switched to tanh: `f(z) = tanh(z)`) |
| `outputActivation` | function of the output layer: sigmoid, `p = 1 / (1 + e^(-z))` – a probability between 0 and 1 for each output word |
| `threshold`        | `0.5` – an output word is chosen when its probability `p > threshold` |
| `selectionRule`    | the same in words. Each word is decided on its own, so any number of words can be chosen |
| `sentence`         | how the chosen words become an English sentence (see below) |

### `tokenizer`

| key              | value | meaning |
|------------------|-------|---------|
| `lowercase`      | `true` | the text is lower-cased first |
| `splitOn`        | `"[^a-z]+"` | regular expression: every run of non-letters separates words |
| `pluralRule`     | text | a word not in the vocabulary that ends with `s` is tried without the `s` (`cats` → `cat`) |
| `unknownWords`   | `"ignored"` | words not in the vocabulary do not take part |
| `duplicateWords` | `"counted once"` | a word typed twice still gives `x = 1` |

### `forwardPass`

```
a0 = x
z_k = a_(k-1) · W_k + b_k     for k = 1..4      (vector × matrix + vector)
a_k = hiddenActivation(z_k)   for k = 1..3
p   = a4 = outputActivation(z4)
```

`W_k` and `b_k` are the `weights` and `biases` of `connections[k-1]`.

### `sentence`

```json
"sentence": {
  "subject": "'The <animal>' if an animal word was typed, otherwise 'It'",
  "order": ["sound", "action", "condition"],
  "conditionPhrases": { "wet": "gets wet", "warm": "feels warm", "cold": "feels cold", "dark": "is in the dark", "outside": "is outside" },
  "empty": "(no confident output)"
}
```

The chosen words are listed by group in the order given by `order`. Words of the group `condition` are
replaced by their phrase from `conditionPhrases`. The last two parts are joined with “and”, the others
with commas. Example: `dog tired night` → `barks sleeps dark` → **“The dog barks, sleeps and is in the dark.”**
If no word is chosen, the text `empty` is shown.

---

## `vocabulary`

```json
"vocabulary": {
  "input":  [ { "n": 1, "word": "cat", "group": "animal" }, ... ],
  "output": [ { "n": 1, "word": "meows", "group": "sound" }, ... ]
}
```

| key     | meaning |
|---------|---------|
| `n`     | number of the word (1-based). Its position in the vectors is `n − 1` |
| `word`  | the word |
| `group` | input: `animal`, `state`, `situation`; output: `sound`, `action`, `condition` |

---

## `connections` – weights and biases

An array of 4 objects, one for each pair of neighbouring layers:

```json
{
  "name": "W1, b1",
  "from": "L1 Input",
  "to": "L2 Hidden 1",
  "shape": [15, 20],
  "columns": ["H1.1", "H1.2", ..., "H1.20"],
  "weights": {
    "cat": [-0.50275727, 0.6605541, ...20 numbers...],
    "dog": [0.74327331, -0.04465894, ...],
    ...
  },
  "biases": [-0.20729933, -0.25064402, ...20 numbers...]
}
```

| key       | meaning |
|-----------|---------|
| `name`    | names of the matrix and bias vector in the formulas (`W1`…`W4`, `b1`…`b4`) |
| `from`    | source layer (rows of the matrix) |
| `to`      | target layer (columns of the matrix) |
| `shape`   | `[rows, columns]` = `[size of from, size of to]` |
| `columns` | the node names of the target layer, in column order |
| `weights` | one entry per source node, in row order. The key is the node name, the value is its row: `weights[source][j]` is the weight from `source` to `columns[j]` |
| `biases`  | one bias per target node, in the order of `columns` |

| connection | from → to           | shape   |
|------------|---------------------|---------|
| `W1, b1`   | L1 Input → L2 Hidden 1  | 15 × 20 |
| `W2, b2`   | L2 Hidden 1 → L3 Hidden 2 | 20 × 20 |
| `W3, b3`   | L3 Hidden 2 → L4 Hidden 3 | 20 × 20 |
| `W4, b4`   | L4 Hidden 3 → L5 Output   | 20 × 15 |

Numbers are rounded to **8 decimal places**. The probabilities computed from the file differ from those of
the live network only in about the 9th decimal place (measured: 3·10⁻¹⁰), so the chosen words are the same.

A positive weight means “when the source node is active, push the target node up”; a negative weight
pushes it down.

---

## Loading the file back – ⋮ → Upload model

**Upload model** reads a `final_model.json` and replaces the network's weights and biases with the ones in the file.
It also takes over the hidden activation, the `threshold` and the pretrained/trained flags from `status`.

The whole file is checked before anything changes. If one check fails, the log says why and the current network
stays as it was. A file is accepted only if:

* `format` is `"mini-chatbot-final-model"` and `version` is `1`;
* `vocabulary.input` and `vocabulary.output` have the same words in the same order as the page (model.js);
* the hidden activation is leaky ReLU with `negativeSlope` 0.01, or tanh, and the output activation is sigmoid;
* `threshold` is a number between 0 and 1;
* there are 4 `connections`, each with exactly one row of the right length for every source node (found by node
  name) and a `biases` array of the right length, all finite numbers.

After loading, Process, Testing and the Math windows use the loaded model at once. The Adam optimiser memory is
not in the file, so if you press Pretraining or Training again, they continue from the loaded weights with a fresh optimiser.

---

## Using the file without the web page

Pseudo-code that gives the same result as the Process button:

```js
const m = JSON.parse(text);
const tokens = phrase.toLowerCase().split(/[^a-z]+/).filter(Boolean);
const index = Object.fromEntries(m.vocabulary.input.map((w) => [w.word, w.n - 1]));
const x = new Array(m.vocabulary.input.length).fill(0);
tokens.forEach((t) => {
  const w = t in index ? t : (t.endsWith("s") && t.slice(0, -1) in index ? t.slice(0, -1) : null);
  if (w) x[index[w]] = 1;                       // unknown words are ignored
});

let a = x;
m.connections.forEach((c, k) => {
  const rows = Object.values(c.weights);        // row i = source node i
  const z = c.biases.map((b, j) => b + a.reduce((s, ai, i) => s + ai * rows[i][j], 0));
  const last = k === m.connections.length - 1;
  a = z.map((v) => last ? 1 / (1 + Math.exp(-v))                                        // sigmoid
                        : v > 0 ? v : m.processing.hiddenActivation.negativeSlope * v); // leaky ReLU
});

const chosen = m.vocabulary.output.filter((w, j) => a[j] > m.processing.threshold).map((w) => w.word);
```
