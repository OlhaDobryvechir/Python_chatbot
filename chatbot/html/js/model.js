/*
 * model.js – the data and the shape of the Mini chatbot neural network.
 *
 * Idea: a tiny "language" about animals.
 *   input  phrase = [animal] [state] [situation]   e.g. "dog hungry rain"
 *   output phrase = what the animal does / how it is   "barks eats wet"
 *
 * Every input word has a real English meaning that "causes" one output word:
 *   cat -> meows, dog -> barks, bird -> sings, fish -> swims, cow -> moos
 *   hungry -> eats, thirsty -> drinks, tired -> sleeps, scared -> hides, happy -> plays
 *   rain -> wet, sun -> warm, snow -> cold, night -> dark, garden -> outside
 *
 * The network never sees these rules – it must discover them from examples.
 *
 *   PRETRAINING  (large set, 140 records): "general knowledge" – every single word,
 *                every word pair and 50 full phrases (like a language model that
 *                reads a lot of text and learns word associations).
 *   TRAINING     (small set, 12 records): new full three-word phrases – the task
 *                we care about (like fine-tuning). While training, the pretraining
 *                records are replayed too, so the network does not forget them.
 *   TESTING      (20 records): three-word combinations the network has NEVER
 *                seen in any set. Correct answers here prove that the network
 *                generalises: it learned the meaning of each word, not a lookup table.
 */
function createModel() {
  "use strict";

  // ---- vocabularies (number = position + 1) ---------------------------------
  const inputWords = [
    // animals
    { n: 1, word: "cat", group: "animal" },
    { n: 2, word: "dog", group: "animal" },
    { n: 3, word: "bird", group: "animal" },
    { n: 4, word: "fish", group: "animal" },
    { n: 5, word: "cow", group: "animal" },
    // states
    { n: 6, word: "hungry", group: "state" },
    { n: 7, word: "thirsty", group: "state" },
    { n: 8, word: "tired", group: "state" },
    { n: 9, word: "scared", group: "state" },
    { n: 10, word: "happy", group: "state" },
    // situations
    { n: 11, word: "rain", group: "situation" },
    { n: 12, word: "sun", group: "situation" },
    { n: 13, word: "snow", group: "situation" },
    { n: 14, word: "night", group: "situation" },
    { n: 15, word: "garden", group: "situation" },
  ];

  const outputWords = [
    // what the animal says / does by nature
    { n: 1, word: "meows", group: "sound" },
    { n: 2, word: "barks", group: "sound" },
    { n: 3, word: "sings", group: "sound" },
    { n: 4, word: "swims", group: "sound" },
    { n: 5, word: "moos", group: "sound" },
    // reaction to the state
    { n: 6, word: "eats", group: "action" },
    { n: 7, word: "drinks", group: "action" },
    { n: 8, word: "sleeps", group: "action" },
    { n: 9, word: "hides", group: "action" },
    { n: 10, word: "plays", group: "action" },
    // effect of the situation
    { n: 11, word: "wet", group: "condition" },
    { n: 12, word: "warm", group: "condition" },
    { n: 13, word: "cold", group: "condition" },
    { n: 14, word: "dark", group: "condition" },
    { n: 15, word: "outside", group: "condition" },
  ];

  // ---- network: 5 layers = input + 3 hidden + output --------------------------
  const layers = [
    { name: "Input", size: 15 },
    { name: "Hidden 1", size: 20 },
    { name: "Hidden 2", size: 20 },
    { name: "Hidden 3", size: 20 },
    { name: "Output", size: 15 },
  ];

  // ---- PRETRAINING: 140 records -----------------------------------------------
  const pretrainSet = [
    // single words (15)
    "cat => meows", "dog => barks", "bird => sings", "fish => swims", "cow => moos",
    "hungry => eats", "thirsty => drinks", "tired => sleeps", "scared => hides", "happy => plays",
    "rain => wet", "sun => warm", "snow => cold", "night => dark", "garden => outside",

    // animal + state (25)
    "cat hungry => meows eats", "cat thirsty => meows drinks", "cat tired => meows sleeps",
    "cat scared => meows hides", "cat happy => meows plays", "dog hungry => barks eats",
    "dog thirsty => barks drinks", "dog tired => barks sleeps", "dog scared => barks hides",
    "dog happy => barks plays", "bird hungry => sings eats", "bird thirsty => sings drinks",
    "bird tired => sings sleeps", "bird scared => sings hides", "bird happy => sings plays",
    "fish hungry => swims eats", "fish thirsty => swims drinks", "fish tired => swims sleeps",
    "fish scared => swims hides", "fish happy => swims plays", "cow hungry => moos eats",
    "cow thirsty => moos drinks", "cow tired => moos sleeps", "cow scared => moos hides",
    "cow happy => moos plays",

    // animal + situation (25)
    "cat rain => meows wet", "cat sun => meows warm", "cat snow => meows cold",
    "cat night => meows dark", "cat garden => meows outside", "dog rain => barks wet",
    "dog sun => barks warm", "dog snow => barks cold", "dog night => barks dark",
    "dog garden => barks outside", "bird rain => sings wet", "bird sun => sings warm",
    "bird snow => sings cold", "bird night => sings dark", "bird garden => sings outside",
    "fish rain => swims wet", "fish sun => swims warm", "fish snow => swims cold",
    "fish night => swims dark", "fish garden => swims outside", "cow rain => moos wet",
    "cow sun => moos warm", "cow snow => moos cold", "cow night => moos dark",
    "cow garden => moos outside",

    // state + situation (25)
    "hungry rain => eats wet", "hungry sun => eats warm", "hungry snow => eats cold",
    "hungry night => eats dark", "hungry garden => eats outside", "thirsty rain => drinks wet",
    "thirsty sun => drinks warm", "thirsty snow => drinks cold", "thirsty night => drinks dark",
    "thirsty garden => drinks outside", "tired rain => sleeps wet", "tired sun => sleeps warm",
    "tired snow => sleeps cold", "tired night => sleeps dark", "tired garden => sleeps outside",
    "scared rain => hides wet", "scared sun => hides warm", "scared snow => hides cold",
    "scared night => hides dark", "scared garden => hides outside", "happy rain => plays wet",
    "happy sun => plays warm", "happy snow => plays cold", "happy night => plays dark",
    "happy garden => plays outside",

    // full phrases (50)
    "cat hungry snow => meows eats cold", "dog tired night => barks sleeps dark",
    "dog scared snow => barks hides cold", "fish scared garden => swims hides outside",
    "fish thirsty rain => swims drinks wet", "dog scared rain => barks hides wet",
    "cat thirsty night => meows drinks dark", "fish hungry garden => swims eats outside",
    "bird thirsty rain => sings drinks wet", "cow scared night => moos hides dark",
    "cat happy night => meows plays dark", "dog happy garden => barks plays outside",
    "cat scared night => meows hides dark", "fish thirsty snow => swims drinks cold",
    "cat scared garden => meows hides outside", "cat hungry night => meows eats dark",
    "dog tired rain => barks sleeps wet", "dog happy rain => barks plays wet",
    "bird hungry snow => sings eats cold", "bird scared night => sings hides dark",
    "cow thirsty rain => moos drinks wet", "bird scared rain => sings hides wet",
    "cat hungry garden => meows eats outside", "dog thirsty garden => barks drinks outside",
    "bird tired sun => sings sleeps warm", "cow thirsty night => moos drinks dark",
    "fish scared rain => swims hides wet", "bird happy sun => sings plays warm",
    "bird tired snow => sings sleeps cold", "cat happy sun => meows plays warm",
    "cat thirsty rain => meows drinks wet", "bird hungry garden => sings eats outside",
    "dog hungry snow => barks eats cold", "bird tired rain => sings sleeps wet",
    "fish happy garden => swims plays outside", "bird thirsty night => sings drinks dark",
    "bird hungry rain => sings eats wet", "fish thirsty sun => swims drinks warm",
    "fish scared snow => swims hides cold", "bird tired night => sings sleeps dark",
    "cow happy garden => moos plays outside", "cat happy garden => meows plays outside",
    "cow happy night => moos plays dark", "fish scared night => swims hides dark",
    "fish happy sun => swims plays warm", "cow thirsty sun => moos drinks warm",
    "cat tired sun => meows sleeps warm", "fish happy rain => swims plays wet",
    "bird thirsty garden => sings drinks outside", "dog hungry sun => barks eats warm",
  ];

  // ---- TRAINING: 12 records (full phrases, the real task) --------------------
  const trainSet = [
    "dog scared night => barks hides dark",
    "cat hungry sun => meows eats warm",
    "cow tired garden => moos sleeps outside",
    "cow scared rain => moos hides wet",
    "fish happy snow => swims plays cold",
    "fish scared sun => swims hides warm",
    "bird happy rain => sings plays wet",
    "cow scared snow => moos hides cold",
    "cat tired garden => meows sleeps outside",
    "fish tired sun => swims sleeps warm",
    "fish tired night => swims sleeps dark",
    "cow hungry snow => moos eats cold",
  ];

  // ---- TESTING: 20 records – combinations never seen before -------------------
  const testSet = [
    "cow happy sun => moos plays warm",
    "bird happy snow => sings plays cold",
    "bird tired garden => sings sleeps outside",
    "cow hungry garden => moos eats outside",
    "cow scared garden => moos hides outside",
    "fish hungry snow => swims eats cold",
    "bird hungry night => sings eats dark",
    "cat tired rain => meows sleeps wet",
    "cow scared sun => moos hides warm",
    "fish thirsty night => swims drinks dark",
    "bird happy garden => sings plays outside",
    "dog scared sun => barks hides warm",
    "fish tired garden => swims sleeps outside",
    "cow thirsty snow => moos drinks cold",
    "dog happy snow => barks plays cold",
    "cat hungry rain => meows eats wet",
    "cat scared rain => meows hides wet",
    "bird thirsty sun => sings drinks warm",
    "cow hungry night => moos eats dark",
    "cow hungry rain => moos eats wet",
  ];

  // ---- MATH EXAMPLE: 1 new record, in none of the sets above --------------------
  // "Training Math" (⋮ menu) uses it to show one training step with all the numbers.
  const mathTrainExample = "dog thirsty snow => barks drinks cold";

  // ---- helpers -----------------------------------------------------------------
  const inIndex = Object.fromEntries(inputWords.map((w, i) => [w.word, i]));
  const outIndex = Object.fromEntries(outputWords.map((w, i) => [w.word, i]));

  function encode(words, index, size) {
    const v = new Array(size).fill(0);
    words.forEach((w) => { if (w in index) v[index[w]] = 1; });
    return v;
  }

  function parseRecord(text) {
    const [left, right] = text.split("=>").map((s) => s.trim().toLowerCase().split(/\s+/).filter(Boolean));
    return {
      text,
      inWords: left,
      outWords: right,
      x: encode(left, inIndex, inputWords.length),
      y: encode(right, outIndex, outputWords.length),
    };
  }

  /** Turn free text typed by the user into an input vector. */
  function encodeText(text) {
    const tokens = text.toLowerCase().split(/[^a-z]+/).filter(Boolean);
    const known = [], unknown = [];
    tokens.forEach((t) => {
      // small forgiveness: "cats" -> "cat", "dogs" -> "dog"
      const w = t in inIndex ? t : (t.endsWith("s") && t.slice(0, -1) in inIndex ? t.slice(0, -1) : null);
      if (w) { if (!known.includes(w)) known.push(w); } else unknown.push(t);
    });
    return { known, unknown, x: encode(known, inIndex, inputWords.length) };
  }

  // how a "condition" output word is written in the sentence
  const conditionPhrases = { wet: "gets wet", warm: "feels warm", cold: "feels cold", dark: "is in the dark", outside: "is outside" };

  /** Make a readable English sentence from predicted output words. */
  function sentence(inWords, outWords) {
    const animal = inWords.find((w) => inputWords[inIndex[w]] && inputWords[inIndex[w]].group === "animal");
    const g = (grp) => outWords.filter((w) => outputWords[outIndex[w]].group === grp);
    const sound = g("sound"), act = g("action"), cond = g("condition");
    const verbs = [...sound, ...act, ...cond.map((c) => conditionPhrases[c])];
    if (!verbs.length) return "(no confident output)";
    const subj = animal ? `The ${animal}` : "It";
    const last = verbs.pop();
    return `${subj} ${verbs.length ? verbs.join(", ") + " and " : ""}${last}.`;
  }

  return {
    inputWords, outputWords, layers,
    pretrain: pretrainSet.map(parseRecord),
    train: trainSet.map(parseRecord),
    test: testSet.map(parseRecord),
    mathExample: parseRecord(mathTrainExample),
    hyper: {
      seed: 2008,
      activation: "lrelu",          // hidden layers: leaky ReLU
      weightDecay: 0.05,            // keeps weights small -> simpler, more general rules
      pretrain: { epochs: 800, lr: 0.01 },
      train: { epochs: 200, lr: 0.003, replay: true },
      threshold: 0.5,               // an output word is "on" when its probability > 0.5
    },
    encodeText, sentence, parseRecord, conditionPhrases,
  };
}

if (typeof module !== "undefined") module.exports = { createModel };
