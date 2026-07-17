(ns genmlx.llm.sampling
  "Pure logit-transform sampling for the owned decode loop (genmlx-djw6).

   Mirrors the NATIVE sampler's semantics exactly (mlx-core sampling.rs —
   itself a port of mlx-lm sample_utils.py), so a genmlx-provider turn at a
   given ChatConfig matches a v1 ChatSession turn on the same logits:

   - greedy iff (f32) temperature <= 1e-6 (GREEDY_TEMPERATURE_EPS) -> argmax
   - filter order: temperature -> top-k -> top-p -> min-p, then categorical
   - top-k: value threshold at the kth largest (ties at the threshold survive)
   - top-p: sorted-desc cumulative probs; keep while the PREVIOUS cumsum < p
     (the token that crosses p is included)
   - min-p: keep prob >= min-p * max-prob
   - repetition penalty (arXiv:1909.05858, asymmetric): over the last
     context-size (default 20) generated tokens, logit < 0 -> *penalty,
     logit >= 0 -> /penalty. Duplicate occurrences write identical values.
   - presence penalty (OpenAI semantics): flat subtraction for any token
     present in the context window.

   Everything here is a pure Layer-B graph transform over [vocab] logits;
   the single mx/item at the end of sample-token is the caller's eval
   boundary."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]))

(def greedy-temperature-eps
  "MUST equal the native GREEDY_TEMPERATURE_EPS (sampling.rs / C++ 1e-6f)."
  1e-6)

(defn greedy?
  "Native is_greedy: nil or (f32) temperature <= 1e-6."
  [temperature]
  (or (nil? temperature) (<= temperature greedy-temperature-eps)))

(defn apply-temperature
  "logits / temperature (identity at 1.0)."
  [logits temperature]
  (if (or (nil? temperature) (== temperature 1.0))
    logits
    (mx/multiply logits (mx/scalar (/ 1.0 temperature)))))

(defn apply-top-k
  "Keep the k largest-valued tokens (value threshold — ties survive, like the
   native argsort-position threshold); k <= 0 disables."
  [logits k]
  (if (or (nil? k) (<= k 0))
    logits
    (let [thresh (mx/amin (mx/topk logits k))]
      (mx/where (mx/greater-equal logits thresh) logits (mx/scalar js/Number.NEGATIVE_INFINITY)))))

(defn apply-top-p
  "Nucleus filter: keep the smallest prefix of descending-prob tokens whose
   previous cumulative prob is < p (so the p-crossing token is included).
   p >= 1 disables."
  [logits p]
  (if (or (nil? p) (>= p 1.0))
    logits
    (let [logprobs     (mx/subtract logits (mx/logsumexp logits))
          sorted-idx   (mx/argsort (mx/negative logprobs))          ; descending
          sorted-probs (mx/exp (mx/take-along-axis logprobs sorted-idx -1))
          prev-cum     (mx/subtract (mx/cumsum sorted-probs) sorted-probs)
          keep-sorted  (mx/less prev-cum (mx/scalar p))
          ;; smallest kept prob = the value threshold, applied in original order
          thresh       (mx/amin (mx/where keep-sorted sorted-probs (mx/scalar 2.0)))
          probs        (mx/exp logprobs)]
      (mx/where (mx/greater-equal probs thresh) logits (mx/scalar js/Number.NEGATIVE_INFINITY)))))

(defn apply-min-p
  "Keep tokens with prob >= min-p * max-prob; min-p <= 0 disables."
  [logits min-p]
  (if (or (nil? min-p) (<= min-p 0.0))
    logits
    (let [probs  (mx/softmax logits)
          thresh (mx/multiply (mx/amax probs) (mx/scalar min-p))]
      (mx/where (mx/greater-equal probs thresh) logits (mx/scalar js/Number.NEGATIVE_INFINITY)))))

(defn- penalty-indices
  "The last context-size in-vocab token ids as an int32 index array, or nil
   when nothing qualifies. context-size nil -> native default 20; <= 0 -> nil."
  [token-ids vocab context-size]
  (let [ctx (or context-size 20)
        ids (when (pos? ctx)
              (->> token-ids
                   (filter #(and (>= % 0) (< % vocab)))
                   (take-last ctx)
                   vec))]
    (when (seq ids)
      (mx/astype (mx/array ids) mx/int32))))

(defn apply-repetition-penalty
  "Asymmetric CTRL penalty over the recent token window (see ns doc)."
  ([logits token-ids penalty] (apply-repetition-penalty logits token-ids penalty nil))
  ([logits token-ids penalty context-size]
   (if (or (nil? penalty) (< (abs (- penalty 1.0)) 1e-10) (empty? token-ids))
     logits
     (let [vocab (first (mx/shape logits))]
       (if-let [idx (penalty-indices token-ids vocab context-size)]
         (let [gathered  (mx/take-along-axis logits idx -1)
               penalized (mx/where (mx/less gathered (mx/scalar 0.0))
                                   (mx/multiply gathered (mx/scalar penalty))
                                   (mx/divide gathered (mx/scalar penalty)))]
           (mx/put-along-axis logits idx penalized -1))
         logits)))))

(defn apply-presence-penalty
  "Flat subtraction for tokens present in the recent window (see ns doc)."
  ([logits token-ids penalty] (apply-presence-penalty logits token-ids penalty nil))
  ([logits token-ids penalty context-size]
   (if (or (nil? penalty) (zero? penalty) (empty? token-ids))
     logits
     (let [vocab (first (mx/shape logits))]
       (if-let [idx (penalty-indices token-ids vocab context-size)]
         (let [gathered (mx/take-along-axis logits idx -1)]
           (mx/put-along-axis logits idx
                              (mx/subtract gathered (mx/scalar penalty))
                              -1))
         logits)))))

(defn filter-logits
  "The native apply_sampling chain: temperature -> top-k -> top-p -> min-p."
  [logits {:keys [temperature top-k top-p min-p]}]
  (-> logits
      (apply-temperature temperature)
      (apply-top-k top-k)
      (apply-top-p top-p)
      (apply-min-p min-p)))

(defn sample-token
  "One native-parity sampling step. logits [vocab]; cfg keys :temperature
   :top-k :top-p :min-p :repetition-penalty :presence-penalty
   :penalty-context-size; recent-token-ids = generated-so-far (host vector).
   Returns [token-id next-key] — the mx/item here is the eval boundary."
  [key logits {:keys [temperature repetition-penalty presence-penalty
                      penalty-context-size] :as cfg} recent-token-ids]
  (let [penalized (-> logits
                      (apply-repetition-penalty recent-token-ids repetition-penalty
                                                penalty-context-size)
                      (apply-presence-penalty recent-token-ids presence-penalty
                                              penalty-context-size))]
    (if (greedy? temperature)
      [(mx/item (mx/argmax penalized)) key]
      (let [[sample-key next-key] (rng/split key)
            filtered (filter-logits penalized cfg)]
        [(mx/item (rng/categorical sample-key filtered)) next-key]))))
