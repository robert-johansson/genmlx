(ns bench.token-smc
  "genmlx-5qk7 V7 — the number that justifies the feature, for the TOPML RRPS
   section: token-SMC vs the baseline (N independent generations + SNIS
   reweighting) at matched posterior quality, measured in FORWARD STEPS (the
   unit that costs GPU time; on the real model each step is one decode).

   Model-free (the coupled-grammar categorical model with an enumerable
   posterior) so the comparison is exact and runs anywhere; the decoder's
   step counter stands in for decode cost 1:1.

   Baseline semantics: sample from the RAW model (no mask), keep grammar-
   valid sequences via SNIS indicator weights — every invalid sample is a
   wasted forward. token-SMC never leaves the grammar, so all its forwards
   are useful; its extra machinery is the mask normalizer + resampling.

   Output: results/token_smc/data.json
   Usage: bunx --bun nbb@1.4.208 bench/token_smc.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.llm.grammar :as grammar]
            [genmlx.llm.smc :as tsmc]
            ["fs" :as fs]))

;; --- the coupled-grammar model (see token_smc_test.cljs) -------------------
(def L [0.4 0.1 -0.2])
(def logits (mx/array L))
(def A 0) (def B 1) (def C 2)
(def T 2)

(def constraint
  {:dfa (grammar/compile-regex "(ab|ac|ba|bc)")
   :token-index ["a" "b" "c"] :eos-id 999 :masks nil})

(def valid-pairs #{[A B] [A C] [B A] [B C]})
(def eL (mapv #(js/Math.exp %) L))
(def S (reduce + eL))
(def Z-total (/ (reduce + (map (fn [[t0 t1]] (* (nth eL t0) (nth eL t1))) valid-pairs))
                (* S S)))
(def exact-post
  (into {} (map (fn [[t0 t1 :as pr]]
                  [pr (/ (/ (* (nth eL t0) (nth eL t1)) (* S S)) Z-total)])
                valid-pairs)))

(defn- tv [p q]
  (* 0.5 (reduce + (map (fn [k] (js/Math.abs (- (get p k 0.0) (get q k 0.0))))
                        (into (set (keys p)) (keys q))))))

;; --- instrumented decoder: counts forward steps (prefill + per-token) ------
(defn- counting-decoder [counter]
  (tsmc/table-decoder (fn [_ _] (swap! counter inc) logits)))

;; --- token-SMC arm ----------------------------------------------------------
(defn- smc-arm [n seed]
  (let [counter (atom 0)
        r (tsmc/token-smc {:particles n :max-tokens T :eos-id 999
                           :proposal :grammar-masked :constraint constraint
                           :decoder (counting-decoder counter)
                           :key (rng/fresh-key seed)}
                          nil [7 7])
        ws (mapv #(mx/realize (:log-w %)) (:particles r))
        m (apply max ws)
        es (mapv #(js/Math.exp (- % m)) ws)
        z (reduce + es)
        post (reduce (fn [acc [pt e]]
                       (update acc (vec (:tokens pt)) (fnil + 0.0) (/ e z)))
                     {} (map vector (:particles r) es))]
    {:tv (tv exact-post post) :forwards @counter}))

;; --- baseline arm: independent raw generations + SNIS indicator weights ----
(defn- snis-arm [n seed]
  (let [counter (atom 0)
        key0 (rng/fresh-key seed)
        keys (rng/split-n key0 n)
        samples (mapv (fn [k]
                        (let [[k0 k1] (rng/split k)
                              t0 (mx/item (dc/dist-sample (dist/categorical logits) k0))
                              _ (swap! counter inc)         ; forward for position 0
                              t1 (mx/item (dc/dist-sample (dist/categorical logits) k1))
                              _ (swap! counter inc)]        ; forward for position 1
                          [t0 t1]))
                      keys)
        valid (filterv valid-pairs samples)
        post (if (seq valid)
               (let [z (count valid)]
                 (reduce (fn [acc pr] (update acc (vec pr) (fnil + 0.0) (/ 1.0 z))) {} valid))
               {})]
    {:tv (tv exact-post post) :forwards @counter :accept-rate (/ (count valid) (double n))}))

;; --- sweep ------------------------------------------------------------------
(let [seeds [11 23 47 61 83]
      mean (fn [xs] (/ (reduce + xs) (count xs)))
      smc-n 64
      smc-runs (mapv #(smc-arm smc-n %) seeds)
      smc-tv (mean (map :tv smc-runs))
      smc-fw (mean (map :forwards smc-runs))
      ;; find the SNIS sample count whose mean TV first matches token-SMC's
      snis-curve (vec (for [n [64 128 256 512 1024 2048]]
                        (let [runs (mapv #(snis-arm n %) seeds)]
                          {:n n
                           :tv (mean (map :tv runs))
                           :forwards (mean (map :forwards runs))
                           :accept-rate (mean (map :accept-rate runs))})))
      matched (first (filter #(<= (:tv %) smc-tv) snis-curve))
      cost-ratio (when matched (/ (:forwards matched) smc-fw))
      out {:exact-log-ml (js/Math.log Z-total)
           :smc {:n smc-n :mean-tv smc-tv :mean-forwards smc-fw}
           :snis-curve snis-curve
           :matched-snis matched
           :snis-over-smc-forward-cost cost-ratio}]
  (println "token-SMC   N=" smc-n ": TV" (.toFixed smc-tv 4) "| forwards" smc-fw)
  (doseq [{:keys [n tv forwards accept-rate]} snis-curve]
    (println "SNIS        N=" n ": TV" (.toFixed tv 4) "| forwards" forwards
             "| accept" (.toFixed accept-rate 2)))
  (if matched
    (println (str "\nV7: SNIS needs ~" (.toFixed cost-ratio 1) "x token-SMC's forwards "
                  "to match TV " (.toFixed smc-tv 4)
                  " (SNIS N=" (:n matched) " vs SMC N=" smc-n ")"))
    (println "\nV7: SNIS never matched token-SMC's TV within the swept budget"))
  (when-not (.existsSync fs "results/token_smc") (.mkdirSync fs "results/token_smc" #js {:recursive true}))
  (.writeFileSync fs "results/token_smc/data.json" (js/JSON.stringify (clj->js out) nil 2))
  (println "wrote results/token_smc/data.json"))
