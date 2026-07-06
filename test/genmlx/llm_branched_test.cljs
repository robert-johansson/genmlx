;; @tier slow
(ns genmlx.llm-branched-test
  "Branch-using GFI (genmlx.llm.branched) on the REAL resident 80B Qwen3-Coder-Next
   (the second path). Thor/CUDA-only + heavy (loads the 42GB MoE) -> gated on the
   model dir existing; skips cleanly otherwise (mirrors the env-gated native Rust
   roundtrip test). Run:

     export GENMLX_MOE_MODEL=/path/to/Qwen3-Coder-Next-4bit/snapshots/<hash>
     bunx --bun nbb@1.4.208 -cp src:test:... test/genmlx/llm_branched_test.cljs

   Validates: (1) branched simulate builds a GFI-valid trace whose score matches the
   handler oracle within the MoE jitter band; (2) regenerate weight matches a fresh
   recompute on GENUINE moves (relative band — the 4-bit MoE has occasional ~0.7-abs
   logit jitter, argmax-stable, so we never bit-exact-assert); (3) token-MCMC keeps
   live branches BOUNDED (loser-disposal); (4) the grammar-masked path conforms."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist.core :as dc]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.protocols :as p]
            [genmlx.mlx.random :as rng]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as core]
            [genmlx.llm.grammar :as gram]
            [genmlx.llm.branched :as br]
            [promesa.core :as pr]
            ["fs" :as fs]))

(def model-dir
  (or (some-> js/process .-env .-GENMLX_MOE_MODEL)
      "/home/robert/code/mlx/models/Qwen3-Coder-Next-4bit/snapshots/7b9321eabb85ce79625cac3f61ea691e4ea984b5"))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn- check [ok? msg] (if ok? (do (swap! pass inc) (println "  ✓" msg))
                                  (do (swap! fail inc) (println "  ✗ FAIL:" msg))))
(defn- median [xs] (let [s (vec (sort xs)) n (count s)]
                     (cond (zero? n) 0.0
                           (odd? n) (nth s (quot n 2))
                           :else (/ (+ (nth s (dec (quot n 2))) (nth s (quot n 2))) 2.0))))
(defn- t-addr [i] (keyword (str "t" i)))
(defn- site-lp [logits tok] (dc/dist-log-prob (dist/categorical logits) tok))

(if-not (.existsSync fs model-dir)
  (do (println "SKIP llm_branched_test — model dir not found:" model-dir)
      (js/process.exit 0))
  (pr/let [{:keys [model tokenizer] :as mm} (llm/load-model model-dir)
           enc (llm/encode tokenizer "# Returns the ")
           prompt (vec enc)
           gf (br/make-llm-gf-branched mm)]
    (println "== llm_branched_test on" (count prompt) "-token prompt; branching?="
             (llm/supports-branching? model))
    (check (llm/supports-branching? model) "model exposes the native branch surface")
    (br/with-llm-branches* model
      (fn []
        (let [args [prompt 10], key (rng/fresh-key 7)
              tr1 (p/simulate (dyn/with-key gf key) args)
              led (br/ledger tr1)
              n (count (:toks led))]
          ;; (1) simulate trace validity + ledger
          (check (= n (count (:branches led)) (count (:logits led)) (count (:dfas led)))
                 "ledger has one branch+logits+dfa per site")
          (check (every? #(cm/has-value? (cm/get-submap (:choices tr1) (t-addr %))) (range n))
                 "trace sites :t0..:tN-1 contiguous")
          (let [{ow :weight} (p/assess (dyn/with-key (core/make-llm-gf mm) key) args (:choices tr1))
                rel (/ (js/Math.abs (- (mx/item (:score tr1)) (mx/item ow))) (+ 1 (js/Math.abs (mx/item ow))))]
            (check (< rel 0.03) (str "simulate score ≈ oracle assess (rel-Δ " (.toFixed rel 4) " < 0.03)")))

          ;; (2) regenerate weight parity vs fresh recompute on GENUINE moves
          (let [k 4, seln (sel/select (t-addr k)), old-toks (:toks led), tr1-br (set (:branches led))
                oracle-W (fn [new-toks]
                           (reduce (fn [acc j]
                                     (if (and (< j (count old-toks)) (not= j k))
                                       (let [ctx (into (vec prompt) (subvec new-toks 0 j)) tokj (nth old-toks j)]
                                         (+ acc (- (mx/item (site-lp (llm/forward-pass model ctx) tokj))
                                                   (mx/item (site-lp (nth (:logits led) j) tokj))))) acc))
                                   0.0 (range (inc k) (count old-toks))))
                rs (doall (for [s (range 10)]
                            (let [{nt :trace w :weight} (p/regenerate (dyn/with-key gf (rng/fresh-key (+ 100 s))) tr1 seln)
                                  ntk (:toks (br/ledger nt)), Wbr (mx/item w), Wor (oracle-W ntk)]
                              (doseq [b (clojure.set/difference (set (:branches (br/ledger nt))) tr1-br)]
                                (br/release-branch! model b))
                              {:g (not= (nth ntk k) (nth old-toks k)) :Wbr Wbr :Wor Wor
                               :rel (/ (js/Math.abs (- Wbr Wor)) (+ 1 (js/Math.abs Wor)))})))
                gen (filter :g rs)
                med (median (map :rel gen))]
            (check (>= (count gen) 1) (str "observed " (count gen) "/10 genuine resamples @ :t4"))
            ;; median (robust to the occasional ~0.7-abs MoE jitter) proves the weight;
            ;; outliers sit deep in the reject region (|W|~15-20) where MH is unaffected.
            (check (< med 0.06) (str "genuine-move weight ≈ fresh-recompute (median rel-Δ "
                                     (.toFixed med 4) " < 0.06)")))

          ;; (3) token-MCMC keeps live branches bounded
          (let [res (br/llm-mh-chain model gf tr1 (sel/select (t-addr 3)) 12 (rng/fresh-key 123))]
            (check (< (:max-live res) 40)
                   (str "token-MCMC live branches bounded (max-live " (:max-live res) " < 40; accept "
                        (.toFixed (:accept-rate res) 2) ")"))))))

    ;; (4) grammar-masked branched path conforms
    (let [constraint (gram/compile-constraint tokenizer "[a-z ]+")
          gcf (br/make-llm-gf-branched mm constraint)]
      (br/with-llm-branches* model
        (fn []
          (let [tr (p/simulate (dyn/with-key gcf (rng/fresh-key 5)) [prompt 8])
                txt (->> (:toks (br/ledger tr)) (map #(nth (:token-index constraint) % "")) (apply str))]
            (check (boolean (re-matches #"[a-z ]*" txt)) (str "grammar-masked simulate conforms: " (pr-str txt)))))))

    (println (str "\n== llm_branched_test: " @pass " passed, " @fail " failed =="))
    (js/process.exit (if (zero? @fail) 0 1))))
