;; @tier slow
(ns genmlx.llm.batched-checkevery-test
  "genmlx-lo6e D3: make-llm-gf-batched's :check-every host early-exit —
   the one 9uyg surface that shipped untested.

   Mechanism under test (core.cljs): every J sites the loop evals
   (mx/any active) and stops once every lane is dead, making the site
   count data-dependent (that is why it is opt-in). We force all-lanes-dead
   deterministically with a :hook that masks site-0 logits to a one-hot
   eos row — every lane samples eos at :t0 and goes inactive.

   Gates (hybrid 0.8b):
     C1 :check-every 1 + forced eos at t0 → the vtrace has EXACTLY one site
        (:t0), tokens [K 1]; without :check-every the same GF runs all
        max-tokens sites (trailing sites pad)
     C2 score equality: early-exited score == full-run score per lane
        (trailing pad sites contribute exactly 0 — the masked-EOS algebra)
     C3 :check-every J>1 stops at the first multiple of J at/after death
        (J=3, death at t0 → exactly 3 sites)
     C4 no-early-death control: :check-every 2 with unconstrained sampling
        that never eoses in 4 sites → all 4 sites present (the check is a
        no-op when lanes live)"
  (:require [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.core :as core]
            [genmlx.mlx.random :as rng]
            [promesa.core :as pr]
            ["fs" :as fs]
            ["os" :as os]
            ["path" :as path]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

(def model-dir
  (let [d (path/join (os/homedir) ".cache" "models" "qwen3.5-0.8b-mlx-bf16")]
    (when (.existsSync fs (path/join d "tokenizer.json")) d)))

(defn- n-sites [vtrace]
  (count (take-while #(cm/has-value? (cm/get-submap (:choices vtrace)
                                                    (keyword (str "t" %))))
                     (range))))

(defn- eos-at-t0-hook
  "Deterministic lane-killer: site 0's logits become a one-hot eos row, so
   every lane samples eos at :t0. Later sites are untouched (the inactive
   pad row takes over anyway)."
  [vocab eos]
  (let [eos-row (core/pad-onehot-row vocab eos)]
    {:init (fn [] nil)
     :mask (fn [_ logits i] (if (zero? i) eos-row logits))
     :advance (fn [s _] s)}))

(defn- summary []
  (println (str "\n== llm-batched-checkevery: " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

(if-not model-dir
  (do (println "SKIP llm-batched-checkevery — 0.8b absent") (summary))
  (->
   (pr/let [mm (llm/load-model model-dir)]
     (let [{:keys [model tokenizer]} mm
           vocab (get-in model [:fwd :config :vocab])
           eos (llm/eos-token-id tokenizer)
           hook (eos-at-t0-hook vocab eos)
           K 4
           max-toks 6]
       (pr/let [ids (llm/encode tokenizer "Hello there" false)]
         (let [prompt (vec ids)
               key (rng/fresh-key 42)

               ;; C1/C2: early exit at :check-every 1 vs the full run
               gf-early (core/make-llm-gf-batched mm {:hook hook :check-every 1})
               gf-full  (core/make-llm-gf-batched mm {:hook hook})
               vt-early (dyn/vsimulate gf-early [prompt max-toks] K key)
               vt-full  (dyn/vsimulate gf-full  [prompt max-toks] K key)]
           (assert-true (str "C1 early-exit stops at 1 site (got " (n-sites vt-early) ")")
                        (= 1 (n-sites vt-early)))
           (assert-true (str "C1 full run keeps all " max-toks " sites (got "
                             (n-sites vt-full) ")")
                        (= max-toks (n-sites vt-full)))
           (let [se (vec (mx/->clj (mx/astype (:score vt-early) mx/float32)))
                 sf (vec (mx/->clj (mx/astype (:score vt-full) mx/float32)))
                 d  (reduce max 0 (map #(js/Math.abs (- %1 %2)) se sf))]
             (assert-true (str "C2 per-lane scores equal (pad sites contribute 0; |d|=" d ")")
                          (< d 1e-5)))

           ;; C3: J=3 → stops at 3 sites exactly
           (let [gf3 (core/make-llm-gf-batched mm {:hook hook :check-every 3})
                 vt3 (dyn/vsimulate gf3 [prompt max-toks] K (rng/fresh-key 7))]
             (assert-true (str "C3 :check-every 3 stops at 3 sites (got " (n-sites vt3) ")")
                          (= 3 (n-sites vt3))))

           ;; C4: lanes alive → the check never fires, all sites present
           (let [gf-live (core/make-llm-gf-batched mm {:check-every 2})
                 vt-live (dyn/vsimulate gf-live [prompt 4] K (rng/fresh-key 11))
                 ns' (n-sites vt-live)]
             ;; greedy-ish sampling of a chat model on a text prompt does not
             ;; eos within 4 tokens for all 4 lanes; if it ever did, sites
             ;; would still be a multiple of 2 — assert full length when the
             ;; final active mask shows any live lane (honest, not flaky)
             (let [any-alive (boolean (mx/item (mx/any (:active (:retval vt-live)))))]
               (assert-true (str "C4 no-early-death control (sites=" ns'
                                 ", any-alive=" any-alive ")")
                            (or (and any-alive (= 4 ns'))
                                (and (not any-alive) (even? ns'))))))
           (mx/force-gc!)
           true))))
   (pr/then (fn [_] (summary)))
   (pr/catch (fn [e]
               (println "ERROR:" (.-message e) "\n" (.-stack e))
               (swap! fail inc)
               (summary)))))
