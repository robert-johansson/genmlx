;; genmlx-4wsn: MoE interpretability on the INSPECTABLE owned forward —
;; router inspection + expert ablation + expert steering on Ornith-1.0-35B.
;;
;; This is the owned-forward payoff demonstrated: every 35B intermediate is
;; a REPL value. The router tap reads each layer's top-8 expert routing as
;; lazy MxArrays mid-forward; the intervention rewrites router probabilities
;; BEFORE top-k — per layer, per expert, from plain Clojure data. The native
;; path exposes none of this: its forward is one opaque NAPI call
;; (logits = f(tokens)); there is no seam to observe a router or edit a
;; routing decision without recompiling the engine.
;;
;; Run (guarded, ONE GPU process):
;;   ~/genmlx-guarded-run.sh moe-interp \
;;     bunx --bun nbb@1.4.208 examples/moe_interpretability_35b.cljs
(ns examples.moe-interpretability-35b
  (:require [genmlx.mlx :as mx]
            [genmlx.llm.qwen35-forward :as q35]
            [clojure.set :as cset]
            [promesa.core :as pr]
            ["fs" :as fs]))

(def model-dir
  (or (some-> js/process .-env .-GENMLX_OWNED_MOE_MODEL)
      (let [base (str (.-HOME js/process.env)
                      "/.cache/huggingface/hub/models--mlx-community--Ornith-1.0-35B-4bit/snapshots")]
        (when (.existsSync fs base)
          (str base "/" (first (js->clj (.readdirSync fs base))))))))

;; the Qwen tokenizer via @genmlx/core directly (this demo drives the
;; q35 forward as plain values — no backend/GF layer needed)
(def core (js/require "@genmlx/core"))

(defn- layer-of [prefix]
  (js/parseInt (second (re-find #"layers\.(\d+)\." prefix))))

(defn- last-token-experts
  "From a tap log [{:prefix :top :topw}...] over a T-token forward, the
   top-k expert ids (+weights) at the LAST position, per layer."
  [taps T]
  (->> taps
       (map (fn [{:keys [prefix top topw]}]
              (let [ids (vec (last (mx/->clj (mx/astype top mx/int32))))
                    ws  (vec (last (mx/->clj topw)))]
                {:layer (layer-of prefix) :experts ids :weights ws})))
       (sort-by :layer)
       vec))

(defn- run-tapped
  "One uncached forward over `ids` with the router tap installed.
   Returns {:logits [T vocab] :taps [{:prefix :top :topw}…]}."
  [model ids]
  (let [log (atom [])   ; caller-owned tap collection (demo state)
        m (assoc-in model [:config :moe-tap]
                    (fn [prefix top topw]
                      (swap! log conj {:prefix prefix :top top :topw topw})))
        logits (q35/forward m ids)]
    (mx/eval! logits)
    {:logits logits :taps @log}))

(defn- top5 [logits-row]
  (let [lp (mx/subtract logits-row (mx/logsumexp logits-row))]
    (mx/eval! lp)
    (let [f (.toFloat32 lp)]
      (->> (range (.-length f))
           (sort-by #(aget f %) >)
           (take 5)
           (mapv (fn [i] [i (aget f i)]))))))

(defn- show-top5 [tokenizer label t5]
  (println (str "  " label ":"))
  (doseq [[id lp] t5]
    (println (str "    " (pr-str (.idToToken tokenizer id))
                  "  lp=" (.toFixed lp 3)))))

(if-not model-dir
  (println "SKIP — Ornith-1.0-35B-4bit not cached")
  (->
   (pr/let [tokenizer (.fromPretrained (.-Qwen3Tokenizer core)
                                       (str model-dir "/tokenizer.json"))
            prompt "The capital of France is"
            ids-p (.encode tokenizer prompt false)
            ids-prose (.encode tokenizer "The old lighthouse keeper watched the ships sail past" false)
            ids-code  (.encode tokenizer "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)" false)
            ids-nums  (.encode tokenizer "3.14159 2.71828 1.41421 0.57721 6.02214" false)]
     (let [model (q35/load-model model-dir)
           ids (vec ids-p)
           T (count ids)]
       (println (str "model: " model-dir))
       (println (str "prompt: " (pr-str prompt) " (" T " tokens)\n"))

       ;; ---------------------------------------------------------------
       ;; 1. ROUTER INSPECTION — which experts fire, per token, per layer
       ;; ---------------------------------------------------------------
       (println "== 1. Router inspection: per-token top-8 experts (3 sample layers) ==")
       (let [{:keys [logits taps]} (run-tapped model ids)
             tok-strs (mapv #(.idToToken tokenizer %) ids)]
         (doseq [li [0 20 39]]
           (let [{:keys [top]} (nth (sort-by (comp layer-of :prefix) taps) li)
                 rows (mx/->clj (mx/astype top mx/int32))]
             (println (str "  layer " li ":"))
             (doseq [[t es] (map-indexed vector rows)]
               (println (str "    " (pr-str (nth tok-strs t)) " -> experts " (vec es))))))
         ;; per-token expert-set overlap at a mid layer: do different tokens
         ;; route to different experts?
         (let [{:keys [top]} (nth (sort-by (comp layer-of :prefix) taps) 20)
               rows (mapv set (mx/->clj (mx/astype top mx/int32)))
               jac (fn [a b] (/ (count (cset/intersection a b))
                                (count (cset/union a b))))]
           (println (str "\n  layer-20 mean pairwise Jaccard overlap of token expert sets: "
                         (.toFixed (/ (reduce + (for [i (range T) j (range T) :when (< i j)]
                                                  (jac (nth rows i) (nth rows j))))
                                      (/ (* T (dec T)) 2))
                                   3)))
           (show-top5 tokenizer "baseline next-token top-5"
                      (top5 (mx/index logits (dec T)))))

         ;; content-type specialization: prose vs code vs digits
         (println "\n== 2. Expert specialization across content types (layer 20) ==")
         (let [probe (fn [pids]
                       (let [{:keys [taps]} (run-tapped model (vec pids))
                             {:keys [top]} (nth (sort-by (comp layer-of :prefix) taps) 20)]
                         (set (flatten (mx/->clj (mx/astype top mx/int32))))))
               prose (probe ids-prose)
               code  (probe ids-code)
               nums  (probe ids-nums)
               jac (fn [a b] (.toFixed (/ (count (cset/intersection a b))
                                          (count (cset/union a b))) 3))]
           (println (str "  distinct experts fired — prose: " (count prose)
                         ", code: " (count code) ", numbers: " (count nums)))
           (println (str "  Jaccard(prose, code)    = " (jac prose code)))
           (println (str "  Jaccard(prose, numbers) = " (jac prose nums)))
           (println (str "  Jaccard(code, numbers)  = " (jac code nums)))
           (println (str "  code-only experts (not in prose): "
                         (vec (take 10 (sort (cset/difference code prose)))))))

         ;; ---------------------------------------------------------------
         ;; 3. INTERVENTION A — ablate the answer's experts, all layers
         ;; ---------------------------------------------------------------
         (println "\n== 3. Ablation: knock out the last-token top-2 experts in EVERY layer ==")
         (let [per-layer (last-token-experts taps T)
               ;; the experts carrying the answer position, per layer
               spec (into {}
                          (map (fn [{:keys [layer experts]}]
                                 [(str "language_model.model.layers." layer ".mlp.")
                                  {:ablate (set (take 2 experts))}])
                               per-layer))
               m' (assoc-in model [:config :moe-intervene] spec)
               logits' (q35/forward m' ids)
               base-t5 (top5 (mx/index logits (dec T)))
               abl-t5  (top5 (mx/index logits' (dec T)))]
           (println (str "  (example: layer 20 ablates experts "
                         (vec (take 2 (:experts (nth per-layer 20)))) ")"))
           (show-top5 tokenizer "BEFORE (baseline)" base-t5)
           (show-top5 tokenizer "AFTER  (top-2 experts ablated per layer)" abl-t5)
           (let [flip? (not= (ffirst base-t5) (ffirst abl-t5))
                 dlp (- (second (first base-t5))
                        (or (some (fn [[id lp]] (when (= id (ffirst base-t5)) lp)) abl-t5)
                            -99.0))]
             (println (str "  argmax " (if flip? "FLIPPED" "held") "; baseline answer's "
                           "logprob dropped by " (.toFixed dlp 2) " nats")))
           ;; escalation: knock out ALL EIGHT answer-position experts per layer
           (let [spec8 (into {}
                             (map (fn [{:keys [layer experts]}]
                                    [(str "language_model.model.layers." layer ".mlp.")
                                     {:ablate (set experts)}])
                                  per-layer))
                 m8 (assoc-in model [:config :moe-intervene] spec8)
                 abl8-t5 (top5 (mx/index (q35/forward m8 ids) (dec T)))]
             (show-top5 tokenizer "AFTER (ALL 8 answer-position experts ablated per layer)" abl8-t5)
             (println (str "  argmax "
                           (if (not= (ffirst base-t5) (ffirst abl8-t5)) "FLIPPED" "held")
                           " under full answer-path ablation"))))

         ;; ---------------------------------------------------------------
         ;; 4. INTERVENTION B — steering: boost one expert everywhere
         ;; ---------------------------------------------------------------
         (println "\n== 4. Steering: pin ONE expert into every token's top-k (boost 1.0) ==")
         (let [{:keys [top]} (nth (sort-by (comp layer-of :prefix) taps) 20)
               ;; the expert the answer position uses most at layer 20
               e (first (last (mx/->clj (mx/astype top mx/int32))))
               m' (assoc-in model [:config :moe-intervene]
                            {:all {:boost {e 1.0}}})
               logits' (q35/forward m' ids)]
           (println (str "  boosting expert " e " (layer-20 answer-position top expert) "
                         "in ALL layers:"))
           (show-top5 tokenizer "AFTER (expert pinned everywhere)"
                      (top5 (mx/index logits' (dec T))))))
       (mx/force-gc!)
       (js/process.exit 0)))
   (pr/catch (fn [e]
               (println "ERROR:" (.-message e) "\n" (.-stack e))
               (js/process.exit 1)))))
