;; @tier slow
(ns genmlx.world-train-test
  "Phase-0 acceptance (bean genmlx-zftr) for the genmlx.world.train TRAINING membrane.

   Proves the training `eval!`-equivalent round-trip end-to-end WITHOUT a real >3GB
   checkpoint (real-checkpoint training is gated by genmlx-o94r): write a tiny RANDOM
   Qwen3.5 checkpoint -> load it -> build a GRPO trainer -> run one `train-step!` with
   a PURE length reward -> assert the reward bridge fired, gradients applied, and the
   model's weights MEASURABLY changed (forward-logits L1 delta > 0). Also covers
   generate-batch (no update), the lifecycle passthroughs, optimizer-state
   checkpointing, dispose! quarantine, and with-trainer teardown-on-throw.

   Style 2 (self-contained assert + println; @tier slow) because it loads native and
   runs GPU training. Run serially: `bun run --bun nbb test/genmlx/world_train_test.cljs`."
  (:require [genmlx.world.train :as train]
            [genmlx.mlx :as mx]
            [promesa.core :as p]))

;; @genmlx/core's main is index.node (a Node-API addon) — it must be js/require'd,
;; not ESM-imported (mirrors mlx.cljs). Same for the node builtins.
(def ^:private gcore (js/require "@genmlx/core"))
(def ^:private os (js/require "os"))
(def ^:private path (js/require "path"))
(def ^:private fs (js/require "fs"))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))

(defn assert-true [label v]
  (if v
    (do (swap! pass inc) (println "  PASS" label))
    (do (swap! fail inc) (println "  FAIL" label))))

(def ^:private dir (.join path (.tmpdir os) "genmlx-phase0-world-train"))

;; The random tiny model keeps compute small (2 layers, hidden 64) but uses the REAL
;; Qwen3.5 vocab + a copied-in REAL tokenizer so the generation path inside a GRPO
;; step actually works (the random checkpoint itself writes no tokenizer).
(def ^:private real-tok-dir
  (.join path (.homedir os) ".cache" "models" "qwen3.5-0.8b-mlx-bf16"))
(def ^:private qwen35-vocab 248320)

(defn- clean-dir! []
  (.rmSync fs dir #js {:recursive true :force true}))

(defn- ids []
  (mx/astype (mx/array [[2 3 4 5]]) mx/int32))

(defn- l1
  "L1 distance between two nested CLJS number trees (forward-logits)."
  [a b]
  (reduce + (map (fn [x y] (js/Math.abs (- x y))) (flatten a) (flatten b))))

(def ^:private group-size 4)

(def ^:private cfg
  {:learning-rate 0.5 :group-size group-size :max-completion-length 12 :loss-type :grpo})

(defn- summary []
  (println (str "\n== world.train Phase 0: " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

(defn run-all []
  (println "\n== genmlx.world.train — Phase 0 acceptance ==")
  (assert-true "available? true on this box (native GRPO engine present)"
               (train/available?))
  (clean-dir!)
  (if-not (.existsSync fs (.join path real-tok-dir "tokenizer.json"))
    (do (println "  SKIP train-step round-trip — no Qwen3.5 tokenizer at" real-tok-dir
                 "(needed to make a trainable random checkpoint)")
        (summary)
        (p/resolved nil))
   (p/let [_       (train/random-qwen35-checkpoint! dir {:vocabSize qwen35-vocab})
          _       (assert-true "random-qwen35-checkpoint! wrote a checkpoint dir"
                               (.existsSync fs dir))
          ;; drop a REAL tokenizer next to the random weights so generation works
          _       (.copyFileSync fs (.join path real-tok-dir "tokenizer.json")
                                 (.join path dir "tokenizer.json"))
          _       (.copyFileSync fs (.join path real-tok-dir "tokenizer_config.json")
                                 (.join path dir "tokenizer_config.json"))
          model   (.load (.-Qwen35Model gcore) dir)
          _       (assert-true "tiny Qwen3.5 model loads from the random checkpoint"
                               (some? model))
          ;; --- weight-change proof: materialize forward logits BEFORE training ---
          in      (ids)
          before  (mx/->clj (.forward model in))
          calls   (atom 0)
          ;; distinct-char count — small, varies with completion CONTENT, so the group
          ;; has nonzero reward variance (a constant reward => zero advantage => no update).
          ;; PURE (prompt, completion)->number; deliberately does NOT forward-pass `model`
          ;; (the training thread owns it) — this is exactly the Phase-1 GFI-reward seam.
          reward-fn (fn [_prompt completion]
                      (swap! calls inc)
                      (double (count (distinct (seq completion)))))
          tr      (train/make-trainer! model cfg)
          _       (assert-true "make-trainer! returns a Trainer" (train/trainer? tr))
          _       (assert-true "fresh trainer is not disposed" (not (train/disposed? tr)))
          ;; --- THE ONE EFFECT: a GRPO step through the pure-reward bridge ---
          res     (train/train-step! tr ["Hi there"] reward-fn)
          _       (assert-true "train-step! resolves to a metrics map" (map? res))
          _       (assert-true "gradients were applied" (true? (:gradients-applied? res)))
          _       (assert-true "loss is finite" (js/isFinite (:loss res)))
          _       (assert-true "reward bridge invoked the PURE reward-fn once per completion"
                               (= group-size @calls))
          _       (assert-true "reward array length = num completions"
                               (= group-size (count (:rewards res))))
          _       (assert-true "completions returned" (= group-size (count (:completions res))))
          _       (assert-true "reward-mean is a number" (number? (:reward-mean res)))
          _       (assert-true "advantage-std > 0 (varied group rewards produced a learning signal)"
                               (> (:advantage-std res) 0.0))
          after   (mx/->clj (.forward model in))
          delta   (l1 before after)
          _       (println "    forward-logits L1 weight Δ =" delta)
          _       (assert-true "train-step! MEASURABLY changed the model's weights (forward-logits Δ > 0)"
                               (> delta 0.0))
          ;; --- optimizer-state checkpoint round-trip (moments only) ---
          optpath (.join path dir "opt-state.safetensors")
          _       (train/save-optimizer-state! tr optpath)
          _       (assert-true "save-optimizer-state! wrote a non-empty AdamW moment file"
                               (and (.existsSync fs optpath) (pos? (.-size (.statSync fs optpath)))))
          _       (train/load-optimizer-state! tr optpath)
          _       (assert-true "load-optimizer-state! restores without error" true)
          ;; --- generate-batch: completions, NO weight update ---
          before2 (mx/->clj (.forward model in))
          comps   (train/generate-batch tr ["Hi there"])
          after2  (mx/->clj (.forward model in))
          _       (assert-true "generate-batch returns a non-empty vector of completions"
                               (and (vector? comps) (pos? (count comps))))
          _       (assert-true "generate-batch does NOT change weights (Δ = 0)"
                               (= 0.0 (l1 before2 after2)))
          ;; --- lifecycle passthroughs ---
          _       (assert-true "step is a number (getter)" (number? (train/step tr)))
          _       (assert-true "epoch is a number (getter)" (number? (train/epoch tr)))
          _       (train/start-epoch! tr)
          em      (train/end-epoch! tr 0.01)
          _       (assert-true "end-epoch! returns an epoch-metrics map"
                               (and (map? em) (contains? em :avg-loss)))
          ;; --- native built-in reward registration ---
          _       (train/register-builtin-reward! tr {:reward-type :length :max-length 64 :use-chars? true})
          _       (assert-true "register-builtin-reward! registers a native builtin reward"
                               (true? (.-hasBuiltinRewards (:engine tr))))
          ;; --- dispose! quarantine ---
          _       (train/dispose! tr)
          _       (assert-true "dispose! marks the trainer disposed" (train/disposed? tr))
          _       (assert-true "an effectful op after dispose throws"
                               (try (train/start-epoch! tr) false (catch :default _ true)))
          ;; --- with-trainer: teardown on throw ---
          captured (atom nil)
          threw    (atom false)
          _       (-> (train/with-trainer model {:group-size 2}
                        (fn [t] (reset! captured t) (p/rejected (ex-info "boom" {}))))
                      (p/catch (fn [_] (reset! threw true))))
          _       (assert-true "with-trainer propagates f's rejection" @threw)
          _       (assert-true "with-trainer disposed the trainer on throw"
                               (train/disposed? @captured))
          ;; --- with-trainer: happy path returns f's value, then disposes ---
          ok      (train/with-trainer model {:group-size 2}
                    (fn [t] (p/resolved (train/step t))))
          _       (assert-true "with-trainer returns f's resolved value" (number? ok))
          ;; --- re-entry over the SAME model + multi-prompt prompt-major demux ---
          re-calls (atom 0)
          re-pairs (atom [])
          re2     (train/with-trainer model {:group-size 2 :max-completion-length 12}
                    (fn [t]
                      (train/train-step! t ["alpha" "beta"]
                        (fn [p c] (swap! re-calls inc) (swap! re-pairs conj p)
                          (double (count (distinct (seq c))))))))
          _       (assert-true "re-entry: a NEW trainer trains over the same model (dispose! freed the run)"
                               (true? (:gradients-applied? re2)))
          _       (assert-true "multi-prompt demux: reward-fn called num-prompts*group-size = 4 times"
                               (= 4 @re-calls))
          _       (assert-true "multi-prompt demux: prompt-major pairing (group 0 -> prompt 0, group 1 -> prompt 1)"
                               (= ["alpha" "alpha" "beta" "beta"] @re-pairs))
          ;; --- KL under autograd is rejected (the honest contract: klCoef>0 errors) ---
          klres   (train/with-trainer model {:group-size 2 :kl-coef 0.1 :max-completion-length 12}
                    (fn [t]
                      (-> (train/train-step! t ["Hi there"] (fn [_ c] (double (count (distinct (seq c))))))
                          (p/then (fn [_] :no-throw))
                          (p/catch (fn [_] :rejected)))))
          _       (assert-true "kl-coef>0 makes train-step! reject under autograd (reference-model KL unsupported)"
                               (= :rejected klres))
          _       (clean-dir!)]
    (summary))))

(-> (run-all)
    (p/catch (fn [e]
               (swap! fail inc)
               (println "  FAIL (uncaught)" (.-message e))
               (println (.-stack e))
               (clean-dir!)
               (set! (.-exitCode js/process) 1))))
