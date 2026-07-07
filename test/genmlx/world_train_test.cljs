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
          ;; --- Phase 1.5 (genmlx-65d5): KL-to-base penalty under autograd ---
          ;; (1) klCoef>0 now TRAINS (the rejection is gone): a true KL-to-base
          ;; penalty is wired through the autograd path, so a step applies gradients
          ;; without error instead of rejecting.
          rfn      (fn [_ c] (double (count (distinct (seq c)))))
          klres   (train/with-trainer model {:group-size 2 :kl-coef 0.1 :max-completion-length 12}
                    (fn [t]
                      (-> (train/train-step! t ["Hi there"] rfn)
                          (p/then (fn [m] (:gradients-applied? m)))
                          (p/catch (fn [_] :threw)))))
          _       (assert-true "kl-coef>0 trains under autograd (reference-model KL now wired, not rejected)"
                               (true? klres))
          ;; (2) KL-to-base wiring, theorem-anchored (genmlx-at2q, REDESIGNED).
          ;;
          ;; HISTORY: the original assert — 'strong-KL drift < KL-free drift over
          ;; 4 steps on a random tiny checkpoint' — was a single unpaired draw and
          ;; failed ~1/3 of runs. Probes (documented on the bean) showed the
          ;; PROPERTY ITSELF is not a theorem in this regime: with common random
          ;; numbers the contrast is deterministic per checkpoint yet its SIGN
          ;; varies by checkpoint (3/6 wins at beta 2 and 5; 1/6 clipped at beta
          ;; 20; a retraction variant 2/4). On a random policy the k3 KL term
          ;; contributes large gradient components, so total drift MAGNITUDE
          ;; confounds gradient magnitude with pull direction — beta>0 can move
          ;; weights MORE while still reshaping the objective toward base. The
          ;; regularization-magnitude claim belongs to real-policy scale runs
          ;; (PHASE1_FULL_TREND territory), not tiny random checkpoints.
          ;;
          ;; What IS a theorem, and what is asserted here under CRN pairing
          ;; (fresh checkpoint per trial + the `:seed` trainer config, which
          ;; seeds the MODEL THREAD's RNG at init — MLX PRNG state is
          ;; thread-local, so a caller-side seed can never reach the training
          ;; sampler; the mlx-node frozen-key compiled-categorical bug that ALSO
          ;; froze the stream is fixed alongside):
          ;;   T1 ANCHOR:     KL(ref||policy) and its k3 gradient are EXACTLY 0
          ;;                  at policy == ref, so the FIRST step of a beta=20
          ;;                  run and a beta=0 run coincide (drift-after-step-1
          ;;                  equal within float-reduction jitter).
          ;;   T2 WIRED:      by step 4 the paired trajectories DIVERGE far
          ;;                  beyond jitter — beta genuinely reshapes training
          ;;                  (a dead KL term fails T2; a mis-anchored one T1).
          run-steps (fn [t n]
                      (reduce (fn [acc _] (p/then acc (fn [_] (train/train-step! t ["Hi there"] rfn))))
                              (p/resolved nil) (range n)))
          drifts  (fn [cdir kl seed]
                    (p/let [m   (.load (.-Qwen35Model gcore) cdir)
                            base (mx/->clj (.forward m in))
                            out (train/with-trainer m
                                  {:group-size 4 :kl-coef kl :learning-rate 0.03
                                   :gradient-clip-norm 1.0 :max-completion-length 12
                                   :seed seed}
                                  (fn [t]
                                    (p/let [r1 (train/train-step! t ["Hi there"] rfn)
                                            d1 (l1 base (mx/->clj (.forward m in)))
                                            _  (run-steps t 3)
                                            d4 (l1 base (mx/->clj (.forward m in)))]
                                      {:d1 d1 :d4 d4
                                       :applied? (boolean (:gradients-applied? r1))})))]
                      out))
          rel-diff (fn [a b] (/ (js/Math.abs (- a b)) (js/Math.max (js/Math.abs b) 1e-9)))
          trial   (fn [i seed]
                    (let [cdir (str dir "-kl" i)]
                      (p/let [_  (.rmSync fs cdir #js {:recursive true :force true})
                              _  (train/random-qwen35-checkpoint! cdir {:vocabSize qwen35-vocab})
                              _  (.copyFileSync fs (.join path real-tok-dir "tokenizer.json")
                                                (.join path cdir "tokenizer.json"))
                              _  (.copyFileSync fs (.join path real-tok-dir "tokenizer_config.json")
                                                (.join path cdir "tokenizer_config.json"))
                              rk (drifts cdir 20.0 seed)
                              rf (drifts cdir 0.0 seed)]
                        (.rmSync fs cdir #js {:recursive true :force true})
                        (println (str "    ckpt " i ": step1 rel-diff " (rel-diff (:d1 rk) (:d1 rf))
                                      " | step4 rel-diff " (rel-diff (:d4 rk) (:d4 rf))
                                      " | d4 kl20=" (:d4 rk) " kl0=" (:d4 rf)))
                        {:kl rk :free rf
                         :t1? (< (rel-diff (:d1 rk) (:d1 rf)) 1e-3)
                         :t2? (> (rel-diff (:d4 rk) (:d4 rf)) 1e-3)})))
          trials  (reduce (fn [acc [i seed]]
                            (p/then acc (fn [rs] (p/then (trial i seed) #(conj rs %)))))
                          (p/resolved []) [[0 11] [1 23]])
          _       (assert-true "runs moved from base and trained (sanity, both checkpoints)"
                               (every? #(and (pos? (get-in % [:free :d4]))
                                             (get-in % [:kl :applied?])
                                             (get-in % [:free :applied?]))
                                       trials))
          _       (assert-true "T1 KL anchor: at policy==ref the beta=20 and beta=0 first steps coincide (CRN, both checkpoints)"
                               (every? :t1? trials))
          _       (assert-true "T2 KL wired: by step 4 the beta=20 trajectory diverges from beta=0 (CRN, both checkpoints)"
                               (every? :t2? trials))
          _       (clean-dir!)]
    (summary))))

(-> (run-all)
    (p/catch (fn [e]
               (swap! fail inc)
               (println "  FAIL (uncaught)" (.-message e))
               (println (.-stack e))
               (clean-dir!)
               (set! (.-exitCode js/process) 1))))
