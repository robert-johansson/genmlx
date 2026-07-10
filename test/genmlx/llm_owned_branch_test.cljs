;; @tier slow
(ns genmlx.llm-owned-branch-test
  "genmlx-7f93: the OWNED forward's branch surface. CljsForwardModel's
   per-layer cache is a persistent vector of immutable MxArrays, so forking
   IS holding a second reference — llm/supports-branching? is now true on the
   owned path, and token-SMC / branched inference run without the O(T)/step
   replay fallback.

   Dense-0.6b gates (deterministic bf16 -> tight bands):
     B1 supports-branching? true; branch ids mint and fork
     B2 fork isolation: sibling branches and the model-internal cache are
        unaffected by a branch's steps
     B3 forward-branch == forward-step == uncached replay (argmax + band)
     B4 rope-delta transparency: install-prefill! with a synthetic delta ==
        driving fwd/step at offset+delta directly, carried across steps
        (the VLM continuation contract, genmlx-52mh, without loading a VLM)
     B5 dispose: unknown-id throws :unknown-branch-id; double-dispose no-ops
     B6 token-SMC: decoder-for picks the branch decoder on owned; forced-token
        weight increments (grammar mask log-normalizers) match the replay
        decoder within band; same-key filters agree (R1/R2, log-ML band when
        the sampled tokens coincide)
     B7 the branched GF (make-llm-gf-branched: fork-at-each-site ledger +
        llm-mh-chain) runs on the owned path; score matches the replay oracle

   Run: bunx --bun nbb@1.4.208 test/genmlx/llm_owned_branch_test.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.selection :as sel]
            [genmlx.llm.backend :as llm]
            [genmlx.llm.forward :as fwd]
            [genmlx.llm.grammar :as gram]
            [genmlx.llm.smc :as tsmc]
            [genmlx.llm.core :as core]
            [genmlx.llm.branched :as br]
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

(def dense-dir
  (let [cands [(path/join (os/homedir) ".cache" "models" "qwen3-0.6b-mlx-bf16")
               (path/join (os/homedir) ".cache" "models" "qwen3-0.6b")]]
    (or (first (filter #(.existsSync fs (path/join % "tokenizer.json")) cands))
        (first cands))))

(defn- mat [a] (mx/materialize! a) a)
(defn- max-abs-diff [a b]
  (mx/eval! a) (mx/eval! b)
  (let [fa (.toFloat32 a) fb (.toFloat32 b)]
    (reduce max 0 (map #(js/Math.abs (- (aget fa %) (aget fb %)))
                       (range (.-length fa))))))
(defn- topk-band [a b k]
  (mx/eval! a) (mx/eval! b)
  (let [fa (.toFloat32 a) fb (.toFloat32 b)
        ids (->> (range (.-length fa)) (sort-by #(aget fa %) >) (take k))]
    (reduce max 0 (map #(js/Math.abs (- (aget fa %) (aget fb %))) ids))))
(defn- lse [logits] (mx/realize (mx/logsumexp logits)))

;; grammar threading (mirrors the private helpers in genmlx.llm.smc)
(defn- gmask [c dfa logits] (gram/apply-mask c dfa logits))
(defn- gadvance [c dfa tok-id]
  (if (= tok-id (:eos-id c))
    dfa
    (gram/dfa-advance-string (:dfa c) dfa (nth (:token-index c) tok-id ""))))

(defn- summary []
  (println (str "\n== llm-owned-branch: " @pass " passed, " @fail " failed =="))
  (when (pos? @fail) (set! (.-exitCode js/process) 1)))

(if-not (.existsSync fs (path/join dense-dir "tokenizer.json"))
  (do (println "SKIP llm-owned-branch — no dense model at" dense-dir) (summary))
  (->
   (pr/let [mm (llm/load-model dense-dir)
            {:keys [model tokenizer]} mm
            enc (llm/encode tokenizer "The capital of France is")]
    (let [prompt (vec enc)
          fm (:fwd model)]
      (println "== owned branch surface on" dense-dir)

      ;; ---- B1: surface present ----
      (assert-true "B1: owned model is a CljsForwardModel" (llm/cljs-forward-model? model))
      (assert-true "B1: supports-branching? => true on the owned forward"
                   (llm/supports-branching? model))

      ;; ---- B2 + B3: fork isolation & step equivalence ----
      (llm/init-cache! model)
      (let [l0  (mat (llm/forward-prefill model prompt))
            st0 @(:cache model)               ; captured for B4 (pre-mutation)
            b1  (llm/branch-cache! model)
            b2  (llm/branch-cache! model)
            tokA (mx/item (mx/argmax l0))
            l1  (mat (llm/forward-branch model b1 tokA))
            tokB (mx/item (mx/argmax l1))
            _   (mat (llm/forward-branch model b1 tokB)) ; advance b1 further
            l2  (mat (llm/forward-branch model b2 tokA)) ; sibling, same token
            d12 (max-abs-diff l1 l2)]
        (assert-true (str "B2: sibling branch unaffected by b1's steps (max|Δ| "
                          (.toExponential d12 2) " < 1e-4)")
                     (< d12 1e-4))
        (let [l3 (mat (llm/forward-step model tokA))     ; model-internal cache
              d13 (max-abs-diff l1 l3)]
          (assert-true (str "B2: model-internal cache unaffected by branch steps (max|Δ| "
                            (.toExponential d13 2) " < 1e-4)")
                       (< d13 1e-4)))
        (let [lr (mat (llm/forward-pass model (conj prompt tokA)))
              band (topk-band lr l1 5)]
          (assert-true "B3: forward-branch argmax == uncached replay argmax"
                       (= (mx/item (mx/argmax l1)) (mx/item (mx/argmax lr))))
          (assert-true (str "B3: forward-branch vs uncached replay top-5 band ("
                            (.toFixed band 4) " < 0.3)")
                       (< band 0.3)))

        ;; ---- B4: rope-delta transparency ----
        (let [delta 7
              c0 (:cache st0) T (:offset st0)]
          (llm/install-prefill! model {:cache c0 :seq-len T :rope-delta delta})
          (let [bd (llm/branch-cache! model)
                lb1 (mat (llm/forward-branch model bd tokA))
                [lo1 c1] (fwd/step fm c0 (+ T delta) tokA)
                _ (mat lo1)
                d1 (max-abs-diff lb1 lo1)
                ;; second step: the delta must persist on the branch entry
                lb2 (mat (llm/forward-branch model bd tokB))
                [lo2 _] (fwd/step fm c1 (+ T 1 delta) tokB)
                _ (mat lo2)
                d2 (max-abs-diff lb2 lo2)
                ;; and on the model-internal cell via forward-step
                ls1 (mat (llm/forward-step model tokA))
                d3 (max-abs-diff ls1 lo1)]
            (assert-true (str "B4: branch step rotates at offset+delta (max|Δ| "
                              (.toExponential d1 2) " < 1e-4)")
                         (< d1 1e-4))
            (assert-true (str "B4: delta persists across branch steps (max|Δ| "
                              (.toExponential d2 2) " < 1e-4)")
                         (< d2 1e-4))
            (assert-true (str "B4: forward-step honors installed rope-delta (max|Δ| "
                              (.toExponential d3 2) " < 1e-4)")
                         (< d3 1e-4))
            ;; delta actually matters: rotating WITHOUT it must diverge
            (let [[lo-no _] (fwd/step fm c0 T tokA)
                  d-no (max-abs-diff lb1 (mat lo-no))]
              (assert-true (str "B4: delta is not a no-op (max|Δ| vs delta-less "
                                (.toExponential d-no 2) " > 1e-3)")
                           (> d-no 1e-3)))
            (llm/dispose-branch! model bd)))

        ;; ---- B5: dispose semantics ----
        (llm/dispose-branch! model b1)
        (assert-true "B5: forward-branch on a disposed id throws :unknown-branch-id"
                     (try (llm/forward-branch model b1 tokA) false
                          (catch :default e
                            (= :unknown-branch-id (:genmlx/error (ex-data e))))))
        (assert-true "B5: branch-from on a disposed id throws :unknown-branch-id"
                     (try (llm/branch-from model b1) false
                          (catch :default e
                            (= :unknown-branch-id (:genmlx/error (ex-data e))))))
        (assert-true "B5: double-dispose is a no-op"
                     (nil? (llm/dispose-branch! model b1)))
        (llm/dispose-branch! model b2)
        (llm/reset-cache! model))

      ;; ---- B6: token-SMC on the owned branch surface ----
      (pr/let [constraint (gram/compile-constraint tokenizer "[0-9]{3}-[0-9]{4}")
               enc2 (llm/encode tokenizer "My phone number is ")]
        (let [prompt2 (vec enc2)
              d-auto (tsmc/decoder-for mm)]
          (assert-true "B6: decoder-for picks the branch decoder on owned"
                       (instance? tsmc/NativeDecoder d-auto))

          ;; forced-token weight-increment parity: branch vs replay logits
          ;; along the SAME greedy-masked token path (deterministic; the
          ;; quantity that IS the SMC weight: the mask log-normalizer).
          (let [steps 7
                walk (fn [decoder]
                       (let [{:keys [root logits]} (tsmc/dec-prefill! decoder prompt2)
                             h (tsmc/dec-fork! decoder root)]
                         (loop [i 0, lg logits, dfa (:start (:dfa constraint)), out []]
                           (if (>= i steps)
                             (do (tsmc/dec-dispose! decoder h)
                                 (tsmc/dec-dispose! decoder root)
                                 out)
                             (let [masked (gmask constraint dfa lg)
                                   inc-w (- (lse masked) (lse lg))
                                   tok (mx/item (mx/argmax masked))
                                   lg' (mat (tsmc/dec-step! decoder h tok))]
                               (recur (inc i) lg' (gadvance constraint dfa tok)
                                      (conj out {:tok tok :inc-w inc-w})))))))
                wb (walk (tsmc/native-decoder model))
                wr (walk (tsmc/replay-decoder model))
                same-toks? (= (mapv :tok wb) (mapv :tok wr))
                dw (when same-toks?
                     (reduce max 0 (map #(js/Math.abs (- (:inc-w %1) (:inc-w %2))) wb wr)))]
            (assert-true "B6: greedy-masked token path identical branch vs replay"
                         same-toks?)
            ;; Band 0.3 = the platform's cached-vs-uncached numeric band (the
            ;; parity suite's step-vs-full gate): each increment is a
            ;; difference of two logsumexps whose logits differ by up to
            ;; ~0.1 between the cached branch path and the uncached replay
            ;; (measured 0.106 on this host's dense bf16).
            (when same-toks?
              (assert-true (str "B6: per-site weight increments match replay (max|Δ| "
                                (.toFixed dw 5) " < 0.3)")
                           (< dw 0.3))))

          ;; full filter, both decoders, same key
          (let [run! (fn [decoder]
                       (let [max-live (atom 0)
                             r (tsmc/token-smc
                                {:particles 4 :max-tokens 10
                                 :eos-id (llm/eos-token-id tokenizer)
                                 :proposal :grammar-masked :constraint constraint
                                 :decoder decoder :key (rng/fresh-key 42)
                                 :callback (fn [_] (swap! max-live max (tsmc/live-handles decoder)))}
                                mm prompt2)]
                         {:r r :max-live @max-live :decoder decoder}))
                br (run! (tsmc/native-decoder model))
                rp (run! (tsmc/replay-decoder model))
                ml-br (mx/realize (:log-ml-estimate (:r br)))
                ml-rp (mx/realize (:log-ml-estimate (:r rp)))
                toks= (= (frequencies (map :tokens (:particles (:r br))))
                         (frequencies (map :tokens (:particles (:r rp)))))]
            (assert-true (str "B6: R1 bounded on the owned branch decoder ("
                              (:max-live br) " <= 5)")
                         (<= (:max-live br) 5))
            (assert-true "B6: R2 no leak after return"
                         (zero? (tsmc/live-handles (:decoder br))))
            (assert-true "B6: finite log-ML on the owned branch decoder"
                         (js/isFinite ml-br))
            (println (str "    [info] log-ML branch=" (.toFixed ml-br 4)
                          " replay=" (.toFixed ml-rp 4)
                          " same-tokens?=" toks=))
            (if toks=
              (assert-true (str "B6: same-key log-ML matches replay (|Δ| "
                                (.toFixed (js/Math.abs (- ml-br ml-rp)) 4) " < 0.5)")
                           (< (js/Math.abs (- ml-br ml-rp)) 0.5))
              (println "    [info] sampled tokens diverged (near-tie flip); log-ML band not asserted")))

          ;; ---- B7: the branched GF (fork ledger + token-MCMC) on owned ----
          (let [gf (br/make-llm-gf-branched mm)]
            (br/with-llm-branches* model
              (fn []
                (let [tr (p/simulate (dyn/with-key gf (rng/fresh-key 7)) [prompt2 6])
                      led (br/ledger tr)
                      n (count (:toks led))
                      {ow :weight} (p/assess (dyn/with-key (core/make-llm-gf mm) (rng/fresh-key 7))
                                             [prompt2 6] (:choices tr))
                      rel (/ (js/Math.abs (- (mx/realize (:score tr)) (mx/realize ow)))
                             (+ 1 (js/Math.abs (mx/realize ow))))]
                  (assert-true (str "B7: branched simulate builds a ledger on owned ("
                                    n " sites)")
                               (= n (count (:branches led))))
                  (assert-true (str "B7: branched score ≈ replay-oracle assess (rel-Δ "
                                    (.toFixed rel 4) " < 0.03)")
                               (< rel 0.03))
                  (let [res (br/llm-mh-chain model gf tr (sel/select :t0) 6 (rng/fresh-key 11))]
                    (assert-true (str "B7: token-MCMC on owned branches bounded (max-live "
                                      (:max-live res) " < 30)")
                                 (< (:max-live res) 30)))))))
          (summary)))))
   (pr/catch (fn [e]
               (swap! fail inc)
               (println "  FAIL (uncaught)" (or (.-message e) e))
               (summary)))))
