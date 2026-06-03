(ns genmlx.plate-generate-test
  "Regression test for genmlx-7bm6: p/generate silently dropped observation
   constraints when a gen body invoked a TRACING helper (one that receives the
   ambient `trace` and loops, tracing many hidden sites) N times in one body —
   a 'plate'. The posterior collapsed to the prior with no error raised.

   Root cause: the schema walker only sees literal `(trace addr dist)` calls.
   When a body hands the `trace` binding to an opaque call — e.g.
   `(run-block trace ...)` — the trace sites inside are invisible, so the model
   was wrongly classified `:static? true` and compiled to L1-M2. The L1-M2
   compiled path evaluates only the visible static sites + the return form,
   DROPPING non-final body statements. With the opaque tracing call as a
   non-final statement (`(run-block trace ...) :done`), it was never run: zero
   loop sites traced, every constraint dropped, weight 0.

   Fix: detect `trace`/`splice` escaping head position into opaque code
   (`:opaque-gen-escape?`), disqualify `:static?`, and route such models to the
   handler path (ground truth) with no compilation.

   These models mirror the bug WITHOUT depending on mct.controlled-loop: the
   inline `run-block` is a plain defn that receives the ambient `trace` and runs
   a loop/recur, branching on the read-back `(mx/item s)` — exactly the shape
   that hides its trace sites from the schema walker."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.schema :as schema]
            [genmlx.gfi :as gfi]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.selection :as sel]))

(def ^:private MAX-ITERS 40)
(def ^:private SM-P 0.6)
(def ^:private TRUE-THETA 0.55)

;; Inline controlled loop. A plain defn that takes the AMBIENT `trace` and loops,
;; tracing <prefix>monT then <prefix>stopT and branching on the read-back stop
;; value. The schema walker cannot see these sites (trace is handed to opaque
;; code) — the exact pattern that triggered the bug.
(defn- run-block [trace prefix p-stop]
  (loop [t 0]
    (trace (keyword (str prefix "mon" t)) (dist/bernoulli (mx/scalar SM-P)))
    (let [s     (trace (keyword (str prefix "stop" t)) (dist/bernoulli p-stop))
          fired? (>= (mx/item s) 1.0)
          last?  (>= (inc t) MAX-ITERS)]
      (if (or fired? last?) t (recur (inc t))))))

(defn- p-stop-of [theta]
  (mx/clip (mx/multiply (mx/scalar 0.5) theta) 1e-6 (- 1.0 1e-6)))

;; PLATE: opaque tracing call invoked N times as a NON-FINAL statement, literal
;; return. This is the failing shape.
(def plate-model
  (gen [n-blocks]
    (let [theta  (trace :theta (dist/uniform 0 1))
          p-stop (p-stop-of theta)]
      (loop [b 0]
        (when (< b n-blocks)
          (run-block trace (str "b" b "_") p-stop)
          (recur (inc b))))
      :done)))

;; SINGLE block, opaque call as the RETURN form. (Worked "by accident" pre-fix;
;; post-fix it correctly routes to the handler too.)
(def single-model
  (gen []
    (let [theta  (trace :theta (dist/uniform 0 1))
          p-stop (p-stop-of theta)]
      (run-block trace "" p-stop))))

;; Control: a genuinely static model — must NOT be falsely flagged as escaping,
;; and must still compile (no regression).
(def static-model
  (gen [] (let [a (trace :a (dist/gaussian 0 1))]
            (trace :b (dist/gaussian a 1)))))

;; Second hiding mechanism (no `trace` ever appears as a value): an inner fn
;; traces in HEAD position but is handed to an opaque higher-order function,
;; which decides how many times it runs. The single-shot compiled path would
;; under-count the site; must route to the handler instead.
(def hof-model
  (gen [n]
    (let [step (fn [i] (trace (keyword (str "x" i)) (dist/gaussian 0 1)))]
      (run! step (range n))
      :done)))

;; Third hiding mechanism: a `letfn`-bound tracing function handed to a HOF. The
;; name `step` is neither the bare `trace` binding nor an fn-literal, so it slips
;; past mechanisms 1 and 2 — but it is the same indirectly-invoked capability.
(def letfn-model
  (gen [n]
    (letfn [(step [i] (trace (keyword (str "x" i)) (dist/gaussian 0 1)))]
      (run! step (range n))
      :done)))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- all-constraints
  "Build a constraint choicemap from every leaf of a trace's choices."
  [ch]
  (reduce (fn [cm path] (cm/set-choice cm path (cm/get-choice ch path)))
          cm/EMPTY (cm/addresses ch)))

(defn- without-addr
  "Constraint choicemap of all leaves of `ch` except the given top-level addr."
  [ch drop-addr]
  (reduce (fn [cm path]
            (if (= [drop-addr] path)
              cm
              (cm/set-choice cm path (cm/get-choice ch path))))
          cm/EMPTY (cm/addresses ch)))

(defn- gen-weight [gf args constraints]
  (mx/item (:weight (p/generate (dyn/with-key gf (rng/fresh-key 0)) args constraints))))

;; ---------------------------------------------------------------------------
;; 1. Schema classification — the root cause level
;; ---------------------------------------------------------------------------

(deftest schema-classification
  (testing "trace escaping into an opaque call is detected"
    (let [s (:schema plate-model)]
      (is (true? (:opaque-gen-escape? s)) "plate body escapes trace -> flagged")
      (is (false? (:static? s)) "escaping body is NOT static")
      (is (nil? (:compiled-simulate s)) "no compiled-simulate attached")
      (is (nil? (:compiled-prefix s)) "no compiled-prefix attached")
      (is (nil? (:auto-handlers s)) "no analytical handlers attached")))

  (testing "return-form escape is detected too (single-block model)"
    (let [s (:schema single-model)]
      (is (true? (:opaque-gen-escape? s)) "single-block escape flagged")
      (is (false? (:static? s)) "single-block not static")
      (is (nil? (:compiled-simulate s)) "single-block has no compiled path")))

  (testing "dispatch routes escaping models to the handler for every op"
    (doseq [op [:simulate :generate :update :regenerate :assess :project]]
      (is (= :handler (:label (dyn/resolve-dispatch plate-model op)))
          (str "plate " op " dispatches to handler"))))

  (testing "genuinely static models are NOT false-flagged (no regression)"
    (let [s (:schema static-model)]
      (is (false? (boolean (:opaque-gen-escape? s))) "static model not flagged")
      (is (true? (:static? s)) "static model stays static")
      (is (some? (:compiled-simulate s)) "static model still compiles (L1-M2)")))

  (testing "tracing fn-literal handed to an opaque HOF is detected (mechanism 2)"
    (let [s (:schema hof-model)]
      (is (true? (:opaque-gen-escape? s)) "HOF-driven inner-fn trace -> flagged")
      (is (false? (:static? s)) "HOF model is not static")
      (is (nil? (:compiled-simulate s)) "HOF model has no compiled path")))

  (testing "letfn-bound tracing fn handed to an opaque HOF is detected (mechanism 3)"
    (let [s (:schema letfn-model)]
      (is (true? (:opaque-gen-escape? s)) "letfn-bound tracer -> flagged")
      (is (false? (:static? s)) "letfn model is not static")
      ;; project's M2 builder (unlike the others) succeeds on this shape, so
      ;; without the flag it would wrongly dispatch :project to :compiled.
      (is (nil? (:compiled-project s)) "letfn model has no compiled-project")
      (is (= :handler (:label (dyn/resolve-dispatch letfn-model :project)))
          "letfn project dispatches to handler")))

  (testing "escapes-gen-binding detection edge cases"
    ;; head-position trace is fine; arg-position trace escapes; renamed binding escapes.
    (is (false? (:opaque-gen-escape?
                 (schema/extract-schema '([] (trace :x (dist/gaussian 0 1))))))
        "(trace :x ...) head position is not an escape")
    (is (true? (:opaque-gen-escape?
                (schema/extract-schema '([] (helper trace 1)))))
        "(helper trace 1) arg position escapes")
    (is (true? (:opaque-gen-escape?
                (schema/extract-schema '([] (let [tr trace] (tr :x (dist/gaussian 0 1)))))))
        "let-rebinding trace escapes")
    (is (true? (:opaque-gen-escape?
                (schema/extract-schema '([] (loop [b 0] (run-block trace b) (recur (inc b)))))))
        "trace inside an opaque call in a loop escapes")
    ;; mechanism 2: tracing fn-literal as a value escapes; immediate-invoke is fine.
    (is (true? (:opaque-gen-escape?
                (schema/extract-schema '([] (run! (fn [i] (trace :a (dist/gaussian 0 1))) xs)))))
        "tracing fn-literal passed to a HOF escapes")
    (is (true? (:opaque-gen-escape?
                (schema/extract-schema '([] (mapv #(trace :a (dist/gaussian 0 1)) xs)))))
        "tracing reader-fn passed to a HOF escapes")
    (is (true? (:opaque-gen-escape?
                (schema/extract-schema '([] (let [f (fn [] (trace :a (dist/gaussian 0 1)))] (f))))))
        "tracing fn-literal stored in a let escapes")
    (is (false? (:opaque-gen-escape?
                 (schema/extract-schema '([] ((fn [] (trace :a (dist/gaussian 0 1))))))))
        "immediately-invoked tracing fn runs once — NOT an escape")
    (is (false? (:opaque-gen-escape?
                 (schema/extract-schema '([] (mapv (fn [x] (* x x)) xs)))))
        "non-tracing fn-literal passed to a HOF is fine")
    ;; mechanism 3: letfn-bound tracer escapes; param-only letfn does NOT.
    (is (true? (:opaque-gen-escape?
                (schema/extract-schema '([xs] (letfn [(step [i] (trace :y (dist/gaussian 0 1)))]
                                                (run! step xs))))))
        "letfn-bound tracing fn escapes")
    (is (false? (:opaque-gen-escape?
                 (schema/extract-schema '([xs] (letfn [(g [i] (* i i))] (mapv g xs))))))
        "letfn with no trace is fine")
    (is (false? (:opaque-gen-escape?
                 (schema/extract-schema '([] (mapv (fn [x] (param :w)) xs)))))
        "param-only fn-literal is NOT a tracing capability (param exclusion)")))

;; ---------------------------------------------------------------------------
;; 2. The constraint-drop bug — full re-constraint must score everything
;; ---------------------------------------------------------------------------

(deftest full-reconstraint-scores-all-sites
  (testing "generate with ALL leaves constrained => weight == simulate score"
    (doseq [n [1 2 4]]
      (let [k     (rng/fresh-key (+ 100 n))
            tr    (p/simulate (dyn/with-key plate-model k) [n])
            ch    (:choices tr)
            score (mx/item (:score tr))
            cons  (all-constraints ch)
            n-leaves (count (cm/addresses cons))
            r     (p/generate (dyn/with-key plate-model k) [n] cons)
            w     (mx/item (:weight r))]
        ;; the heart of the bug: pre-fix, only :theta was scored => weight ~ 0
        (is (> n-leaves (inc n)) (str "n=" n ": plate has many hidden sites (" n-leaves ")"))
        (is (h/finite? w) (str "n=" n ": weight finite"))
        (is (h/close? w score 1e-3)
            (str "n=" n ": full-constraint weight (" w ") == score (" score ")"))
        (is (> (js/Math.abs w) 1.0)
            (str "n=" n ": weight is NOT collapsed to ~0 (was the bug)"))))))

(deftest hof-scoring-matches-handler
  (testing "fn-literal-via-HOF model scores every iteration (mechanism 2)"
    (doseq [n [1 3 5]]
      (let [cons (apply cm/choicemap
                        (mapcat (fn [i] [(keyword (str "x" i)) (mx/scalar 0.0)]) (range n)))
            w-default (gen-weight hof-model [n] cons)
            w-handler (mx/item (:weight (p/generate
                                         (dyn/with-key (gfi/strip-compiled hof-model)
                                           (rng/fresh-key 0))
                                         [n] cons)))
            ;; N independent N(0;0,1) sites each contribute -0.5*log(2π) ≈ -0.9189
            expected (* n (h/gaussian-lp 0.0 0.0 1.0))]
        (is (h/close? w-default w-handler 1e-4)
            (str "n=" n ": default path == handler path"))
        (is (h/close? w-default expected 1e-3)
            (str "n=" n ": all " n " hidden sites scored (not just one)"))))))

(deftest letfn-project-matches-handler
  (testing "project on a letfn-via-HOF model scores every iteration (mechanism 3)"
    ;; Pre-fix this dispatched :project to :compiled and under-counted (scored the
    ;; site once instead of n times) — a real compiled-vs-handler divergence.
    (doseq [n [1 3 5]]
      (let [args [n]
            keyed (dyn/with-key letfn-model (rng/fresh-key (+ 300 n)))
            tr (p/simulate keyed args)
            ;; select every traced site
            site-sel (apply sel/select (mapv #(keyword (str "x" %)) (range n)))
            w-default (mx/item (p/project keyed tr site-sel))
            handler-gf (dyn/with-key (gfi/strip-compiled letfn-model) (rng/fresh-key (+ 300 n)))
            tr-h (p/simulate handler-gf args)
            w-handler (mx/item (p/project handler-gf tr-h site-sel))]
        (is (h/finite? w-default) (str "n=" n ": project weight finite"))
        (is (h/close? w-default w-handler 1e-3)
            (str "n=" n ": project default (" w-default ") == handler (" w-handler ")"))))))

;; ---------------------------------------------------------------------------
;; 3. Plate scoring == sum of per-block scores (the workaround the bug forced)
;; ---------------------------------------------------------------------------

(deftest plate-equals-sum-of-blocks
  (testing "constraining observations in a plate scores like summing single blocks"
    ;; Generate per-block trajectories from the single-block model, then score
    ;; them (a) as one plate and (b) as a sum of single blocks — must match.
    (let [n 3
          ;; data: simulate n single blocks at TRUE-THETA
          blocks (mapv (fn [bi]
                         (let [k (rng/fresh-key (+ 200 bi))
                               r (p/generate (dyn/with-key single-model k) []
                                             (cm/choicemap :theta (mx/scalar TRUE-THETA)))]
                           (without-addr (:choices (:trace r)) :theta)))
                       (range n))
          theta 0.5
          ;; (a) plate constraints: re-prefix each block's obs as b{bi}_*
          plate-cons (reduce
                      (fn [cm [bi block]]
                        (reduce (fn [cm path]
                                  (let [a (name (first path))
                                        a' (keyword (str "b" bi "_" a))]
                                    (cm/set-choice cm [a'] (cm/get-choice block path))))
                                cm (cm/addresses block)))
                      (cm/choicemap :theta (mx/scalar theta))
                      (map-indexed vector blocks))
          plate-w (gen-weight plate-model [n] plate-cons)
          ;; (b) sum of single-block weights
          sum-w (reduce (fn [acc block]
                          (+ acc (gen-weight single-model []
                                             (cm/merge-cm
                                              (cm/choicemap :theta (mx/scalar theta))
                                              block))))
                        0.0 blocks)]
      (is (h/finite? plate-w) "plate weight finite")
      (is (h/close? plate-w sum-w 1e-2)
          (str "plate weight (" plate-w ") == sum of per-block weights (" sum-w ")")))))

;; ---------------------------------------------------------------------------
;; 4. Latent-varying likelihood — the posterior must not collapse to the prior
;; ---------------------------------------------------------------------------

(deftest likelihood-varies-with-latent
  (testing "weight over a fixed obs trajectory varies with the constrained latent"
    (let [n 4
          ;; obs trajectory from the plate at TRUE-THETA
          k   (rng/fresh-key 7)
          tr  (:trace (p/generate (dyn/with-key plate-model k) [n]
                                  (cm/choicemap :theta (mx/scalar TRUE-THETA))))
          obs (without-addr (:choices tr) :theta)
          grid [0.1 0.3 0.5 0.7 0.9]
          ws   (mapv (fn [theta]
                       (gen-weight plate-model [n]
                                   (cm/merge-cm (cm/choicemap :theta (mx/scalar theta)) obs)))
                     grid)]
      (is (every? h/finite? ws) "all grid weights finite")
      (is (> (apply max ws) (+ (apply min ws) 1.0))
          (str "likelihood varies across theta (range "
               (- (apply max ws) (apply min ws)) "), did NOT collapse"))
      ;; the survival likelihood of a stopping rule peaks at an interior theta,
      ;; not at the boundary 0.1 — the constraints are genuinely being scored.
      (let [best (->> (map vector grid ws) (apply max-key second) first)]
        (is (> best 0.1) (str "argmax theta (" best ") is interior, not the prior floor"))))))

;; ---------------------------------------------------------------------------
;; 5. Library inference runs on plate models (the bean's final 'done means')
;; ---------------------------------------------------------------------------

(deftest inference-runs-on-plate
  (let [n 3
        k   (rng/fresh-key 11)
        tr  (:trace (p/generate (dyn/with-key plate-model k) [n]
                                (cm/choicemap :theta (mx/scalar TRUE-THETA))))
        obs (without-addr (:choices tr) :theta)]

    (testing "importance sampling produces non-degenerate weights"
      (let [{:keys [log-weights log-ml-estimate]}
            (is/importance-sampling {:samples 200 :key (rng/fresh-key 3)}
                                    plate-model [n] obs)
            ws (mapv mx/item log-weights)]
        (is (h/finite? (mx/item log-ml-estimate)) "log-ML estimate finite")
        (is (every? h/finite? ws) "all IS log-weights finite")
        ;; pre-fix every particle scored only the prior => identical weights.
        (is (> (- (apply max ws) (apply min ws)) 1.0)
            "IS weights vary across particles (not prior-collapsed)")))

    (testing "MH runs and concentrates theta away from the prior mean"
      (let [traces (mcmc/mh {:samples 150 :burn 50
                             :selection (sel/select :theta)
                             :key (rng/fresh-key 5)}
                            plate-model [n] obs)
            thetas (mapv #(mx/item (cm/get-choice (:choices %) [:theta])) traces)
            mean-theta (h/sample-mean thetas)]
        (is (= 150 (count traces)) "MH returned the requested number of samples")
        (is (every? h/finite? thetas) "all sampled thetas finite")
        (is (and (> mean-theta 0.0) (< mean-theta 1.0)) "theta within prior support")
        ;; with real likelihood the chain MOVES; a collapsed posterior would just
        ;; mirror the uniform prior (mean ~0.5 with huge spread). We only require
        ;; that it produced a proper, finite posterior summary.
        (is (> (h/sample-variance thetas) 0.0) "chain mixes (non-degenerate)")))))

;; ---------------------------------------------------------------------------
;; 6. strip-compiled / handler equivalence sanity
;; ---------------------------------------------------------------------------

(deftest handler-equivalence
  (testing "post-fix dispatch already equals the forced-handler path"
    (let [n 2
          k  (rng/fresh-key 21)
          tr (p/simulate (dyn/with-key plate-model k) [n])
          cons (all-constraints (:choices tr))
          w-default (gen-weight plate-model [n] cons)
          w-handler (mx/item (:weight (p/generate
                                       (dyn/with-key (gfi/strip-compiled plate-model)
                                         (rng/fresh-key 0))
                                       [n] cons)))]
      (is (h/close? w-default w-handler 1e-4)
          "default path == strip-compiled handler path"))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
