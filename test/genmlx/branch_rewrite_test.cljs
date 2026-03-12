(ns genmlx.branch-rewrite-test
  "L1-M4 tests: automatic branch rewriting for compiled gen functions.

   Tests cover:
   1. Gate dispatch (which compilation level each model gets)
   2. Value equivalence (compiled vs handler, same key)
   3. Score equivalence
   4. PRNG consistency (same key → same values, cross-path equivalence)
   5. Distribution variety (branches with different dist types)
   6. Condition types (boolean flag, MLX comparison, traced value)
   7. GFI operations (generate/update/regenerate on M4 models)
   8. Edge cases"
  (:require [genmlx.gen :refer [gen]]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.compiled :as compiled]
            [genmlx.schema :as schema]
            [genmlx.selection :as sel]))

;; ---------------------------------------------------------------------------
;; Test helpers
;; ---------------------------------------------------------------------------

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn- assert-true [desc pred]
  (if pred
    (do (swap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " desc)))))

(defn- assert-close [desc expected actual tol]
  (let [diff (js/Math.abs (- expected actual))]
    (if (<= diff tol)
      (do (swap! pass-count inc)
          (println (str "  PASS: " desc " (diff=" (.toFixed diff 6) ")")))
      (do (swap! fail-count inc)
          (println (str "  FAIL: " desc " expected=" expected " actual=" actual " diff=" diff))))))

(defn- assert-equal [desc expected actual]
  (if (= expected actual)
    (do (swap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " desc " expected=" expected " actual=" actual)))))

(defn- force-handler
  "Return a copy of gf that always uses the handler path (no compiled paths)."
  [gf]
  (dyn/->DynamicGF (:body-fn gf) (:source gf)
                    (dissoc (:schema gf) :compiled-simulate :compiled-prefix
                                         :compiled-prefix-addrs)))

(defn- cv
  "Get choice value at addr as JS number."
  [trace addr]
  (mx/item (cm/get-value (cm/get-submap (:choices trace) addr))))

(defn- has-compiled-simulate? [gf]
  (some? (:compiled-simulate (:schema gf))))

(defn- has-compiled-prefix? [gf]
  (some? (:compiled-prefix (:schema gf))))

;; ---------------------------------------------------------------------------
;; M4-eligible models (should get :compiled-simulate via branch rewriting)
;; ---------------------------------------------------------------------------

;; M4-1: Simplest branch — if with same addr, same dist, different args
(def branch-simple
  (gen [flag]
    (if flag
      (trace :x (dist/gaussian 0 1))
      (trace :x (dist/gaussian 0 2)))))

;; M4-2: if-not variant
(def branch-ifnot
  (gen [flag]
    (if-not flag
      (trace :x (dist/gaussian 0 1))
      (trace :x (dist/gaussian 0 2)))))

;; M4-3: Standard traces before branch
(def branch-with-prior
  (gen [flag]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (if flag
        (trace :x (dist/gaussian mu 1))
        (trace :x (dist/gaussian mu 2))))))

;; M4-4: Branch in let binding, value used later
(def branch-let-binding
  (gen [flag]
    (let [x (if flag
              (trace :x (dist/gaussian 0 1))
              (trace :x (dist/gaussian 0 2)))]
      (mx/add x (mx/scalar 1)))))

;; M4-5: Branch value used as arg to subsequent trace
(def branch-dep-chain
  (gen [flag]
    (let [x (if flag
              (trace :x (dist/gaussian 0 1))
              (trace :x (dist/gaussian 0 2)))]
      (trace :y (dist/gaussian x 1)))))

;; M4-6: Two independent branches
(def branch-multi-indep
  (gen [f1 f2]
    (let [x (if f1
              (trace :x (dist/gaussian 0 1))
              (trace :x (dist/gaussian 0 2)))
          y (if f2
              (trace :y (dist/gaussian 0 3))
              (trace :y (dist/gaussian 0 4)))]
      (mx/add x y))))

;; M4-7: Branch with uniform distribution
(def branch-uniform
  (gen [flag]
    (if flag
      (trace :x (dist/uniform 0 1))
      (trace :x (dist/uniform -1 1)))))

;; M4-8: Branch with exponential distribution
(def branch-exponential
  (gen [flag]
    (if flag
      (trace :x (dist/exponential 1))
      (trace :x (dist/exponential 2)))))

;; M4-9: Branch with bernoulli
(def branch-bernoulli
  (gen [flag]
    (if flag
      (trace :x (dist/bernoulli 0.3))
      (trace :x (dist/bernoulli 0.7)))))

;; M4-10: Branch with laplace
(def branch-laplace
  (gen [flag]
    (if flag
      (trace :x (dist/laplace 0 1))
      (trace :x (dist/laplace 0 2)))))

;; M4-11: Branch with cauchy
(def branch-cauchy
  (gen [flag]
    (if flag
      (trace :x (dist/cauchy 0 1))
      (trace :x (dist/cauchy 0 2)))))

;; M4-12: Branch with log-normal
(def branch-log-normal
  (gen [flag]
    (if flag
      (trace :x (dist/log-normal 0 1))
      (trace :x (dist/log-normal 0 2)))))

;; M4-13: Branch with delta
(def branch-delta
  (gen [flag]
    (if flag
      (trace :x (dist/delta 1))
      (trace :x (dist/delta 2)))))

;; M4-14: MLX comparison as condition
(def branch-mlx-cond
  (gen [x]
    (if (mx/less x (mx/scalar 0))
      (trace :y (dist/gaussian 0 1))
      (trace :y (dist/gaussian 0 2)))))

;; M4-15: Condition references a traced value
(def branch-traced-cond
  (gen []
    (let [b (trace :b (dist/bernoulli 0.5))]
      (if (mx/less-equal b (mx/scalar 0))
        (trace :x (dist/gaussian 0 1))
        (trace :x (dist/gaussian 0 2))))))

;; M4-16: Branch with identical args (trivial rewrite)
(def branch-identical-args
  (gen [flag]
    (if flag
      (trace :x (dist/gaussian 0 1))
      (trace :x (dist/gaussian 0 1)))))

;; ---------------------------------------------------------------------------
;; Fallback models (should NOT get M4)
;; ---------------------------------------------------------------------------

;; FB-1: Different addresses → NOT M4
(def branch-diff-addr
  (gen [flag]
    (if flag
      (trace :x (dist/gaussian 0 1))
      (trace :y (dist/gaussian 0 2)))))

;; FB-2: Different dist types → NOT M4
(def branch-diff-dist
  (gen [flag]
    (if flag
      (trace :x (dist/gaussian 0 1))
      (trace :x (dist/uniform 0 1)))))

;; FB-3: when (one-sided) → NOT M4
(def branch-when
  (gen [flag]
    (when flag
      (trace :x (dist/gaussian 0 1)))))

;; FB-4: Multiple traces per branch → NOT M4
(def branch-multi-trace
  (gen [flag]
    (if flag
      (do (trace :x (dist/gaussian 0 1))
          (trace :y (dist/gaussian 0 1)))
      (do (trace :x (dist/gaussian 0 2))
          (trace :y (dist/gaussian 0 2))))))

;; FB-5: Nested if → NOT M4
(def branch-nested
  (gen [f1 f2]
    (if f1
      (if f2
        (trace :x (dist/gaussian 0 1))
        (trace :x (dist/gaussian 0 2)))
      (trace :x (dist/gaussian 0 3)))))

;; FB-6: Branch + loop → NOT M4 (has-loops)
(def branch-with-loop
  (gen [flag xs]
    (let [mu (if flag
               (trace :mu (dist/gaussian 0 1))
               (trace :mu (dist/gaussian 0 2)))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j)) (dist/gaussian mu 1)))
      mu)))

;; FB-7: Static model → M2 not M4
(def static-model
  (gen [x]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y (dist/gaussian mu 1)))))

;; FB-8: Branch with unsupported dist (beta-dist)
(def branch-unsupported-dist
  (gen [flag]
    (if flag
      (trace :x (dist/beta-dist 2 3))
      (trace :x (dist/beta-dist 3 2)))))

;; FB-9: cond expression → NOT M4
(def branch-cond
  (gen [x]
    (cond
      (mx/less x (mx/scalar 0))    (trace :y (dist/gaussian -1 1))
      (mx/less x (mx/scalar 1))    (trace :y (dist/gaussian 0 1))
      :else                         (trace :y (dist/gaussian 1 1)))))

;; ===========================================================================
;; TESTS
;; ===========================================================================

;; ---------------------------------------------------------------------------
;; 1. Gate Dispatch Tests
;; ---------------------------------------------------------------------------

(println "\n== L1-M4 Gate Dispatch Tests ==")

(println "\n-- M4-eligible models should get :compiled-simulate --")
(assert-true "branch-simple gets M4 :compiled-simulate"
  (has-compiled-simulate? branch-simple))
(assert-true "branch-ifnot gets M4 :compiled-simulate"
  (has-compiled-simulate? branch-ifnot))
(assert-true "branch-with-prior gets M4 :compiled-simulate"
  (has-compiled-simulate? branch-with-prior))
(assert-true "branch-let-binding gets M4 :compiled-simulate"
  (has-compiled-simulate? branch-let-binding))
(assert-true "branch-dep-chain gets M4 :compiled-simulate"
  (has-compiled-simulate? branch-dep-chain))
(assert-true "branch-multi-indep gets M4 :compiled-simulate"
  (has-compiled-simulate? branch-multi-indep))
(assert-true "branch-uniform gets M4 :compiled-simulate"
  (has-compiled-simulate? branch-uniform))
(assert-true "branch-exponential gets M4 :compiled-simulate"
  (has-compiled-simulate? branch-exponential))
(assert-true "branch-bernoulli gets M4 :compiled-simulate"
  (has-compiled-simulate? branch-bernoulli))
(assert-true "branch-laplace gets M4 :compiled-simulate"
  (has-compiled-simulate? branch-laplace))
(assert-true "branch-cauchy gets M4 :compiled-simulate"
  (has-compiled-simulate? branch-cauchy))
(assert-true "branch-log-normal gets M4 :compiled-simulate"
  (has-compiled-simulate? branch-log-normal))
(assert-true "branch-delta gets M4 :compiled-simulate"
  (has-compiled-simulate? branch-delta))
;; branch-mlx-cond and branch-traced-cond have MLX array conditions
;; (always truthy in CLJS if), so M4 correctly rejects them.
(assert-true "branch-identical-args gets M4 :compiled-simulate"
  (has-compiled-simulate? branch-identical-args))

(println "\n-- Fallback models should NOT get M4 :compiled-simulate --")
(assert-true "branch-diff-addr: no :compiled-simulate"
  (not (has-compiled-simulate? branch-diff-addr)))
(assert-true "branch-diff-dist: no :compiled-simulate"
  (not (has-compiled-simulate? branch-diff-dist)))
(assert-true "branch-when: no :compiled-simulate"
  (not (has-compiled-simulate? branch-when)))
(assert-true "branch-multi-trace: no :compiled-simulate"
  (not (has-compiled-simulate? branch-multi-trace)))
(assert-true "branch-nested: no :compiled-simulate"
  (not (has-compiled-simulate? branch-nested)))
(assert-true "branch-with-loop: no :compiled-simulate (has loop)"
  (not (has-compiled-simulate? branch-with-loop)))
(assert-true "static-model: gets :compiled-simulate via M2 (not M4)"
  (and (has-compiled-simulate? static-model)
       (:static? (:schema static-model))))
(assert-true "branch-unsupported-dist: no :compiled-simulate"
  (not (has-compiled-simulate? branch-unsupported-dist)))
(assert-true "branch-cond: no :compiled-simulate"
  (not (has-compiled-simulate? branch-cond)))
(assert-true "branch-mlx-cond: no M4 (MLX array condition)"
  (not (has-compiled-simulate? branch-mlx-cond)))
(assert-true "branch-traced-cond: no M4 (traced value condition)"
  (not (has-compiled-simulate? branch-traced-cond)))

;; ---------------------------------------------------------------------------
;; 2. Value Equivalence Tests
;; ---------------------------------------------------------------------------

(println "\n== Value Equivalence Tests ==")

(println "\n-- branch-simple: compiled vs handler --")
(let [key (rng/fresh-key 42)
      gf (dyn/with-key branch-simple key)
      gf-h (dyn/with-key (force-handler branch-simple) key)]
  (let [t1 (p/simulate gf [true])
        t2 (p/simulate gf-h [true])]
    (assert-close "simple flag=true :x value match" (cv t2 :x) (cv t1 :x) 1e-5))
  (let [key2 (rng/fresh-key 43)
        gf2 (dyn/with-key branch-simple key2)
        gf2-h (dyn/with-key (force-handler branch-simple) key2)
        t1 (p/simulate gf2 [false])
        t2 (p/simulate gf2-h [false])]
    (assert-close "simple flag=false :x value match" (cv t2 :x) (cv t1 :x) 1e-5)))

(println "\n-- branch-ifnot: compiled vs handler --")
(let [key (rng/fresh-key 44)
      gf (dyn/with-key branch-ifnot key)
      gf-h (dyn/with-key (force-handler branch-ifnot) key)]
  (let [t1 (p/simulate gf [true])
        t2 (p/simulate gf-h [true])]
    (assert-close "ifnot flag=true :x match" (cv t2 :x) (cv t1 :x) 1e-5))
  (let [key2 (rng/fresh-key 45)
        gf2 (dyn/with-key branch-ifnot key2)
        gf2-h (dyn/with-key (force-handler branch-ifnot) key2)
        t1 (p/simulate gf2 [false])
        t2 (p/simulate gf2-h [false])]
    (assert-close "ifnot flag=false :x match" (cv t2 :x) (cv t1 :x) 1e-5)))

(println "\n-- branch-with-prior: compiled vs handler --")
(let [key (rng/fresh-key 50)
      gf (dyn/with-key branch-with-prior key)
      gf-h (dyn/with-key (force-handler branch-with-prior) key)]
  (let [t1 (p/simulate gf [true])
        t2 (p/simulate gf-h [true])]
    (assert-close "prior flag=true :mu match" (cv t2 :mu) (cv t1 :mu) 1e-5)
    (assert-close "prior flag=true :x match" (cv t2 :x) (cv t1 :x) 1e-5))
  (let [key2 (rng/fresh-key 51)
        gf2 (dyn/with-key branch-with-prior key2)
        gf2-h (dyn/with-key (force-handler branch-with-prior) key2)
        t1 (p/simulate gf2 [false])
        t2 (p/simulate gf2-h [false])]
    (assert-close "prior flag=false :mu match" (cv t2 :mu) (cv t1 :mu) 1e-5)
    (assert-close "prior flag=false :x match" (cv t2 :x) (cv t1 :x) 1e-5)))

(println "\n-- branch-let-binding: return value match --")
(let [key (rng/fresh-key 52)
      gf (dyn/with-key branch-let-binding key)
      gf-h (dyn/with-key (force-handler branch-let-binding) key)]
  (let [t1 (p/simulate gf [true])
        t2 (p/simulate gf-h [true])]
    (assert-close "let-binding flag=true retval match"
      (mx/item (:retval t2)) (mx/item (:retval t1)) 1e-5))
  (let [key2 (rng/fresh-key 53)
        gf2 (dyn/with-key branch-let-binding key2)
        gf2-h (dyn/with-key (force-handler branch-let-binding) key2)
        t1 (p/simulate gf2 [false])
        t2 (p/simulate gf2-h [false])]
    (assert-close "let-binding flag=false retval match"
      (mx/item (:retval t2)) (mx/item (:retval t1)) 1e-5)))

(println "\n-- branch-dep-chain: downstream trace depends on branch --")
(let [key (rng/fresh-key 54)
      gf (dyn/with-key branch-dep-chain key)
      gf-h (dyn/with-key (force-handler branch-dep-chain) key)]
  (let [t1 (p/simulate gf [true])
        t2 (p/simulate gf-h [true])]
    (assert-close "dep-chain flag=true :x match" (cv t2 :x) (cv t1 :x) 1e-5)
    (assert-close "dep-chain flag=true :y match" (cv t2 :y) (cv t1 :y) 1e-5))
  (let [key2 (rng/fresh-key 55)
        gf2 (dyn/with-key branch-dep-chain key2)
        gf2-h (dyn/with-key (force-handler branch-dep-chain) key2)
        t1 (p/simulate gf2 [false])
        t2 (p/simulate gf2-h [false])]
    (assert-close "dep-chain flag=false :x match" (cv t2 :x) (cv t1 :x) 1e-5)
    (assert-close "dep-chain flag=false :y match" (cv t2 :y) (cv t1 :y) 1e-5)))

(println "\n-- branch-multi-indep: two independent branches --")
(doseq [[f1 f2 label] [[true true "tt"] [true false "tf"]
                        [false true "ft"] [false false "ff"]]]
  (let [key (rng/fresh-key (+ 60 (hash [f1 f2])))
        gf (dyn/with-key branch-multi-indep key)
        gf-h (dyn/with-key (force-handler branch-multi-indep) key)
        t1 (p/simulate gf [f1 f2])
        t2 (p/simulate gf-h [f1 f2])]
    (assert-close (str "multi " label " :x match") (cv t2 :x) (cv t1 :x) 1e-5)
    (assert-close (str "multi " label " :y match") (cv t2 :y) (cv t1 :y) 1e-5)))

;; ---------------------------------------------------------------------------
;; 3. Score Equivalence Tests
;; ---------------------------------------------------------------------------

(println "\n== Score Equivalence Tests ==")

(println "\n-- Scores match handler for all M4 models --")
(doseq [[model-name model flag-vals]
        [["branch-simple" branch-simple [[true] [false]]]
         ["branch-with-prior" branch-with-prior [[true] [false]]]
         ["branch-dep-chain" branch-dep-chain [[true] [false]]]
         ["branch-multi-indep" branch-multi-indep [[true true] [true false]
                                                    [false true] [false false]]]]]
  (doseq [args flag-vals]
    (let [key (rng/fresh-key (hash [model-name args]))
          gf (dyn/with-key model key)
          gf-h (dyn/with-key (force-handler model) key)
          t1 (p/simulate gf args)
          t2 (p/simulate gf-h args)]
      (assert-close (str model-name " " args " score match")
        (mx/item (:score t2)) (mx/item (:score t1)) 1e-5))))

;; ---------------------------------------------------------------------------
;; 4. PRNG Consistency Tests
;; ---------------------------------------------------------------------------

(println "\n== PRNG Consistency Tests ==")

(println "\n-- Same key → same values --")
(let [key (rng/fresh-key 100)]
  (let [gf1 (dyn/with-key branch-simple key)
        gf2 (dyn/with-key branch-simple key)
        t1 (p/simulate gf1 [true])
        t2 (p/simulate gf2 [true])]
    (assert-close "same key → same :x (flag=true)" (cv t1 :x) (cv t2 :x) 0))
  (let [gf1 (dyn/with-key branch-simple key)
        gf2 (dyn/with-key branch-simple key)
        t1 (p/simulate gf1 [false])
        t2 (p/simulate gf2 [false])]
    (assert-close "same key → same :x (flag=false)" (cv t1 :x) (cv t2 :x) 0)))

(println "\n-- Different key → different values --")
(let [gf1 (dyn/with-key branch-simple (rng/fresh-key 200))
      gf2 (dyn/with-key branch-simple (rng/fresh-key 201))
      t1 (p/simulate gf1 [true])
      t2 (p/simulate gf2 [true])]
  (assert-true "diff key → diff :x"
    (not= (cv t1 :x) (cv t2 :x))))

(println "\n-- Cross-path PRNG equivalence (compiled = handler, same key) --")
(doseq [[model-name model args]
        [["branch-with-prior" branch-with-prior [true]]
         ["branch-dep-chain" branch-dep-chain [false]]]]
  (let [key (rng/fresh-key 300)
        gf (dyn/with-key model key)
        gf-h (dyn/with-key (force-handler model) key)
        t1 (p/simulate gf args)
        t2 (p/simulate gf-h args)]
    (doseq [addr (keys (:m (:choices t2)))]
      (assert-close (str model-name " cross-path PRNG " addr)
        (mx/item (cm/get-value (cm/get-submap (:choices t2) addr)))
        (mx/item (cm/get-value (cm/get-submap (:choices t1) addr)))
        1e-5))))

;; ---------------------------------------------------------------------------
;; 5. Distribution Variety Tests
;; ---------------------------------------------------------------------------

(println "\n== Distribution Variety Tests ==")

(println "\n-- Each dist type: compiled vs handler match --")
(doseq [[model-name model]
        [["gaussian" branch-simple]
         ["uniform" branch-uniform]
         ["exponential" branch-exponential]
         ["bernoulli" branch-bernoulli]
         ["laplace" branch-laplace]
         ["cauchy" branch-cauchy]
         ["log-normal" branch-log-normal]
         ["delta" branch-delta]]]
  (let [key (rng/fresh-key (+ 400 (hash model-name)))
        gf (dyn/with-key model key)
        gf-h (dyn/with-key (force-handler model) key)
        t1 (p/simulate gf [true])
        t2 (p/simulate gf-h [true])]
    (assert-close (str model-name " flag=true :x match")
      (cv t2 :x) (cv t1 :x) 1e-5)))

;; ---------------------------------------------------------------------------
;; 6. Condition Type Tests
;; ---------------------------------------------------------------------------

(println "\n== Condition Type Tests ==")

(println "\n-- MLX comparison conditions fall back to handler (MLX arrays always truthy in CLJS if) --")
(assert-true "branch-mlx-cond falls back (no M4)"
  (not (has-compiled-simulate? branch-mlx-cond)))
(assert-true "branch-traced-cond falls back (no M4)"
  (not (has-compiled-simulate? branch-traced-cond)))
;; Both models still work via handler
(let [gf (dyn/auto-key branch-mlx-cond)
      t (p/simulate gf [(mx/scalar -1)])]
  (assert-true "mlx-cond model works via handler" (some? (:choices t))))
(let [gf (dyn/auto-key branch-traced-cond)
      t (p/simulate gf [])]
  (assert-true "traced-cond model works via handler" (some? (:choices t))))

;; ---------------------------------------------------------------------------
;; 7. GFI Operations Tests
;; ---------------------------------------------------------------------------

(println "\n== GFI Operations Tests ==")

(println "\n-- generate with constraints on M4 model --")
(let [key (rng/fresh-key 600)
      gf (dyn/with-key branch-with-prior key)
      obs (cm/set-value cm/EMPTY :x (mx/scalar 5.0))
      {:keys [trace weight]} (p/generate gf [true] obs)]
  (assert-close "generate constrains :x" 5.0 (cv trace :x) 1e-6)
  (assert-true "generate returns weight" (number? (mx/item weight))))

(println "\n-- update on M4 model --")
(let [key (rng/fresh-key 601)
      gf (dyn/with-key branch-with-prior key)
      trace (p/simulate gf [true])
      new-obs (cm/set-value cm/EMPTY :x (mx/scalar 3.0))
      {:keys [trace weight]} (p/update gf trace new-obs)]
  (assert-close "update constrains :x" 3.0 (cv trace :x) 1e-6)
  (assert-true "update returns weight" (number? (mx/item weight))))

(println "\n-- regenerate on M4 model --")
(let [key1 (rng/fresh-key 602)
      key2 (rng/fresh-key 603)
      gf1 (dyn/with-key branch-with-prior key1)
      trace (p/simulate gf1 [true])
      old-mu (cv trace :mu)
      gf2 (dyn/with-key branch-with-prior key2)
      sel-mu (sel/select :mu)
      {:keys [trace weight]} (p/regenerate gf2 trace sel-mu)]
  (assert-true "regenerate changes :mu" (not= old-mu (cv trace :mu)))
  (assert-true "regenerate returns weight" (number? (mx/item weight))))

(println "\n-- assess on M4 model --")
(let [key (rng/fresh-key 603)
      gf (dyn/with-key branch-simple key)
      choices (cm/set-value cm/EMPTY :x (mx/scalar 0.5))
      {:keys [weight]} (p/assess gf [true] choices)]
  (assert-true "assess returns weight" (number? (mx/item weight))))

(println "\n-- propose on M4 model --")
(let [key (rng/fresh-key 604)
      gf (dyn/with-key branch-simple key)
      {:keys [choices weight retval]} (p/propose gf [true])]
  (assert-true "propose returns choices" (some? choices))
  (assert-true "propose returns weight" (number? (mx/item weight))))

;; ---------------------------------------------------------------------------
;; 8. Edge Cases
;; ---------------------------------------------------------------------------

(println "\n== Edge Cases ==")

(println "\n-- Identical args in both branches (trivial rewrite) --")
(let [key (rng/fresh-key 700)
      gf (dyn/with-key branch-identical-args key)
      gf-h (dyn/with-key (force-handler branch-identical-args) key)
      t1 (p/simulate gf [true])
      t2 (p/simulate gf-h [true])
      t3 (p/simulate (dyn/with-key branch-identical-args key) [false])
      t4 (p/simulate (dyn/with-key (force-handler branch-identical-args) key) [false])]
  (assert-close "identical-args flag=true match" (cv t2 :x) (cv t1 :x) 1e-5)
  (assert-close "identical-args flag=false match" (cv t4 :x) (cv t3 :x) 1e-5))

(println "\n-- Schema flags are correct for branch models --")
(assert-true "branch-simple has-branches?" (:has-branches? (:schema branch-simple)))
(assert-true "branch-simple NOT static?" (not (:static? (:schema branch-simple))))
(assert-true "branch-simple no splice-sites" (empty? (:splice-sites (:schema branch-simple))))
(assert-true "branch-simple no param-sites" (empty? (:param-sites (:schema branch-simple))))

(println "\n-- Fallback models still work via handler --")
(let [gf (dyn/auto-key branch-diff-addr)
      t (p/simulate gf [true])]
  (assert-true "diff-addr model works" (some? (:choices t))))
(let [gf (dyn/auto-key branch-when)
      t (p/simulate gf [true])]
  (assert-true "when model works" (some? (:choices t))))

;; ===========================================================================
;; Summary
;; ===========================================================================

(println "\n================================================")
(println (str "L1-M4 Branch Rewrite Tests: "
              @pass-count " passed, " @fail-count " failed, "
              (+ @pass-count @fail-count) " total"))
(println "================================================")

(when (pos? @fail-count)
  (js/process.exit 1))
