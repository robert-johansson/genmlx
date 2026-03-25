(ns genmlx.branch-rewrite-test
  "L1-M4 tests: automatic branch rewriting for compiled gen functions."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.gen :refer [gen]]
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
;; Non-assert helpers (kept)
;; ---------------------------------------------------------------------------

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

(def branch-simple
  (gen [flag]
    (if flag
      (trace :x (dist/gaussian 0 1))
      (trace :x (dist/gaussian 0 2)))))

(def branch-ifnot
  (gen [flag]
    (if-not flag
      (trace :x (dist/gaussian 0 1))
      (trace :x (dist/gaussian 0 2)))))

(def branch-with-prior
  (gen [flag]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (if flag
        (trace :x (dist/gaussian mu 1))
        (trace :x (dist/gaussian mu 2))))))

(def branch-let-binding
  (gen [flag]
    (let [x (if flag
              (trace :x (dist/gaussian 0 1))
              (trace :x (dist/gaussian 0 2)))]
      (mx/add x (mx/scalar 1)))))

(def branch-dep-chain
  (gen [flag]
    (let [x (if flag
              (trace :x (dist/gaussian 0 1))
              (trace :x (dist/gaussian 0 2)))]
      (trace :y (dist/gaussian x 1)))))

(def branch-multi-indep
  (gen [f1 f2]
    (let [x (if f1
              (trace :x (dist/gaussian 0 1))
              (trace :x (dist/gaussian 0 2)))
          y (if f2
              (trace :y (dist/gaussian 0 3))
              (trace :y (dist/gaussian 0 4)))]
      (mx/add x y))))

(def branch-uniform
  (gen [flag]
    (if flag
      (trace :x (dist/uniform 0 1))
      (trace :x (dist/uniform -1 1)))))

(def branch-exponential
  (gen [flag]
    (if flag
      (trace :x (dist/exponential 1))
      (trace :x (dist/exponential 2)))))

(def branch-bernoulli
  (gen [flag]
    (if flag
      (trace :x (dist/bernoulli 0.3))
      (trace :x (dist/bernoulli 0.7)))))

(def branch-laplace
  (gen [flag]
    (if flag
      (trace :x (dist/laplace 0 1))
      (trace :x (dist/laplace 0 2)))))

(def branch-cauchy
  (gen [flag]
    (if flag
      (trace :x (dist/cauchy 0 1))
      (trace :x (dist/cauchy 0 2)))))

(def branch-log-normal
  (gen [flag]
    (if flag
      (trace :x (dist/log-normal 0 1))
      (trace :x (dist/log-normal 0 2)))))

(def branch-delta
  (gen [flag]
    (if flag
      (trace :x (dist/delta 1))
      (trace :x (dist/delta 2)))))

(def branch-mlx-cond
  (gen [x]
    (if (mx/less x (mx/scalar 0))
      (trace :y (dist/gaussian 0 1))
      (trace :y (dist/gaussian 0 2)))))

(def branch-traced-cond
  (gen []
    (let [b (trace :b (dist/bernoulli 0.5))]
      (if (mx/less-equal b (mx/scalar 0))
        (trace :x (dist/gaussian 0 1))
        (trace :x (dist/gaussian 0 2))))))

(def branch-identical-args
  (gen [flag]
    (if flag
      (trace :x (dist/gaussian 0 1))
      (trace :x (dist/gaussian 0 1)))))

;; ---------------------------------------------------------------------------
;; Fallback models (should NOT get M4)
;; ---------------------------------------------------------------------------

(def branch-diff-addr
  (gen [flag]
    (if flag
      (trace :x (dist/gaussian 0 1))
      (trace :y (dist/gaussian 0 2)))))

(def branch-diff-dist
  (gen [flag]
    (if flag
      (trace :x (dist/gaussian 0 1))
      (trace :x (dist/uniform 0 1)))))

(def branch-when
  (gen [flag]
    (when flag
      (trace :x (dist/gaussian 0 1)))))

(def branch-multi-trace
  (gen [flag]
    (if flag
      (do (trace :x (dist/gaussian 0 1))
          (trace :y (dist/gaussian 0 1)))
      (do (trace :x (dist/gaussian 0 2))
          (trace :y (dist/gaussian 0 2))))))

(def branch-nested
  (gen [f1 f2]
    (if f1
      (if f2
        (trace :x (dist/gaussian 0 1))
        (trace :x (dist/gaussian 0 2)))
      (trace :x (dist/gaussian 0 3)))))

(def branch-with-loop
  (gen [flag xs]
    (let [mu (if flag
               (trace :mu (dist/gaussian 0 1))
               (trace :mu (dist/gaussian 0 2)))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace (keyword (str "y" j)) (dist/gaussian mu 1)))
      mu)))

(def static-model
  (gen [x]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y (dist/gaussian mu 1)))))

(def branch-unsupported-dist
  (gen [flag]
    (if flag
      (trace :x (dist/beta-dist 2 3))
      (trace :x (dist/beta-dist 3 2)))))

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

(deftest gate-dispatch-m4-eligible-test
  (testing "M4-eligible models get :compiled-simulate"
    (is (has-compiled-simulate? branch-simple)
        "branch-simple gets M4 :compiled-simulate")
    (is (has-compiled-simulate? branch-ifnot)
        "branch-ifnot gets M4 :compiled-simulate")
    (is (has-compiled-simulate? branch-with-prior)
        "branch-with-prior gets M4 :compiled-simulate")
    (is (has-compiled-simulate? branch-let-binding)
        "branch-let-binding gets M4 :compiled-simulate")
    (is (has-compiled-simulate? branch-dep-chain)
        "branch-dep-chain gets M4 :compiled-simulate")
    (is (has-compiled-simulate? branch-multi-indep)
        "branch-multi-indep gets M4 :compiled-simulate")
    (is (has-compiled-simulate? branch-uniform)
        "branch-uniform gets M4 :compiled-simulate")
    (is (has-compiled-simulate? branch-exponential)
        "branch-exponential gets M4 :compiled-simulate")
    (is (has-compiled-simulate? branch-bernoulli)
        "branch-bernoulli gets M4 :compiled-simulate")
    (is (has-compiled-simulate? branch-laplace)
        "branch-laplace gets M4 :compiled-simulate")
    (is (has-compiled-simulate? branch-cauchy)
        "branch-cauchy gets M4 :compiled-simulate")
    (is (has-compiled-simulate? branch-log-normal)
        "branch-log-normal gets M4 :compiled-simulate")
    (is (has-compiled-simulate? branch-delta)
        "branch-delta gets M4 :compiled-simulate")
    (is (has-compiled-simulate? branch-identical-args)
        "branch-identical-args gets M4 :compiled-simulate")))

(deftest gate-dispatch-fallback-test
  (testing "Fallback models should NOT get M4 :compiled-simulate"
    (is (not (has-compiled-simulate? branch-diff-addr))
        "branch-diff-addr: no :compiled-simulate")
    (is (not (has-compiled-simulate? branch-diff-dist))
        "branch-diff-dist: no :compiled-simulate")
    (is (not (has-compiled-simulate? branch-when))
        "branch-when: no :compiled-simulate")
    (is (not (has-compiled-simulate? branch-multi-trace))
        "branch-multi-trace: no :compiled-simulate")
    (is (not (has-compiled-simulate? branch-nested))
        "branch-nested: no :compiled-simulate")
    (is (not (has-compiled-simulate? branch-with-loop))
        "branch-with-loop: no :compiled-simulate (has loop)")
    (is (and (has-compiled-simulate? static-model)
             (:static? (:schema static-model)))
        "static-model: gets :compiled-simulate via M2 (not M4)")
    (is (not (has-compiled-simulate? branch-unsupported-dist))
        "branch-unsupported-dist: no :compiled-simulate")
    (is (not (has-compiled-simulate? branch-cond))
        "branch-cond: no :compiled-simulate")
    (is (not (has-compiled-simulate? branch-mlx-cond))
        "branch-mlx-cond: no M4 (MLX array condition)")
    (is (not (has-compiled-simulate? branch-traced-cond))
        "branch-traced-cond: no M4 (traced value condition)")))

;; ---------------------------------------------------------------------------
;; 2. Value Equivalence Tests
;; ---------------------------------------------------------------------------

(deftest value-equivalence-simple-test
  (testing "branch-simple: compiled vs handler"
    (let [key (rng/fresh-key 42)
          gf (dyn/with-key branch-simple key)
          gf-h (dyn/with-key (force-handler branch-simple) key)]
      (let [t1 (p/simulate gf [true])
            t2 (p/simulate gf-h [true])]
        (is (h/close? (cv t2 :x) (cv t1 :x) 1e-5)
            "simple flag=true :x value match"))
      (let [key2 (rng/fresh-key 43)
            gf2 (dyn/with-key branch-simple key2)
            gf2-h (dyn/with-key (force-handler branch-simple) key2)
            t1 (p/simulate gf2 [false])
            t2 (p/simulate gf2-h [false])]
        (is (h/close? (cv t2 :x) (cv t1 :x) 1e-5)
            "simple flag=false :x value match")))))

(deftest value-equivalence-ifnot-test
  (testing "branch-ifnot: compiled vs handler"
    (let [key (rng/fresh-key 44)
          gf (dyn/with-key branch-ifnot key)
          gf-h (dyn/with-key (force-handler branch-ifnot) key)]
      (let [t1 (p/simulate gf [true])
            t2 (p/simulate gf-h [true])]
        (is (h/close? (cv t2 :x) (cv t1 :x) 1e-5)
            "ifnot flag=true :x match"))
      (let [key2 (rng/fresh-key 45)
            gf2 (dyn/with-key branch-ifnot key2)
            gf2-h (dyn/with-key (force-handler branch-ifnot) key2)
            t1 (p/simulate gf2 [false])
            t2 (p/simulate gf2-h [false])]
        (is (h/close? (cv t2 :x) (cv t1 :x) 1e-5)
            "ifnot flag=false :x match")))))

(deftest value-equivalence-prior-test
  (testing "branch-with-prior: compiled vs handler"
    (let [key (rng/fresh-key 50)
          gf (dyn/with-key branch-with-prior key)
          gf-h (dyn/with-key (force-handler branch-with-prior) key)]
      (let [t1 (p/simulate gf [true])
            t2 (p/simulate gf-h [true])]
        (is (h/close? (cv t2 :mu) (cv t1 :mu) 1e-5)
            "prior flag=true :mu match")
        (is (h/close? (cv t2 :x) (cv t1 :x) 1e-5)
            "prior flag=true :x match"))
      (let [key2 (rng/fresh-key 51)
            gf2 (dyn/with-key branch-with-prior key2)
            gf2-h (dyn/with-key (force-handler branch-with-prior) key2)
            t1 (p/simulate gf2 [false])
            t2 (p/simulate gf2-h [false])]
        (is (h/close? (cv t2 :mu) (cv t1 :mu) 1e-5)
            "prior flag=false :mu match")
        (is (h/close? (cv t2 :x) (cv t1 :x) 1e-5)
            "prior flag=false :x match")))))

(deftest value-equivalence-let-binding-test
  (testing "branch-let-binding: return value match"
    (let [key (rng/fresh-key 52)
          gf (dyn/with-key branch-let-binding key)
          gf-h (dyn/with-key (force-handler branch-let-binding) key)]
      (let [t1 (p/simulate gf [true])
            t2 (p/simulate gf-h [true])]
        (is (h/close? (mx/item (:retval t2)) (mx/item (:retval t1)) 1e-5)
            "let-binding flag=true retval match"))
      (let [key2 (rng/fresh-key 53)
            gf2 (dyn/with-key branch-let-binding key2)
            gf2-h (dyn/with-key (force-handler branch-let-binding) key2)
            t1 (p/simulate gf2 [false])
            t2 (p/simulate gf2-h [false])]
        (is (h/close? (mx/item (:retval t2)) (mx/item (:retval t1)) 1e-5)
            "let-binding flag=false retval match")))))

(deftest value-equivalence-dep-chain-test
  (testing "branch-dep-chain: downstream trace depends on branch"
    (let [key (rng/fresh-key 54)
          gf (dyn/with-key branch-dep-chain key)
          gf-h (dyn/with-key (force-handler branch-dep-chain) key)]
      (let [t1 (p/simulate gf [true])
            t2 (p/simulate gf-h [true])]
        (is (h/close? (cv t2 :x) (cv t1 :x) 1e-5)
            "dep-chain flag=true :x match")
        (is (h/close? (cv t2 :y) (cv t1 :y) 1e-5)
            "dep-chain flag=true :y match"))
      (let [key2 (rng/fresh-key 55)
            gf2 (dyn/with-key branch-dep-chain key2)
            gf2-h (dyn/with-key (force-handler branch-dep-chain) key2)
            t1 (p/simulate gf2 [false])
            t2 (p/simulate gf2-h [false])]
        (is (h/close? (cv t2 :x) (cv t1 :x) 1e-5)
            "dep-chain flag=false :x match")
        (is (h/close? (cv t2 :y) (cv t1 :y) 1e-5)
            "dep-chain flag=false :y match")))))

(deftest value-equivalence-multi-indep-test
  (testing "branch-multi-indep: two independent branches"
    (doseq [[f1 f2 label] [[true true "tt"] [true false "tf"]
                            [false true "ft"] [false false "ff"]]]
      (let [key (rng/fresh-key (+ 60 (hash [f1 f2])))
            gf (dyn/with-key branch-multi-indep key)
            gf-h (dyn/with-key (force-handler branch-multi-indep) key)
            t1 (p/simulate gf [f1 f2])
            t2 (p/simulate gf-h [f1 f2])]
        (is (h/close? (cv t2 :x) (cv t1 :x) 1e-5)
            (str "multi " label " :x match"))
        (is (h/close? (cv t2 :y) (cv t1 :y) 1e-5)
            (str "multi " label " :y match"))))))

;; ---------------------------------------------------------------------------
;; 3. Score Equivalence Tests
;; ---------------------------------------------------------------------------

(deftest score-equivalence-test
  (testing "Scores match handler for all M4 models"
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
          (is (h/close? (mx/item (:score t2)) (mx/item (:score t1)) 1e-5)
              (str model-name " " args " score match")))))))

;; ---------------------------------------------------------------------------
;; 4. PRNG Consistency Tests
;; ---------------------------------------------------------------------------

(deftest prng-consistency-test
  (testing "Same key produces same values"
    (let [key (rng/fresh-key 100)]
      (let [gf1 (dyn/with-key branch-simple key)
            gf2 (dyn/with-key branch-simple key)
            t1 (p/simulate gf1 [true])
            t2 (p/simulate gf2 [true])]
        (is (h/close? (cv t1 :x) (cv t2 :x) 0)
            "same key -> same :x (flag=true)"))
      (let [gf1 (dyn/with-key branch-simple key)
            gf2 (dyn/with-key branch-simple key)
            t1 (p/simulate gf1 [false])
            t2 (p/simulate gf2 [false])]
        (is (h/close? (cv t1 :x) (cv t2 :x) 0)
            "same key -> same :x (flag=false)"))))

  (testing "Different key produces different values"
    (let [gf1 (dyn/with-key branch-simple (rng/fresh-key 200))
          gf2 (dyn/with-key branch-simple (rng/fresh-key 201))
          t1 (p/simulate gf1 [true])
          t2 (p/simulate gf2 [true])]
      (is (not= (cv t1 :x) (cv t2 :x))
          "diff key -> diff :x")))

  (testing "Cross-path PRNG equivalence (compiled = handler, same key)"
    (doseq [[model-name model args]
            [["branch-with-prior" branch-with-prior [true]]
             ["branch-dep-chain" branch-dep-chain [false]]]]
      (let [key (rng/fresh-key 300)
            gf (dyn/with-key model key)
            gf-h (dyn/with-key (force-handler model) key)
            t1 (p/simulate gf args)
            t2 (p/simulate gf-h args)]
        (doseq [addr (keys (:m (:choices t2)))]
          (is (h/close? (mx/item (cm/get-value (cm/get-submap (:choices t2) addr)))
                        (mx/item (cm/get-value (cm/get-submap (:choices t1) addr)))
                        1e-5)
              (str model-name " cross-path PRNG " addr)))))))

;; ---------------------------------------------------------------------------
;; 5. Distribution Variety Tests
;; ---------------------------------------------------------------------------

(deftest distribution-variety-test
  (testing "Each dist type: compiled vs handler match"
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
        (is (h/close? (cv t2 :x) (cv t1 :x) 1e-5)
            (str model-name " flag=true :x match"))))))

;; ---------------------------------------------------------------------------
;; 6. Condition Type Tests
;; ---------------------------------------------------------------------------

(deftest condition-type-test
  (testing "MLX comparison conditions fall back to handler"
    (is (not (has-compiled-simulate? branch-mlx-cond))
        "branch-mlx-cond falls back (no M4)")
    (is (not (has-compiled-simulate? branch-traced-cond))
        "branch-traced-cond falls back (no M4)")
    (let [gf (dyn/auto-key branch-mlx-cond)
          t (p/simulate gf [(mx/scalar -1)])]
      (is (some? (:choices t))
          "mlx-cond model works via handler"))
    (let [gf (dyn/auto-key branch-traced-cond)
          t (p/simulate gf [])]
      (is (some? (:choices t))
          "traced-cond model works via handler"))))

;; ---------------------------------------------------------------------------
;; 7. GFI Operations Tests
;; ---------------------------------------------------------------------------

(deftest gfi-generate-test
  (testing "generate with constraints on M4 model"
    (let [key (rng/fresh-key 600)
          gf (dyn/with-key branch-with-prior key)
          obs (cm/set-value cm/EMPTY :x (mx/scalar 5.0))
          {:keys [trace weight]} (p/generate gf [true] obs)]
      (is (h/close? 5.0 (cv trace :x) 1e-6)
          "generate constrains :x")
      (is (number? (mx/item weight))
          "generate returns weight"))))

(deftest gfi-update-test
  (testing "update on M4 model"
    (let [key (rng/fresh-key 601)
          gf (dyn/with-key branch-with-prior key)
          trace (p/simulate gf [true])
          new-obs (cm/set-value cm/EMPTY :x (mx/scalar 3.0))
          {:keys [trace weight]} (p/update gf trace new-obs)]
      (is (h/close? 3.0 (cv trace :x) 1e-6)
          "update constrains :x")
      (is (number? (mx/item weight))
          "update returns weight"))))

(deftest gfi-regenerate-test
  (testing "regenerate on M4 model"
    (let [key1 (rng/fresh-key 602)
          key2 (rng/fresh-key 603)
          gf1 (dyn/with-key branch-with-prior key1)
          trace (p/simulate gf1 [true])
          old-mu (cv trace :mu)
          gf2 (dyn/with-key branch-with-prior key2)
          sel-mu (sel/select :mu)
          {:keys [trace weight]} (p/regenerate gf2 trace sel-mu)]
      (is (not= old-mu (cv trace :mu))
          "regenerate changes :mu")
      (is (number? (mx/item weight))
          "regenerate returns weight"))))

(deftest gfi-assess-test
  (testing "assess on M4 model"
    (let [key (rng/fresh-key 603)
          gf (dyn/with-key branch-simple key)
          choices (cm/set-value cm/EMPTY :x (mx/scalar 0.5))
          {:keys [weight]} (p/assess gf [true] choices)]
      (is (number? (mx/item weight))
          "assess returns weight"))))

(deftest gfi-propose-test
  (testing "propose on M4 model"
    (let [key (rng/fresh-key 604)
          gf (dyn/with-key branch-simple key)
          {:keys [choices weight retval]} (p/propose gf [true])]
      (is (some? choices)
          "propose returns choices")
      (is (number? (mx/item weight))
          "propose returns weight"))))

;; ---------------------------------------------------------------------------
;; 8. Edge Cases
;; ---------------------------------------------------------------------------

(deftest edge-cases-identical-args-test
  (testing "Identical args in both branches (trivial rewrite)"
    (let [key (rng/fresh-key 700)
          gf (dyn/with-key branch-identical-args key)
          gf-h (dyn/with-key (force-handler branch-identical-args) key)
          t1 (p/simulate gf [true])
          t2 (p/simulate gf-h [true])
          t3 (p/simulate (dyn/with-key branch-identical-args key) [false])
          t4 (p/simulate (dyn/with-key (force-handler branch-identical-args) key) [false])]
      (is (h/close? (cv t2 :x) (cv t1 :x) 1e-5)
          "identical-args flag=true match")
      (is (h/close? (cv t4 :x) (cv t3 :x) 1e-5)
          "identical-args flag=false match"))))

(deftest edge-cases-schema-flags-test
  (testing "Schema flags are correct for branch models"
    (is (:has-branches? (:schema branch-simple))
        "branch-simple has-branches?")
    (is (not (:static? (:schema branch-simple)))
        "branch-simple NOT static?")
    (is (empty? (:splice-sites (:schema branch-simple)))
        "branch-simple no splice-sites")
    (is (empty? (:param-sites (:schema branch-simple)))
        "branch-simple no param-sites")))

(deftest edge-cases-fallback-works-test
  (testing "Fallback models still work via handler"
    (let [gf (dyn/auto-key branch-diff-addr)
          t (p/simulate gf [true])]
      (is (some? (:choices t))
          "diff-addr model works"))
    (let [gf (dyn/auto-key branch-when)
          t (p/simulate gf [true])]
      (is (some? (:choices t))
          "when model works"))))

(cljs.test/run-tests)
