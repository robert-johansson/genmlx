;; @tier fast core
(ns genmlx.strip-compiled-test
  "genmlx-pkmx: strip-compiled must remove ALL alternate-path schema keys
   (full-compile + prefix + analytical) and preserve gen-fn metadata
   (genmlx-3lgy); update/project/regenerate on an analytically-scored
   (:marginal) trace must fail loudly instead of mixing decompositions."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.gfi :as gfi]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ^:private full-compile-keys
  [:compiled-simulate :compiled-generate :compiled-update :compiled-assess
   :compiled-project :compiled-regenerate])

(def ^:private prefix-keys
  [:compiled-prefix :compiled-prefix-generate :compiled-prefix-update
   :compiled-prefix-regenerate :compiled-prefix-assess :compiled-prefix-project])

(def ^:private analytical-keys
  [:auto-handlers :conjugate-pairs :has-conjugate? :analytical-plan
   :auto-regenerate-transition])

(defn- conjugate-model
  "mu ~ N(0,1); y ~ N(mu,1). Normal-normal conjugate — L3 analytical keys."
  []
  (gen []
    (let [mu (trace :mu (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
      (trace :y (dist/gaussian mu (mx/scalar 1)))
      mu)))

(defn- prefix-model
  "Static prefix + dynamic loop suffix — L1-M3 prefix keys."
  []
  (gen [n]
    (let [mu (trace :mu (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
      (doseq [i (range n)]
        (trace (keyword (str "y" i)) (dist/gaussian mu (mx/scalar 1))))
      mu)))

;; ============================================================
;; 1. strip-compiled removes every alternate-path key
;; ============================================================
(deftest test-strip-removes-all-alternate-paths
  (testing "analytical keys stripped"
    (let [model (conjugate-model)
          sch (:schema model)
          stripped (:schema (gfi/strip-compiled model))]
      (is (seq (:auto-handlers sch)) "conjugate model HAS analytical keys pre-strip")
      (is (some sch full-compile-keys) "static model HAS full-compile keys pre-strip")
      (is (not-any? stripped (concat full-compile-keys prefix-keys analytical-keys))
          "no alternate-path key survives strip-compiled")))
  (testing "prefix keys stripped"
    (let [model (prefix-model)
          sch (:schema model)
          stripped (:schema (gfi/strip-compiled model))]
      (is (some sch prefix-keys) "prefix model HAS prefix keys pre-strip")
      (is (not-any? stripped (concat full-compile-keys prefix-keys analytical-keys))
          "no alternate-path key survives strip-compiled"))))

;; ============================================================
;; 2. Stripped model dispatches to the handler path
;; ============================================================
(deftest test-stripped-dispatch-is-handler
  (testing "simulate resolves :handler after strip"
    (let [model (conjugate-model)
          pre (dyn/resolve-dispatch model :simulate)
          post (dyn/resolve-dispatch (gfi/strip-compiled model) :simulate)]
      (is (not= :handler (:label pre)) "compiled model does not resolve handler")
      (is (= :handler (:label post)) "stripped model resolves handler"))))

;; ============================================================
;; 3. Metadata (PRNG key) survives strip-compiled (genmlx-3lgy)
;; ============================================================
(deftest test-strip-preserves-key-metadata
  (testing "a keyed model can still run sampling ops after strip"
    (let [model (dyn/with-key (conjugate-model) (rng/fresh-key 5))
          stripped (gfi/strip-compiled model)
          tr (p/simulate stripped [])]
      (is (some? tr) "simulate runs without 'No PRNG key' error")
      (is (cm/has-value? (cm/get-submap (:choices tr) :mu)) "trace has :mu"))))

;; ============================================================
;; 4. Marginal-trace guard on update/project; regenerate stays analytical
;; ============================================================
(deftest test-marginal-trace-guard
  (let [model (dyn/with-key (conjugate-model) (rng/fresh-key 7))
        obs (cm/choicemap :y 1.5)
        {:keys [trace]} (p/generate model [] obs)]
    (testing "precondition: analytical generate produced a marginal trace"
      (is (= :marginal (:genmlx.dynamic/score-type (meta trace)))
          "trace is analytically scored"))
    (testing "update on a marginal trace throws instead of mixing scores"
      (is (thrown-with-msg? js/Error #"mix score decompositions"
            (p/update model trace (cm/choicemap :y 2.0)))))
    (testing "project on a marginal trace throws instead of mixing scores"
      (is (thrown-with-msg? js/Error #"mix score decompositions"
            (p/project model trace (sel/select :mu)))))
    (testing "regenerate keeps its analytical path (no throw)"
      (let [{t' :trace} (p/regenerate model trace (sel/select :mu))]
        (is (some? t') "analytical regenerate works on marginal traces")))
    (testing "joint traces are unaffected by the guard"
      (let [stripped (dyn/auto-key (gfi/strip-compiled model))
            {jt :trace} (p/generate stripped [] obs)
            upd (p/update stripped jt (cm/choicemap :y 2.0))]
        (is (nil? (:genmlx.dynamic/score-type (meta jt))) "handler trace is joint")
        (is (some? (:trace upd)) "update on joint trace works")))))

(cljs.test/run-tests)
