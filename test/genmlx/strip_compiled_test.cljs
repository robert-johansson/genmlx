;; @tier fast core
(ns genmlx.strip-compiled-test
  "genmlx-pkmx: strip-compiled must remove ALL alternate-path schema keys
   (full-compile + prefix + analytical) and preserve gen-fn metadata
   (genmlx-3lgy).

   genmlx-ctpw: a :marginal (Rao-Blackwellized) trace's score IS the marginal
   log-density (latents integrated out of the score; the recorded latent is a
   posterior-mean annotation, not a scored choice). So a value-only observation
   update/project STAYS :marginal and returns the Δ marginal-LL — the exact,
   lower-variance RB weight — NOT the joint handler delta. Demanding per-op
   equality with the handler would forbid all of L3 (analytical *generate*
   already returns a different, exact weight than the handler). Conversion to
   :joint is required ONLY when an op re-opens an eliminated latent by
   constraining it. linear_gaussian_elim_test U2 (re-open → :joint) and U3
   (value update → :marginal, vs closed-form) are the independent reference for
   this same contract; the marginal CHAIN telescopes to the true block
   evidence, which is the coherence property that matters."
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
;; 4. :marginal trace update/project semantics (genmlx-ctpw)
;; ============================================================
;; Model: mu ~ N(0,1), y ~ N(mu,1)  =>  marginal y ~ N(0, prior-var+obs-var=2).
;; Independent oracle: the closed-form marginal log-density (NOT the handler
;; joint delta, which would be circular for the marginal contract — the
;; wrong-oracle trap). Mirrors linear_gaussian_elim_test U3 on the same model.
(deftest test-marginal-trace-conversion
  (let [model (dyn/with-key (conjugate-model) (rng/fresh-key 7))
        obs (cm/choicemap :y 1.5)
        {:keys [trace] gen-w :weight} (p/generate model [] obs)
        stripped (dyn/auto-key (gfi/strip-compiled model))
        {jt :trace} (p/generate stripped [] (:choices trace))
        mvar 2.0
        lz (fn [y] (- (* -0.5 (js/Math.log (* 2 js/Math.PI mvar)))
                      (/ (* y y) (* 2 mvar))))]
    (testing "precondition: analytical generate produced a marginal trace"
      (is (= :marginal (:genmlx.trace/score-type (meta trace)))
          "trace is analytically scored"))
    (testing "value-only obs update on a :marginal trace stays marginal (Δ marginal-LL)"
      (let [u-marg (p/update model trace (cm/choicemap :y 2.0))
            _ (mx/materialize! (:weight u-marg))
            w-marg (mx/item (:weight u-marg))
            w-oracle (- (lz 2.0) (lz 1.5))]            ; = -0.4375
        (is (< (js/Math.abs (- w-marg w-oracle)) 2e-4)
            (str "marginal update weight = Δ marginal-LL: " w-marg
                 " vs closed-form oracle " w-oracle))
        (is (= :marginal (:genmlx.trace/score-type (meta (:trace u-marg))))
            "result STAYS :marginal (Rao-Blackwellized — genmlx-ctpw)")
        (is (< (js/Math.abs (- (mx/item (cm/get-value
                                          (cm/get-submap (:choices (:trace u-marg)) :mu)))
                               1.0)) 1e-3)
            "recorded latent moved to the new posterior mean E[mu|y=2.0]=1.0")))
    (testing "marginal chain telescopes to the true block evidence (coherence)"
      ;; gen weight + update weight = log p_marg(y=2.0). This is the property the
      ;; superseded marginal-gen + joint-update mixing violated by +0.0625 nats —
      ;; the test that would have caught the contradiction.
      (let [u-marg (p/update model trace (cm/choicemap :y 2.0))
            _ (mx/materialize! gen-w (:weight u-marg))
            cumulative (+ (mx/item gen-w) (mx/item (:weight u-marg)))]
        (is (< (js/Math.abs (- cumulative (lz 2.0))) 1e-4)
            (str "cumulative marginal weight = log p_marg(y=2.0): "
                 cumulative " vs " (lz 2.0)))))
    (testing "a genuine JOINT (handler) trace updates jointly and stays :joint"
      (let [u-joint (p/update stripped jt (cm/choicemap :y 2.0))
            _ (mx/materialize! (:weight u-joint))
            w-joint (mx/item (:weight u-joint))]
        ;; joint delta holds mu fixed at the recorded 0.75:
        ;; logN(2|.75,1) - logN(1.5|.75,1) = -0.5
        (is (< (js/Math.abs (- w-joint -0.5)) 1e-3)
            (str "joint update weight = Δ joint-LL at fixed mu: " w-joint))
        (is (= :joint (:genmlx.trace/score-type (meta (:trace u-joint))))
            "handler trace stays :joint")))
    (testing "project on a :marginal trace is finite (marginal density contribution)"
      (let [p-marg (p/project model trace (sel/select :mu))
            p-joint (p/project stripped jt (sel/select :mu))
            _ (mx/materialize! p-marg p-joint)]
        ;; project on a :marginal trace returns its own (marginal) contribution;
        ;; it need not equal the joint-rescored project. Assert both finite.
        (is (js/isFinite (mx/item p-marg)) "marginal project finite")
        (is (js/isFinite (mx/item p-joint)) "joint project finite")))
    (testing "regenerate keeps its analytical path (no conversion needed)"
      (let [{t' :trace} (p/regenerate model trace (sel/select :mu))]
        (is (some? t') "analytical regenerate works on marginal traces")))
    (testing "joint traces are passed through untouched"
      (let [upd (p/update stripped jt (cm/choicemap :y 2.0))]
        (is (= :joint (:genmlx.trace/score-type (meta jt)))
            "handler trace is joint (explicitly tagged — genmlx-lbae)")
        (is (some? (:trace upd)) "update on joint trace works")))))

(cljs.test/run-tests)
