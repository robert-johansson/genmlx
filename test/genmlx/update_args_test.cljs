;; @tier medium
(ns genmlx.update-args-test
  "genmlx-s8e8: GFI update with new arguments (thesis x' on IUpdate).

   update-with-args re-executes a trace's model under NEW arguments:
   retained choices are re-scored under the new dist params, fresh sites
   cancel (internal proposal = prior), removed sites are charged via the
   old score and land in the discard. Weight invariant:

     w = nonfresh-score(t'; x') - score(t; x)

   Structure-preserving corollary (what SMC over growing data uses):
   no fresh/removed sites => w = assess(x', t') - assess(x, t) exactly.

   Every numeric assertion here is checked against a hand-derived
   closed-form oracle (log-normal-pdf below), never against the path
   under test (independent-oracle rule, the ke9i lesson)."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.gfi :as gfi]
            [genmlx.trace :as tr]
            [genmlx.dist :as dist]
            [genmlx.diff :as diff]
            [genmlx.edit :as edit]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.choicemap :as cm]
            [genmlx.combinators :as comb]
            [genmlx.inference.smc :as smc])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Oracles and helpers
;; ---------------------------------------------------------------------------

(defn- log-normal-pdf
  "Hand-derived gaussian log-density — the independent oracle."
  [x mu sigma]
  (- (* -0.5 (js/Math.log (* 2 js/Math.PI)))
     (js/Math.log sigma)
     (* 0.5 (js/Math.pow (/ (- x mu) sigma) 2))))

(defn- close? [a b tol] (< (js/Math.abs (- a b)) tol))

(defn- w-item [result] (mx/item (:weight result)))

(defn- choice-at
  "JS number at path in a trace's choices."
  [trace path]
  (mx/item (cm/get-choice (:choices trace) path)))

(defn- unsupported-error?
  "True iff (f) throws the update-with-args-unsupported contract error."
  [f]
  (try (f) false
       (catch :default e
         (= :update-with-args-unsupported (:genmlx/error (ex-data e))))))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

(defn- shift-model
  "x ~ N(m,1); y ~ N(x,1) — static, structure-preserving under m change."
  []
  (gen [m]
    (let [x (trace :x (dist/gaussian m 1))]
      (trace :y (dist/gaussian x 1))
      x)))

(defn- grow-model
  "mu ~ N(0,1); y_i ~ N(mu,1) for i < n — address set grows with n."
  []
  (gen [n]
    (let [mu (trace :mu (dist/gaussian 0 1))]
      (doseq [i (range n)]
        (trace (keyword (str "y" i)) (dist/gaussian mu 1)))
      mu)))

(defn- prefix-model
  "Static prefix + dynamic loop — exercises the L1-M3 prefix update path."
  []
  (gen [n]
    (let [mu (trace :mu (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
      (doseq [i (range n)]
        (trace (keyword (str "y" i)) (dist/gaussian mu (mx/scalar 1))))
      mu)))

(defn- uniform-model
  "u ~ U(0, hi) — support shrinks with hi."
  []
  (gen [hi]
    (trace :u (dist/uniform 0 hi))))

(defn- sub-model
  "z ~ N(m,1) — spliced child."
  []
  (gen [m]
    (trace :z (dist/gaussian m 1))))

(defn- splice-model
  "Parent passes a derived arg to the splice — new parent args must
   propagate to the child through body re-execution alone."
  [sub]
  (gen [m]
    (splice :sub sub (* 2 m))))

;; ============================================================
;; 1. No-op equivalence: new-args = old args === p/update
;; ============================================================

(deftest noop-equivalence-unconstrained
  (let [k (rng/fresh-key 1)
        model (dyn/with-key (shift-model) k)
        t (p/simulate model [0.5])
        upd (p/update (dyn/with-key (shift-model) (rng/fresh-key 2)) t cm/EMPTY)
        uwa (p/update-with-args (dyn/with-key (shift-model) (rng/fresh-key 2))
                                t [0.5] :unknown cm/EMPTY)]
    (is (close? (mx/item (:score (:trace upd))) (mx/item (:score (:trace uwa))) 1e-6)
        "same score as p/update")
    (is (close? (w-item upd) (w-item uwa) 1e-6) "same weight as p/update")
    (is (= (choice-at (:trace upd) [:x]) (choice-at (:trace uwa) [:x]))
        "same retained :x")
    (is (= [0.5] (:args (:trace uwa))) "trace args unchanged")))

(deftest noop-equivalence-constrained
  (let [model (dyn/with-key (shift-model) (rng/fresh-key 3))
        t (p/simulate model [0.5])
        sigma (cm/choicemap :y (mx/scalar 2.0))
        upd (p/update (dyn/with-key (shift-model) (rng/fresh-key 4)) t sigma)
        uwa (p/update-with-args (dyn/with-key (shift-model) (rng/fresh-key 4))
                                t [0.5] :unknown sigma)]
    (is (close? (w-item upd) (w-item uwa) 1e-6)
        "constrained no-op matches p/update weight")
    (is (close? (mx/item (cm/get-choice (:discard uwa) [:y]))
                (choice-at t [:y]) 1e-6)
        "discard carries the overwritten :y")))

(deftest no-change-fast-path
  (let [model (dyn/with-key (shift-model) (rng/fresh-key 5))
        t (p/simulate model [0.5])
        r (p/update-with-args model t [0.5] diff/no-change cm/EMPTY)]
    (is (identical? t (:trace r)) "no-change + empty constraints returns the trace")
    (is (zero? (mx/item (:weight r))) "zero weight")
    (is (= cm/EMPTY (:discard r)) "empty discard")))

;; ============================================================
;; 2. Weight = density ratio under new args (the payload)
;; ============================================================

(deftest weight-is-density-ratio-unconstrained
  (let [model (dyn/with-key (shift-model) (rng/fresh-key 7))
        t (p/simulate model [0.0])
        x (choice-at t [:x])
        r (p/update-with-args model t [1.5] :unknown cm/EMPTY)
        expected (- (log-normal-pdf x 1.5 1) (log-normal-pdf x 0.0 1))]
    (is (close? (w-item r) expected 1e-5)
        "w = log N(x;m',1) - log N(x;m,1), hand-derived")
    (is (= [1.5] (:args (:trace r))) "new trace carries new args")
    (is (close? (choice-at (:trace r) [:x]) x 1e-9) "x retained verbatim")
    (is (close? (mx/item (:score (:trace r)))
                (+ (log-normal-pdf x 1.5 1)
                   (log-normal-pdf (choice-at t [:y]) x 1))
                1e-5)
        "new score is the full joint under new args")))

(deftest weight-matches-assess-ratio
  (let [model (dyn/with-key (shift-model) (rng/fresh-key 8))
        t (p/simulate model [0.0])
        r (p/update-with-args model t [2.0] :unknown cm/EMPTY)
        a-new (mx/item (:weight (p/assess model [2.0] (:choices (:trace r)))))
        a-old (mx/item (:weight (p/assess model [0.0] (:choices t))))]
    (is (close? (w-item r) (- a-new a-old) 1e-5)
        "structure-preserving: w = assess(x',t') - assess(x,t)")))

(deftest weight-with-constraint-and-new-args
  (let [model (dyn/with-key (shift-model) (rng/fresh-key 9))
        t (p/simulate model [0.0])
        x (choice-at t [:x])
        y-old (choice-at t [:y])
        r (p/update-with-args model t [1.0] :unknown
                              (cm/choicemap :y (mx/scalar 3.0)))
        nonfresh (+ (log-normal-pdf x 1.0 1) (log-normal-pdf 3.0 x 1))
        old-score (+ (log-normal-pdf x 0.0 1) (log-normal-pdf y-old x 1))]
    (is (close? (w-item r) (- nonfresh old-score) 1e-5)
        "w = nonfresh(x') - score(t;x), both terms hand-derived")
    (is (close? (choice-at (:trace r) [:y]) 3.0 1e-9) "constraint applied")
    (is (close? (mx/item (cm/get-choice (:discard r) [:y])) y-old 1e-9)
        "old :y in discard")))

;; ============================================================
;; 3. Fresh and removed sites (address set changes with args)
;; ============================================================

(deftest fresh-site-constrained
  (let [model (dyn/with-key (grow-model) (rng/fresh-key 11))
        {t :trace} (p/generate model [2] (cm/choicemap :y0 (mx/scalar 1.0)
                                                       :y1 (mx/scalar -1.0)))
        mu (choice-at t [:mu])
        r (p/update-with-args model t [3] :unknown
                              (cm/choicemap :y2 (mx/scalar 0.5)))]
    (is (close? (w-item r) (log-normal-pdf 0.5 mu 1) 1e-5)
        "constrained fresh site contributes exactly its new log-prob")
    (is (close? (choice-at (:trace r) [:y2]) 0.5 1e-9) "new site present")))

(deftest fresh-site-unconstrained-cancels
  (let [model (dyn/with-key (grow-model) (rng/fresh-key 12))
        {t :trace} (p/generate model [1] (cm/choicemap :y0 (mx/scalar 1.0)))
        r (p/update-with-args model t [2] :unknown cm/EMPTY)]
    (is (close? (w-item r) 0.0 1e-5)
        "unconstrained fresh site cancels: retained dists unchanged => w = 0")
    (is (some? (cm/get-choice (:choices (:trace r)) [:y1]))
        "fresh :y1 was sampled")))

(deftest removed-site-charged-and-discarded
  (let [model (dyn/with-key (grow-model) (rng/fresh-key 13))
        {t :trace} (p/generate model [2] (cm/choicemap :y0 (mx/scalar 1.0)
                                                       :y1 (mx/scalar -1.0)))
        mu (choice-at t [:mu])
        r (p/update-with-args model t [1] :unknown cm/EMPTY)]
    (is (close? (w-item r) (- (log-normal-pdf -1.0 mu 1)) 1e-5)
        "removed site charged via the old score: w = -lp(y1)")
    (is (close? (mx/item (cm/get-choice (:discard r) [:y1])) -1.0 1e-9)
        "removed :y1 value lands in the discard")
    (is (nil? (cm/get-choice (:choices (:trace r)) [:y1]))
        "removed site absent from new choices")))

(deftest retained-value-out-of-support
  (let [model (dyn/with-key (uniform-model) (rng/fresh-key 14))
        {t :trace} (p/generate model [10.0] (cm/choicemap :u (mx/scalar 5.0)))
        r (p/update-with-args model t [1.0] :unknown cm/EMPTY)]
    (is (= js/Number.NEGATIVE_INFINITY (w-item r))
        "retained value outside new support => -inf weight, not an error")))

;; ============================================================
;; 4. Splice propagation (no executor change — body re-execution)
;; ============================================================

(deftest splice-new-args-propagate
  (let [sub (sub-model)
        model (dyn/with-key (splice-model sub) (rng/fresh-key 15))
        t (p/simulate model [1.0])
        z (choice-at t [:sub :z])
        r (p/update-with-args model t [2.0] :unknown cm/EMPTY)
        expected (- (log-normal-pdf z 4.0 1) (log-normal-pdf z 2.0 1))]
    (is (close? (w-item r) expected 1e-5)
        "child :z re-scored under the NEW derived sub-arg (2m)")
    (is (close? (choice-at (:trace r) [:sub :z]) z 1e-9)
        "child choice retained verbatim")))

;; ============================================================
;; 5. Prefix-compiled path (L1-M3) threads new args
;; ============================================================

(deftest prefix-path-new-args
  (let [model (dyn/with-key (prefix-model) (rng/fresh-key 16))
        {t :trace} (p/generate model [1] (cm/choicemap :y0 (mx/scalar 1.0)))
        mu (choice-at t [:mu])
        r (p/update-with-args model t [2] :unknown
                              (cm/choicemap :y1 (mx/scalar 0.0)))]
    (is (close? (w-item r) (log-normal-pdf 0.0 mu 1) 1e-5)
        "prefix model: growing the dynamic suffix adds the new site's lp")))

;; ============================================================
;; 6. Compiled parity (L1-M2): compiled === handler under new args
;; ============================================================

(deftest compiled-parity-static-model
  (let [mk #(dyn/with-key (shift-model) (rng/fresh-key 17))
        compiled (mk)
        stripped (gfi/strip-compiled (mk))
        t-c (p/simulate compiled [0.0])
        ;; same choices into the stripped model via fully-constrained generate
        t-h (:trace (p/generate stripped [0.0] (:choices t-c)))
        r-c (p/update-with-args compiled t-c [1.0] :unknown cm/EMPTY)
        r-h (p/update-with-args stripped t-h [1.0] :unknown cm/EMPTY)]
    (is (close? (w-item r-c) (w-item r-h) 1e-5)
        "compiled and handler paths agree on the weight")
    (is (close? (mx/item (:score (:trace r-c)))
                (mx/item (:score (:trace r-h))) 1e-5)
        "and on the new score")))

;; ============================================================
;; 7. Score-type boundary (lbae guard runs on the new entry)
;; ============================================================

(deftest marginal-trace-converts-before-update-with-args
  (let [model (dyn/with-key
                (gen []
                  (let [mu (trace :mu (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                    (trace :y (dist/gaussian mu (mx/scalar 1)))
                    mu))
                (rng/fresh-key 18))
        {t :trace} (p/generate model [] (cm/choicemap :y (mx/scalar 1.5)))]
    (when (= :marginal (tr/score-type t))
      (let [r (p/update-with-args model t [] :unknown cm/EMPTY)]
        (is (= :joint (tr/score-type (:trace r)))
            "marginal input converted: result is joint-scored")
        (is (js/isFinite (w-item r)) "finite weight after conversion")))))

;; ============================================================
;; 8. Map combinator
;; ============================================================

(defn- map-kernel [] (gen [m] (trace :v (dist/gaussian m 1))))

(deftest map-element-change-vector-diff-and-unknown
  (let [mapped (dyn/with-key (comb/map-combinator (map-kernel)) (rng/fresh-key 21))
        t (p/simulate mapped [[0.0 1.0]])
        v1 (choice-at t [1 :v])
        expected (- (log-normal-pdf v1 2.0 1) (log-normal-pdf v1 1.0 1))
        r-vd (p/update-with-args mapped t [[0.0 2.0]]
                                 (diff/vector-diff #{1}) cm/EMPTY)
        r-uk (p/update-with-args mapped t [[0.0 2.0]] :unknown cm/EMPTY)]
    (is (close? (w-item r-vd) expected 1e-5) "vector-diff fast path: exact ratio")
    (is (close? (w-item r-uk) expected 1e-5) "full sweep: same exact ratio")
    (is (close? (choice-at (:trace r-vd) [0 :v]) (choice-at t [0 :v]) 1e-9)
        "untouched element retained")))

(deftest map-grow-constrained
  (let [mapped (dyn/with-key (comb/map-combinator (map-kernel)) (rng/fresh-key 22))
        t (p/simulate mapped [[0.0 1.0]])
        r (p/update-with-args mapped t [[0.0 1.0 2.0]] :unknown
                              (cm/set-choice cm/EMPTY [2 :v] (mx/scalar 2.5)))]
    (is (close? (w-item r) (log-normal-pdf 2.5 2.0 1) 1e-5)
        "constrained new element contributes its log-prob")
    (is (close? (choice-at (:trace r) [2 :v]) 2.5 1e-9) "element present")))

(deftest map-shrink-discards
  (let [mapped (dyn/with-key (comb/map-combinator (map-kernel)) (rng/fresh-key 23))
        t (p/simulate mapped [[0.0 1.0]])
        v1 (choice-at t [1 :v])
        r (p/update-with-args mapped t [[0.0]] :unknown cm/EMPTY)]
    (is (close? (w-item r) (- (log-normal-pdf v1 1.0 1)) 1e-5)
        "dropped element charged via old score")
    (is (close? (mx/item (cm/get-choice (:discard r) [1 :v])) v1 1e-9)
        "dropped element's choices in discard")))

;; ============================================================
;; 9. Unfold combinator
;; ============================================================

(defn- unfold-kernel [] (gen [t prev] (trace :s (dist/gaussian prev 1))))

(deftest unfold-extend-equivalence
  (let [unf (dyn/with-key (comb/unfold-combinator (unfold-kernel)) (rng/fresh-key 24))
        t (p/simulate unf [3 0.0])
        sigma (cm/choicemap :s (mx/scalar 1.0))
        ;; fully-constrained new step: both paths are deterministic
        ext (comb/unfold-extend t sigma (rng/fresh-key 25))
        r (p/update-with-args unf t [4 0.0] :unknown
                              (cm/set-choice cm/EMPTY [3] sigma))]
    (is (close? (w-item ext) (w-item r) 1e-5)
        "unfold-extend is the n'=n+1 special case (same weight)")
    (is (close? (mx/item (:score (:trace ext))) (mx/item (:score (:trace r))) 1e-5)
        "same extended score")
    (is (close? (choice-at (:trace r) [3 :s]) 1.0 1e-9) "new step constrained")))

(deftest unfold-truncate
  (let [unf (dyn/with-key (comb/unfold-combinator (unfold-kernel)) (rng/fresh-key 26))
        t (p/simulate unf [3 0.0])
        s1 (choice-at t [1 :s])
        s2 (choice-at t [2 :s])
        r (p/update-with-args unf t [2 0.0] :unknown cm/EMPTY)]
    (is (close? (w-item r) (- (log-normal-pdf s2 s1 1)) 1e-5)
        "truncated step charged via old score")
    (is (close? (mx/item (cm/get-choice (:discard r) [2 :s])) s2 1e-9)
        "truncated step in discard")
    (is (nil? (cm/get-choice (:choices (:trace r)) [2 :s]))
        "step absent from new choices")))

(deftest unfold-init-state-change-full-sweep
  (let [unf (dyn/with-key (comb/unfold-combinator (unfold-kernel)) (rng/fresh-key 27))
        t (p/simulate unf [2 0.0])
        s0 (choice-at t [0 :s])
        r (p/update-with-args unf t [2 5.0] :unknown cm/EMPTY)
        expected (- (log-normal-pdf s0 5.0 1) (log-normal-pdf s0 0.0 1))]
    (is (close? (w-item r) expected 1e-5)
        "only step 0 depends on init: w is its hand-derived ratio")
    (is (close? (choice-at (:trace r) [1 :s]) (choice-at t [1 :s]) 1e-9)
        "downstream step retained (its dist depends on s0, unchanged)")))

;; ============================================================
;; 10. Scan combinator
;; ============================================================

(defn- scan-kernel []
  (gen [carry x]
    (let [v (trace :v (dist/gaussian (+ carry x) 1))]
      [v v])))

(deftest scan-grow-constrained
  (let [sc (dyn/with-key (comb/scan-combinator (scan-kernel)) (rng/fresh-key 28))
        t (p/simulate sc [0.0 [1.0 2.0]])
        v1 (choice-at t [1 :v])             ;; carry after step 1 = v1
        r (p/update-with-args sc t [0.0 [1.0 2.0 3.0]] :unknown
                              (cm/set-choice cm/EMPTY [2 :v] (mx/scalar 0.0)))]
    (is (close? (w-item r) (log-normal-pdf 0.0 (+ v1 3.0) 1) 1e-5)
        "new step scored at carry+input, hand-derived")))

(deftest scan-input-change-full-sweep
  (let [sc (dyn/with-key (comb/scan-combinator (scan-kernel)) (rng/fresh-key 29))
        t (p/simulate sc [0.0 [1.0 2.0]])
        v0 (choice-at t [0 :v])
        v1 (choice-at t [1 :v])
        r (p/update-with-args sc t [0.0 [4.0 2.0]] :unknown cm/EMPTY)
        ;; step 0: mean 0+4 (was 0+1); step 1: carry v0 unchanged, mean v0+2
        expected (- (log-normal-pdf v0 4.0 1) (log-normal-pdf v0 1.0 1))]
    (is (close? (w-item r) expected 1e-5)
        "changed input re-scores step 0; step 1's carry is the retained v0")))

;; ============================================================
;; 11. Switch combinator
;; ============================================================

(deftest switch-same-branch-delegates
  (let [b0 (gen [m] (trace :a (dist/gaussian m 1)))
        b1 (gen [m] (trace :b (dist/gaussian m 2)))
        sw (dyn/with-key (comb/switch-combinator b0 b1) (rng/fresh-key 31))
        t (p/simulate sw [0 0.0])
        a (choice-at t [:a])
        r (p/update-with-args sw t [0 1.0] :unknown cm/EMPTY)
        expected (- (log-normal-pdf a 1.0 1) (log-normal-pdf a 0.0 1))]
    (is (close? (w-item r) expected 1e-5) "same branch: delegated ratio")))

(deftest switch-branch-flip
  (let [b0 (gen [m] (trace :a (dist/gaussian m 1)))
        b1 (gen [m] (trace :b (dist/gaussian m 2)))
        sw (dyn/with-key (comb/switch-combinator b0 b1) (rng/fresh-key 32))
        t (p/simulate sw [0 0.0])
        a (choice-at t [:a])
        r (p/update-with-args sw t [1 0.0] :unknown
                              (cm/choicemap :b (mx/scalar 1.0)))]
    (is (close? (w-item r)
                (- (log-normal-pdf 1.0 0.0 2) (log-normal-pdf a 0.0 1))
                1e-5)
        "flip: new branch nonfresh minus old branch score")
    (is (close? (mx/item (cm/get-choice (:discard r) [:a])) a 1e-9)
        "old branch choices discarded")))

;; ============================================================
;; 12. Contramap / MapRetval delegation; unsupported combinators throw
;; ============================================================

(deftest contramap-delegates
  (let [inner (dyn/with-key (shift-model) (rng/fresh-key 33))
        cgf (comb/contramap-gf inner (fn [[m]] [(* 2 m)]))
        t (p/simulate cgf [0.5])             ;; inner sees m=1.0
        x (choice-at t [:x])
        r (p/update-with-args cgf t [1.0] :unknown cm/EMPTY)  ;; inner sees 2.0
        expected (- (log-normal-pdf x 2.0 1) (log-normal-pdf x 1.0 1))]
    (is (close? (w-item r) expected 1e-5) "f applied to new args before inner")))

(deftest map-retval-delegates
  (let [inner (dyn/with-key (shift-model) (rng/fresh-key 34))
        mgf (comb/map-retval inner (fn [x] (mx/multiply x (mx/scalar 10))))
        t (p/simulate mgf [0.0])
        r (p/update-with-args mgf t [1.0] :unknown cm/EMPTY)]
    (is (js/isFinite (w-item r)) "delegates and produces a finite weight")
    (is (close? (mx/item (:retval (:trace r)))
                (* 10 (choice-at (:trace r) [:x])) 1e-4)
        "retval re-mapped on the updated trace")))

(deftest unsupported-combinators-throw
  (let [inner (dyn/auto-key (gen [] (trace :x (dist/gaussian 0 1))))
        masked (comb/mask-combinator inner)
        t (p/simulate masked [true])]
    (is (unsupported-error?
          #(p/update-with-args masked t [false] :unknown cm/EMPTY))
        "Mask throws the informative unsupported error (not silence)")))

;; ============================================================
;; 13. Edit flavor: ArgsUpdateEdit forward/backward roundtrip
;; ============================================================

(deftest args-update-edit-roundtrip
  (let [model (dyn/with-key (shift-model) (rng/fresh-key 35))
        t (p/simulate model [0.0])
        fwd (edit/edit model t (edit/args-update-edit [1.0] cm/EMPTY))
        bwd-req (:backward-request fwd)
        bwd (edit/edit model (:trace fwd) bwd-req)]
    (is (instance? edit/ArgsUpdateEdit bwd-req) "backward request is an ArgsUpdateEdit")
    (is (= [0.0] (:new-args bwd-req)) "backward carries the old args")
    (is (close? (w-item bwd) (- (w-item fwd)) 1e-5)
        "structure-preserving roundtrip: w_back = -w_fwd")
    (is (close? (choice-at (:trace bwd) [:x]) (choice-at t [:x]) 1e-9)
        "roundtrip restores choices")))

;; ============================================================
;; 14. Vectorized path: vupdate-args
;; ============================================================

(deftest vupdate-args-matches-scalar-oracle
  (let [n 8
        model (dyn/with-key (shift-model) (rng/fresh-key 36))
        vt (dyn/vsimulate model [0.0] n (rng/fresh-key 37))
        xs (mx/->clj (cm/get-value (cm/get-submap (:choices vt) :x)))
        {w :weight vt' :vtrace} (dyn/vupdate-args model vt [1.0] cm/EMPTY
                                                  (rng/fresh-key 38))
        ws (mx/->clj w)]
    (is (= n (count ws)) "weight is [N]-shaped")
    (doseq [[x w-i] (map vector xs ws)]
      (is (close? w-i (- (log-normal-pdf x 1.0 1) (log-normal-pdf x 0.0 1)) 1e-4)
          "each particle's weight matches the hand-derived ratio"))
    (is (= [1.0] (:args vt')) "vtrace carries new args")))

;; ============================================================
;; 15. SMC with changing arguments, end-to-end (growing data)
;; ============================================================

(deftest smc-args-growing-data-log-ml
  ;; mu ~ N(0,1); y_i ~ N(mu,1). Closed-form sequential predictive:
  ;; after i obs, posterior mu | y_<i ~ N(m_i, s_i^2), s_i^2 = 1/(1+i),
  ;; m_i = s_i^2 * sum(y_<i); predictive y_i ~ N(m_i, sqrt(s_i^2 + 1)).
  (let [ys [1.0 -0.5 0.8 0.2]
        closed-form (loop [i 0 sum-y 0.0 acc 0.0]
                      (if (= i (count ys))
                        acc
                        (let [s2 (/ 1.0 (+ 1 i))
                              m (* s2 sum-y)
                              y (nth ys i)]
                          (recur (inc i) (+ sum-y y)
                                 (+ acc (log-normal-pdf y m (js/Math.sqrt (+ s2 1.0))))))))
        steps (map-indexed
                (fn [t y]
                  {:args [(inc t)]
                   :constraints (cm/choicemap (keyword (str "y" t)) (mx/scalar y))})
                ys)
        {:keys [log-ml traces final-ess]}
        (smc/smc-args {:particles 300 :key (rng/fresh-key 39)}
                      (grow-model) steps)]
    (is (close? (mx/item log-ml) closed-form 0.25)
        (str "SMC log-ML " (mx/item log-ml) " within MC tolerance of closed form " closed-form))
    (is (= 300 (count traces)) "all particles survive")
    (is (pos? final-ess) "honest pre-resample final ESS reported")
    (is (every? #(= [4] (:args %)) traces) "final traces carry the final args")))

(defn- linreg-model
  "slope ~ N(0,1); intercept ~ N(0,1); y_i ~ N(slope*x_i + intercept, 1)
   for i < (count xs) — stable y_i addresses, args grow with the data."
  []
  (gen [xs]
    (let [slope (trace :slope (dist/gaussian 0 1))
          intercept (trace :intercept (dist/gaussian 0 1))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace (keyword (str "y" i))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar x))
                                      intercept)
                              1)))
      slope)))

(defn- linreg-log-ml
  "Hand-derived sequential log-ML for Bayesian linear regression with
   prior N(0, I) on [slope intercept] and unit observation noise:
   log p(y) = sum_t log N(y_t; phi_t . m_(t-1), sqrt(phi_t . S_(t-1) phi_t + 1))
   with S = inv(I + sum phi phi^T), m = S (sum phi y) — 2x2 algebra inline,
   independent of all genmlx code."
  [xs ys]
  (loop [t 0
         ;; precision Lambda = I + sum phi phi^T, b = sum phi y
         l00 1.0 l01 0.0 l11 1.0 b0 0.0 b1 0.0
         acc 0.0]
    (if (= t (count ys))
      acc
      (let [x (nth xs t) y (nth ys t)
            det (- (* l00 l11) (* l01 l01))
            ;; S = inv(Lambda)
            s00 (/ l11 det) s01 (- (/ l01 det)) s11 (/ l00 det)
            ;; posterior mean m = S b
            m0 (+ (* s00 b0) (* s01 b1))
            m1 (+ (* s01 b0) (* s11 b1))
            ;; predictive: mean phi.m, var phi.S.phi + 1, phi = [x 1]
            mean (+ (* x m0) m1)
            var (+ (* x (+ (* s00 x) s01))
                   (+ (* s01 x) s11)
                   1.0)]
        (recur (inc t)
               (+ l00 (* x x)) (+ l01 x) (+ l11 1.0)
               (+ b0 (* x y)) (+ b1 y)
               (+ acc (log-normal-pdf y mean (js/Math.sqrt var))))))))

(deftest smc-args-growing-linreg-log-ml
  (let [xs [0.0 1.0 2.0 3.0]
        ys [0.5 1.8 2.9 4.4]
        closed-form (linreg-log-ml xs ys)
        steps (map-indexed
                (fn [t y]
                  {:args [(subvec (vec xs) 0 (inc t))]
                   :constraints (cm/choicemap (keyword (str "y" t)) (mx/scalar y))})
                ys)
        {:keys [log-ml traces]}
        (smc/smc-args {:particles 400 :key (rng/fresh-key 42)}
                      (linreg-model) steps)]
    (is (close? (mx/item log-ml) closed-form 0.3)
        (str "growing-data linreg SMC log-ML " (mx/item log-ml)
             " within MC tolerance of hand-derived closed form " closed-form))
    (is (every? #(= [[0.0 1.0 2.0 3.0]] (:args %)) traces)
        "final traces carry the full xs")))

;; ============================================================
;; 16. Laws registered
;; ============================================================

(deftest update-args-laws-registered
  (let [names (set (map :name gfi/laws))]
    (doseq [law [:update-args-noop
                 :update-args-weight-is-density-ratio
                 :update-args-roundtrip
                 :update-args-compiled-parity
                 :unfold-extend-equivalence
                 :update-args-structure-change]]
      (is (contains? names law) (str law " is in the law catalog")))))

(cljs.test/run-tests)
