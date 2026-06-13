;; @tier medium
(ns genmlx.regen-structure-test
  "genmlx-hmch + genmlx-yep2: regenerate through structure change, and the
   retained-only MH weight algebra.

   The DML regenerate weight is, for selection S with the internal prior
   proposal, W = Σ over RETAINED sites of [lp(v; new ctx) - lp(v; old ctx)],
   where retained = unselected AND present in BOTH executions; selected,
   freshly-appearing, and removed sites all cancel to 0 (proven in the
   math-verifier derivation on genmlx-hmch).

   Every oracle here is an INDEPENDENT closed form (h/gaussian-lp, Bayes'
   rule) — never computed via the regenerate code under test. See
   feedback_independent_oracle_tests."
  (:require-macros [genmlx.gen :refer [gen]])
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.choicemap :as cm]
            [genmlx.trace :as tr]
            [genmlx.selection :as sel]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn force-handler
  "Strip all compiled/analytical paths so a gen-fn runs through the handler."
  [gf]
  (assoc gf :schema
         (dissoc (:schema gf)
                 :compiled-simulate :compiled-generate :compiled-update
                 :compiled-assess :compiled-project :compiled-regenerate
                 :compiled-prefix :compiled-prefix-addrs :compiled-prefix-generate
                 :compiled-prefix-update :compiled-prefix-assess
                 :compiled-prefix-project :compiled-prefix-regenerate
                 :auto-handlers :conjugate-pairs :has-conjugate?
                 :analytical-plan :auto-regenerate-transition)))

(defn cval
  "Realize a scalar choice value at address."
  [choices addr]
  (mx/item (cm/get-value (cm/get-submap choices addr))))

(defn has-addr? [choices addr]
  (cm/has-value? (cm/get-submap choices addr)))

(defn gen-trace
  "Deterministic trace via generate with the given constraints."
  [gf args constraints key]
  (:trace (p/generate (dyn/with-key (force-handler gf) key) args constraints)))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

;; yep2: dependent pair — b's params depend on a.
(def dep-pair
  (gen []
    (let [a (trace :a (dist/gaussian 0 1))
          b (trace :b (dist/gaussian a 1))]
      b)))

;; yep2 cascade: a -> b -> y, y observed/retained.
(def cascade
  (gen [sigma]
    (let [a (trace :a (dist/gaussian 0 1))
          b (trace :b (dist/gaussian a 1))
          y (trace :y (dist/gaussian b sigma))]
      y)))

;; Independent pair — fast-path-eligible (no dependency between selected sites).
(def indep-pair
  (gen []
    (let [a (trace :a (dist/gaussian 0 1))
          b (trace :b (dist/gaussian 3 2))]
      (mx/add a b))))

;; Branch flip, SHARED :y address (no structure change), per-arm params.
(def branch-shared-y
  (gen [p mu1 s1 mu0 s0]
    (let [bb (trace :branch (dist/bernoulli p))]
      (if (pos? (mx/item bb))
        (trace :y (dist/gaussian mu1 s1))
        (trace :y (dist/gaussian mu0 s0))))))

;; Prior-only flip — DIFFERENT latent addresses per arm (structure change),
;; no observation.  Flipping :branch makes one arm's latent fresh and the
;; other's removed.
(def prior-flip
  (gen []
    (let [bb (trace :branch (dist/bernoulli 0.5))]
      (if (pos? (mx/item bb))
        (trace :xa (dist/gaussian 0 1))
        (trace :xb (dist/gaussian 5 1))))))

;; Structure-change WITH a shared observation: flipping :branch swaps the
;; latent address (:ya <-> :yb, structure change) AND the mean of the shared
;; :obs leaf.  The latent does not feed :obs, so it integrates out and the
;; posterior over :branch is plain Bayes on the two obs-means.
(def struct-obs
  (gen [obs]
    (let [bb (trace :branch (dist/bernoulli 0.5))]
      (if (pos? (mx/item bb))
        (do (trace :ya (dist/gaussian 0 1))
            (trace :obs (dist/gaussian 2.0 1.0)))
        (do (trace :yb (dist/gaussian 5 1))
            (trace :obs (dist/gaussian -2.0 1.0)))))))

;; Splice: parent splices the dependent cascade child, plus a parent-direct
;; retained observation.
(def child-cascade
  (gen [sigma]
    (let [a (trace :a (dist/gaussian 0 1))
          b (trace :b (dist/gaussian a 1))]
      b)))

(def parent-splice
  (gen [sigma]
    (let [b (splice :child child-cascade [sigma])
          y (trace :y (dist/gaussian b sigma))]
      y)))

;; ---------------------------------------------------------------------------
;; 1. yep2 — select-all on a DEPENDENT pair: W = 0 (every draw)
;; ---------------------------------------------------------------------------

(deftest select-all-dependent-zero
  (testing "handler: select {a,b} on b~N(a,1) => weight 0"
    (dotimes [i 25]
      (let [t  (gen-trace dep-pair [] (cm/choicemap :a (mx/scalar 0.7) :b (mx/scalar 1.3))
                          (rng/fresh-key (+ 100 i)))
            {:keys [weight]} (p/regenerate (dyn/with-key (force-handler dep-pair) (rng/fresh-key (+ 500 i)))
                                           t (sel/from-paths [[:a] [:b]]))]
        (is (h/close? 0.0 (mx/item weight) 1e-4)
            "dependent select-all weight is exactly 0"))))
  (testing "auto-dispatched (compilable model): dependent select-all routes to general, weight 0"
    ;; dep-pair is static/compilable, but a dependent JOINT selection is not
    ;; fast-eligible, so the dispatcher defers the compiled per-site regen to
    ;; the handler general path. This pins that the dispatched (non-forced)
    ;; result is the correct 0, never the buggy per-site residual.
    (dotimes [i 15]
      (let [t  (gen-trace dep-pair [] (cm/choicemap :a (mx/scalar 0.7) :b (mx/scalar 1.3))
                          (rng/fresh-key (+ 200 i)))
            {:keys [weight]} (p/regenerate (dyn/with-key dep-pair (rng/fresh-key (+ 600 i)))
                                           t (sel/from-paths [[:a] [:b]]))]
        (is (h/close? 0.0 (mx/item weight) 1e-4)
            "dispatched dependent select-all weight is 0")))))

;; ---------------------------------------------------------------------------
;; 2. yep2 — cascading pair {a,b} selected, y retained: closed-form W
;; ---------------------------------------------------------------------------

(deftest cascading-pair-oracle
  (testing "handler: W = logN(y;b_new,σ) - logN(y;b_old,σ)"
    (let [sigma 0.7
          y     1.1]
      (dotimes [i 25]
        (let [t (gen-trace cascade [sigma]
                           (cm/choicemap :a (mx/scalar 0.3) :b (mx/scalar 0.9) :y (mx/scalar y))
                           (rng/fresh-key (+ 700 i)))
              b-old (cval (:choices t) :b)
              {:keys [trace weight]} (p/regenerate
                                       (dyn/with-key (force-handler cascade) (rng/fresh-key (+ 800 i)))
                                       t (sel/from-paths [[:a] [:b]]))
              b-new  (cval (:choices trace) :b)
              oracle (- (h/gaussian-lp y b-new sigma) (h/gaussian-lp y b-old sigma))]
          (is (h/close? oracle (mx/item weight) 1e-4)
              "cascade weight matches retained-y log-density delta")))))
  (testing "auto-dispatched (compilable model): dependent joint routes to general, same closed form"
    (let [sigma 0.7 y 1.1]
      (dotimes [i 15]
        (let [t (gen-trace cascade [sigma]
                           (cm/choicemap :a (mx/scalar 0.3) :b (mx/scalar 0.9) :y (mx/scalar y))
                           (rng/fresh-key (+ 900 i)))
              b-old (cval (:choices t) :b)
              {:keys [trace weight]} (p/regenerate (dyn/with-key cascade (rng/fresh-key (+ 1000 i)))
                                                   t (sel/from-paths [[:a] [:b]]))
              b-new (cval (:choices trace) :b)
              oracle (- (h/gaussian-lp y b-new sigma) (h/gaussian-lp y b-old sigma))]
          (is (h/close? oracle (mx/item weight) 1e-4)
              "compiled cascade weight matches oracle"))))))

;; ---------------------------------------------------------------------------
;; 3. fast ≡ general path equivalence (single-site & independent selections)
;; ---------------------------------------------------------------------------

(deftest fast-general-equivalence
  (testing "single-site selection: fast path == forced-general path"
    (dotimes [i 20]
      (let [t (gen-trace cascade [0.7]
                         (cm/choicemap :a (mx/scalar 0.3) :b (mx/scalar 0.9) :y (mx/scalar 1.1))
                         (rng/fresh-key (+ 1100 i)))
            k (rng/fresh-key (+ 1200 i))
            fast    (p/regenerate (dyn/with-key (force-handler cascade) k) t (sel/select :b))
            general (binding [dyn/*force-general-regen* true]
                      (p/regenerate (dyn/with-key (force-handler cascade) k) t (sel/select :b)))]
        (is (h/close? (mx/item (:weight fast)) (mx/item (:weight general)) 1e-5)
            "single-site fast == general weight")
        (is (h/close? (mx/item (:score (:trace fast))) (mx/item (:score (:trace general))) 1e-5)
            "single-site fast == general score"))))
  (testing "independent joint selection: fast path == forced-general path, both 0"
    (dotimes [i 20]
      (let [t (gen-trace indep-pair [] (cm/choicemap :a (mx/scalar 0.5) :b (mx/scalar 4.0))
                         (rng/fresh-key (+ 1300 i)))
            k (rng/fresh-key (+ 1400 i))
            fast    (p/regenerate (dyn/with-key (force-handler indep-pair) k) t (sel/from-paths [[:a] [:b]]))
            general (binding [dyn/*force-general-regen* true]
                      (p/regenerate (dyn/with-key (force-handler indep-pair) k) t (sel/from-paths [[:a] [:b]])))]
        (is (h/close? (mx/item (:weight fast)) (mx/item (:weight general)) 1e-5)
            "independent fast == general")
        (is (h/close? 0.0 (mx/item (:weight general)) 1e-4)
            "independent select-all weight 0")))))

;; ---------------------------------------------------------------------------
;; 4. Branch flip with SHARED :y — weight oracle + MH posterior
;; ---------------------------------------------------------------------------

(defn- normal-pdf [x mu s]
  (js/Math.exp (h/gaussian-lp x mu s)))

(deftest branch-flip-shared-weight-oracle
  (testing "flip 0->1 weight = logN(y;mu1,s1) - logN(y;mu0,s0)"
    (let [args [0.5 2.0 1.0 -2.0 1.0]  ; p mu1 s1 mu0 s0
          yv   0.4]
      (dotimes [i 20]
        ;; force branch=0 old trace
        (let [t (gen-trace branch-shared-y args
                           (cm/choicemap :branch (mx/scalar 0.0) :y (mx/scalar yv))
                           (rng/fresh-key (+ 1500 i)))
              {:keys [trace weight]} (p/regenerate
                                       (dyn/with-key (force-handler branch-shared-y) (rng/fresh-key (+ 1600 i)))
                                       t (sel/select :branch))
              b-new (cval (:choices trace) :branch)
              oracle (if (pos? b-new)
                       (- (h/gaussian-lp yv 2.0 1.0) (h/gaussian-lp yv -2.0 1.0))   ; flipped 0->1
                       0.0)]                                                          ; stayed 0
          (is (h/close? oracle (mx/item weight) 1e-4)
              "shared-y branch flip weight matches likelihood log-ratio"))))))

(deftest branch-flip-mh-posterior
  (testing "MH selecting :branch converges to Bayes posterior p(branch=1|y)"
    (let [p 0.5 mu1 2.0 s1 1.0 mu0 -2.0 s0 1.0
          yv 0.7
          L1 (normal-pdf yv mu1 s1)
          L0 (normal-pdf yv mu0 s0)
          post (/ (* p L1) (+ (* p L1) (* (- 1 p) L0)))   ; closed-form oracle
          gf  (force-handler branch-shared-y)
          args [p mu1 s1 mu0 s0]
          t0  (gen-trace branch-shared-y args
                         (cm/choicemap :branch (mx/scalar 0.0) :y (mx/scalar yv))
                         (rng/fresh-key 1700))
          n   4000
          ones (loop [i 0, t t0, acc 0]
                 (if (>= i n)
                   acc
                   (let [{:keys [trace weight]} (p/regenerate
                                                  (dyn/with-key gf (rng/fresh-key (+ 2000 i)))
                                                  t (sel/select :branch))
                         w (mx/item weight)
                         [_ uk] (rng/split (rng/fresh-key (+ 30000 i)))
                         u (mx/item (rng/uniform uk []))
                         accept? (< (js/Math.log u) w)
                         t' (if accept? trace t)
                         b  (cval (:choices t') :branch)]
                     (recur (inc i) t' (+ acc (if (pos? b) 1 0))))))
          emp (/ ones n)]
      (is (h/close? post emp 0.03)
          (str "empirical p(branch=1)=" emp " vs oracle " post)))))

;; ---------------------------------------------------------------------------
;; 5. Prior-only flip (DIFFERENT addresses) — no throw, W = 0
;; ---------------------------------------------------------------------------

(deftest prior-only-flip-structure-change
  (testing "selecting :branch flips arms (xa<->xb) without throwing; W = 0"
    (let [n-flips (atom 0)]
      (dotimes [i 40]
        (let [;; old trace in arm 0 (has :xb)
              t (gen-trace prior-flip [] (cm/choicemap :branch (mx/scalar 0.0) :xb (mx/scalar 4.2))
                           (rng/fresh-key (+ 1800 i)))
              {:keys [trace weight]} (p/regenerate
                                       (dyn/with-key (force-handler prior-flip) (rng/fresh-key (+ 1900 i)))
                                       t (sel/select :branch))
              b-new (cval (:choices trace) :branch)]
          (when (pos? b-new) (swap! n-flips inc))
          (is (h/close? 0.0 (mx/item weight) 1e-4)
              "prior-only flip weight is 0 (fresh/removed cancel)")
          ;; structure invariant: exactly one of xa/xb present, matching branch
          (is (if (pos? b-new)
                (and (has-addr? (:choices trace) :xa) (not (has-addr? (:choices trace) :xb)))
                (and (has-addr? (:choices trace) :xb) (not (has-addr? (:choices trace) :xa))))
              "exactly the active arm's latent is present")))
      (is (pos? @n-flips) "at least one actual branch flip occurred (structure changed)"))))

;; ---------------------------------------------------------------------------
;; 6. Structure change WITH observation — weight oracle + MH posterior
;; ---------------------------------------------------------------------------

(deftest struct-change-obs-weight-oracle
  (testing "flip swaps latent address AND obs-mean; W depends only on retained :obs"
    (let [obsv -0.3]
      (dotimes [i 25]
        (let [t (gen-trace struct-obs [obsv]
                           (cm/choicemap :branch (mx/scalar 0.0) :yb (mx/scalar 5.1) :obs (mx/scalar obsv))
                           (rng/fresh-key (+ 2100 i)))
              {:keys [trace weight]} (p/regenerate
                                       (dyn/with-key (force-handler struct-obs) (rng/fresh-key (+ 2200 i)))
                                       t (sel/select :branch))
              b-new (cval (:choices trace) :branch)
              oracle (if (pos? b-new)
                       (- (h/gaussian-lp obsv 2.0 1.0) (h/gaussian-lp obsv -2.0 1.0))  ; 0->1
                       0.0)]
          ;; fresh latent must be present, removed latent gone
          (is (if (pos? b-new)
                (and (has-addr? (:choices trace) :ya) (not (has-addr? (:choices trace) :yb)))
                (and (has-addr? (:choices trace) :yb) (not (has-addr? (:choices trace) :ya)))
                )
              "active-arm latent present, other removed")
          (is (h/close? oracle (mx/item weight) 1e-4)
              "structure-change weight depends only on retained :obs"))))))

(deftest struct-change-mh-posterior
  (testing "MH over :branch with structure change converges to Bayes posterior"
    (let [obsv 0.2
          L1 (normal-pdf obsv 2.0 1.0)
          L0 (normal-pdf obsv -2.0 1.0)
          post (/ (* 0.5 L1) (+ (* 0.5 L1) (* 0.5 L0)))
          gf  (force-handler struct-obs)
          t0  (gen-trace struct-obs [obsv]
                         (cm/choicemap :branch (mx/scalar 0.0) :yb (mx/scalar 5.0) :obs (mx/scalar obsv))
                         (rng/fresh-key 2300))
          n   4000
          ones (loop [i 0, t t0, acc 0]
                 (if (>= i n)
                   acc
                   (let [{:keys [trace weight]} (p/regenerate
                                                  (dyn/with-key gf (rng/fresh-key (+ 2500 i)))
                                                  t (sel/select :branch))
                         w (mx/item weight)
                         u (mx/item (rng/uniform (rng/fresh-key (+ 40000 i)) []))
                         t' (if (< (js/Math.log u) w) trace t)
                         b  (cval (:choices t') :branch)]
                     (recur (inc i) t' (+ acc (if (pos? b) 1 0))))))
          emp (/ ones n)]
      (is (h/close? post emp 0.03)
          (str "structure-change MH empirical p=" emp " vs oracle " post)))))

;; ---------------------------------------------------------------------------
;; 7. Reversibility — flipping back restores the address set
;; ---------------------------------------------------------------------------

(deftest structure-change-reversibility
  (testing "flip then flip back restores original branch address set"
    (let [t0 (gen-trace prior-flip [] (cm/choicemap :branch (mx/scalar 1.0) :xa (mx/scalar 0.1))
                        (rng/fresh-key 2600))
          ;; force a flip by regenerating until branch becomes 0
          flip-to-0 (loop [i 0]
                      (let [{:keys [trace]} (p/regenerate
                                              (dyn/with-key (force-handler prior-flip) (rng/fresh-key (+ 2700 i)))
                                              t0 (sel/select :branch))]
                        (if (or (not (pos? (cval (:choices trace) :branch))) (> i 50))
                          trace
                          (recur (inc i)))))]
      (is (has-addr? (:choices flip-to-0) :xb) "after flip to 0, :xb present")
      (is (not (has-addr? (:choices flip-to-0) :xa)) "after flip to 0, :xa gone"))))

;; ---------------------------------------------------------------------------
;; 8. Splice composition — regenerate through a spliced dependent child
;; ---------------------------------------------------------------------------

(deftest splice-regenerate-composition
  (testing "select child {a,b}: parent W = retained :y delta only (child contributes 0)"
    (let [sigma 0.7]
      (dotimes [i 15]
        (let [t (gen-trace parent-splice [sigma]
                           (cm/choicemap :child (cm/choicemap :a (mx/scalar 0.2) :b (mx/scalar 0.8))
                                         :y (mx/scalar 1.0))
                           (rng/fresh-key (+ 2800 i)))
              y-old (cval (:choices t) :y)
              b-old (mx/item (cm/get-choice (:choices t) [:child :b]))
              ;; select both child latents; :y (parent-direct) retained
              {:keys [trace weight]} (p/regenerate
                                       (dyn/with-key (force-handler parent-splice) (rng/fresh-key (+ 2900 i)))
                                       t (sel/from-paths [[:child :a] [:child :b]]))
              b-new (mx/item (cm/get-choice (:choices trace) [:child :b]))
              ;; retained :y depends on child :b via the parent dist N(b, sigma)
              oracle (- (h/gaussian-lp y-old b-new sigma) (h/gaussian-lp y-old b-old sigma))]
          (is (h/close? oracle (mx/item weight) 1e-4)
              "spliced regenerate weight = retained parent-direct :y delta"))))))

;; ---------------------------------------------------------------------------
;; 9. Batched: uniform flip coherent; divergent flip throws
;; ---------------------------------------------------------------------------

(deftest batched-regenerate-fast-eligibility-gate
  ;; The batched per-site convention is exact only for fast-eligible
  ;; selections; a dependent JOINT batched move would carry the yep2 residual,
  ;; and a structure-changing batched flip is ill-posed under shape-batching
  ;; (math-verifier §7). vregenerate rejects both loudly rather than silently
  ;; miscalibrate. The full batched retained-only path is genmlx-8xia.
  (testing "dependent joint batched regenerate is rejected (no silent yep2)"
    (let [vt (dyn/vsimulate dep-pair [] 8 (rng/fresh-key 3000))]
      (is (thrown? :default
            (dyn/vregenerate dep-pair vt (sel/from-paths [[:a] [:b]]) (rng/fresh-key 3100)))
          "dependent joint batched regenerate throws, not silently miscalibrated")))
  (testing "single-site batched regenerate is fast-eligible and succeeds"
    (let [vt (dyn/vsimulate dep-pair [] 8 (rng/fresh-key 3200))
          {:keys [vtrace]} (dyn/vregenerate dep-pair vt (sel/select :b) (rng/fresh-key 3300))]
      (is (some? vtrace) "single-site batched regenerate is allowed (fast-eligible)"))))

(cljs.test/run-tests)
