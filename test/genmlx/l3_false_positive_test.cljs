;; @tier medium
(ns genmlx.l3-false-positive-test
  "L3 analytical false positives (genmlx-b470).

   The analytical eliminator must NEVER fire on a model it does not actually
   handle — same 'silently wrong number' class as the fixed ke9i/lwhw scars:
   1. bilinear obs means pass joint-affinity per-pair (h probes to 0)
   2. affine deps accepted for families that drop the coefficient
   3. families without a runtime handler factory counted as eliminated
   4. constrained priors / partially-constrained obs silently marginalized
   plus kalman/ekf obs-handler guards and lg regenerate obs selection.

   Ground truth per family is an INDEPENDENT closed-form oracle computed
   host-side in this file (never the function under test); fallthrough cases
   are verified by same-key parity against the stripped handler path.

   Run: bun run --bun nbb test/genmlx/l3_false_positive_test.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx.random :as rng]
            [genmlx.method-selection :as ms]
            [genmlx.conjugacy :as conj]
            [genmlx.linear-gaussian :as lg]
            [genmlx.inference.kalman :as kal]
            [genmlx.inference.ekf :as ekf]
            [genmlx.inference.ekf-nd :as ekfnd])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Assertion helpers
;; ---------------------------------------------------------------------------

(def ^:dynamic *pass* (volatile! 0))
(def ^:dynamic *fail* (volatile! 0))

(defn assert-true [desc pred]
  (if pred
    (do (vswap! *pass* inc) (println (str "  PASS: " desc)))
    (do (vswap! *fail* inc) (println (str "  FAIL: " desc)))))

(defn assert-close [desc expected actual tol]
  (let [d (js/Math.abs (- expected actual))]
    (if (<= d tol)
      (do (vswap! *pass* inc)
          (println (str "  PASS: " desc " (" (.toFixed actual 6) " ~ " (.toFixed expected 6) ", |Δ|=" (.toExponential d 2) ")")))
      (do (vswap! *fail* inc)
          (println (str "  FAIL: " desc " (" (.toFixed actual 6) " vs " (.toFixed expected 6) ", |Δ|=" (.toExponential d 2) " > " tol ")"))))))

(def TOL 2e-4)

;; ---------------------------------------------------------------------------
;; Host-side oracles (independent of genmlx math)
;; ---------------------------------------------------------------------------

(def LOG2PI (js/Math.log (* 2 js/Math.PI)))

(defn norm-lpdf [x mu variance]
  (* -0.5 (+ LOG2PI (js/Math.log variance)
             (/ (* (- x mu) (- x mu)) variance))))

(defn mvn2-lpdf [[x0 x1] [m0 m1] [[a b] [c d]]]
  (let [det (- (* a d) (* b c))
        dx0 (- x0 m0) dx1 (- x1 m1)
        q (/ (+ (* dx0 (- (* d dx0) (* b dx1)))
                (* dx1 (- (* a dx1) (* c dx0))))
             det)]
    (* -0.5 (+ (* 2 LOG2PI) (js/Math.log det) q))))

(defn iid-shared-mu-lpdf
  "log N(y; m0*1, s2*I + t2*11') via the matrix-determinant lemma
   (verified against numpy in the ke9i referee work)."
  [ys m0 t2 s2]
  (let [T (count ys)
        ds (mapv #(- % m0) ys)
        sum-d (reduce + ds)
        sum-d2 (reduce + (map #(* % %) ds))
        denom (+ s2 (* T t2))
        logdet (+ (* (dec T) (js/Math.log s2)) (js/Math.log denom))
        quad (/ (- sum-d2 (* (/ t2 denom) (* sum-d sum-d))) s2)]
    (* -0.5 (+ (* T LOG2PI) logdet quad))))

;; ---------------------------------------------------------------------------
;; Utilities
;; ---------------------------------------------------------------------------

(defn strip-l3
  "Force the pure handler path: remove analytical AND compiled schema keys."
  [model]
  (dyn/->DynamicGF (:body-fn model) (:source model)
                   (dissoc (:schema model)
                           :auto-handlers :auto-regenerate-transition
                           :auto-regenerate-handlers :analytical-plan
                           :linear-gaussian-blocks :conjugate-pairs
                           :has-conjugate?
                           :compiled-simulate :compiled-generate
                           :compiled-update :compiled-assess
                           :compiled-project :compiled-regenerate
                           :compiled-prefix :compiled-prefix-generate
                           :compiled-prefix-update :compiled-prefix-regenerate
                           :compiled-prefix-assess :compiled-prefix-project)))

(defn gen-weight [model constraints key]
  (mx/item (:weight (p/generate (dyn/with-key model key) [] constraints))))

(defn gen-trace [model constraints key]
  (:trace (p/generate (dyn/with-key model key) [] constraints)))

(defn marginal-trace? [trace]
  (= :marginal (:genmlx.trace/score-type (meta trace))))

(defn cmv [m] (reduce-kv (fn [c k v] (cm/set-value c k (mx/scalar v))) cm/EMPTY m))

(defn choice-val [trace addr]
  (mx/item (cm/get-value (cm/get-submap (:choices trace) addr))))

;; ===========================================================================
;; SECTION 1 — Bilinear obs mean: block must decline, not fabricate h=0
;; ===========================================================================

(println "\n== Section 1: bilinear mean declines ==")

(def bilinear-model
  (gen []
    (let [a (trace :a (dist/gaussian 1.0 1.0))
          b (trace :b (dist/gaussian 2.0 1.0))]
      (trace :y (dist/gaussian (mx/multiply a b) 1.0))
      a)))

(let [schema (:schema bilinear-model)]
  (assert-true "bilinear: no linear-Gaussian block claimed (static interaction check)"
               (empty? (:linear-gaussian-blocks schema)))
  (assert-true "bilinear: no analytical handlers installed"
               (empty? (:auto-handlers schema))))

(let [k (rng/fresh-key 42)
      obs (cmv {:y 2.0})
      r (p/generate (dyn/with-key bilinear-model k) [] obs)
      w (mx/item (:weight r))
      tr (:trace r)
      a (choice-val tr :a)
      b (choice-val tr :b)
      w-handler (gen-weight (strip-l3 bilinear-model) obs k)
      wrong-marginal (norm-lpdf 2.0 0.0 1.0)]   ; what h=[0,0],c=0 would produce
  (assert-true "bilinear: trace NOT labeled :marginal" (not (marginal-trace? tr)))
  (assert-close "bilinear: weight = joint p(y|a,b) from trace values"
                (norm-lpdf 2.0 (* a b) 1.0) w TOL)
  (assert-close "bilinear: weight matches handler path (same key)" w-handler w TOL)
  (assert-true "bilinear: weight is NOT the fabricated h=0 marginal"
               (> (js/Math.abs (- w wrong-marginal)) 1e-3)))

;; Runtime probe backstop, exercised directly through make-lg-handlers with a
;; synthetic block whose static detection hypothetically passed.
(println "\n-- runtime joint-affinity probe (backstop) --")
(let [mk-state (fn [] {:model-args [] :constraints (cmv {:y 2.0})
                       :choices cm/EMPTY :score (mx/scalar 0.0)
                       :weight (mx/scalar 0.0)})
      fake-dist {:params {:mu (mx/scalar 0.0) :sigma (mx/scalar 1.0)}}
      bilinear-block {:id [:a :b] :latents [:a :b] :p 2
                      :latent-index {:a 0 :b 1}
                      :obs [{:addr :y
                             :mean-fn (fn [env _] (mx/multiply (get env :a) (get env :b)))
                             :sigma-fn (fn [_ _] (mx/scalar 1.0))}]
                      :obs-addrs [:y] :latent-addrs [:a :b]
                      :noise-latents #{} :all-addrs #{:a :b :y}}
      affine-block (assoc bilinear-block :obs
                          [{:addr :y
                            :mean-fn (fn [env _] (mx/add (get env :a)
                                                         (mx/multiply (mx/scalar 2.0) (get env :b))))
                            :sigma-fn (fn [_ _] (mx/scalar 1.0))}])
      h-bad (get (lg/make-lg-handlers bilinear-block) :a)
      h-good (get (lg/make-lg-handlers affine-block) :a)]
  (assert-true "probe: bilinear mean-fn → latent handler falls through"
               (nil? (h-bad (mk-state) :a fake-dist)))
  (assert-true "probe: affine mean-fn → latent handler proceeds"
               (some? (h-good (mk-state) :a fake-dist))))

;; ===========================================================================
;; SECTION 2 — Per-family closed-form oracle (still-eliminated cases stay exact)
;; ===========================================================================

(println "\n== Section 2: per-family marginal oracle ==")

;; normal-normal, two obs (sequential updates must equal the joint marginal)
(def nn-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 1.0 2.0))]
      (trace :y1 (dist/gaussian mu 0.5))
      (trace :y2 (dist/gaussian mu 0.5))
      mu)))

(let [k (rng/fresh-key 1)
      obs (cmv {:y1 2.5 :y2 0.5})
      r (p/generate (dyn/with-key nn-model k) [] obs)
      oracle (mvn2-lpdf [2.5 0.5] [1.0 1.0] [[4.25 4.0] [4.0 4.25]])]
  (assert-true "NN: trace labeled :marginal" (marginal-trace? (:trace r)))
  (assert-close "NN: generate weight = joint marginal oracle"
                oracle (mx/item (:weight r)) TOL)
  (assert-close "NN: assess weight = joint marginal oracle"
                oracle (mx/item (:weight (p/assess (dyn/with-key nn-model k) [] obs))) TOL))

;; normal-iid-normal
(def nn-iid-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 0.0 1.0))]
      (trace :x (dist/iid-gaussian mu 1.0 3))
      mu)))

(let [k (rng/fresh-key 2)
      obs (cm/set-value cm/EMPTY :x (mx/array [0.5 -0.3 1.2]))
      r (p/generate (dyn/with-key nn-iid-model k) [] obs)
      oracle (iid-shared-mu-lpdf [0.5 -0.3 1.2] 0.0 1.0 1.0)]
  (assert-true "NN-iid: trace labeled :marginal" (marginal-trace? (:trace r)))
  (assert-close "NN-iid: generate weight = shared-mu joint marginal"
                oracle (mx/item (:weight r)) TOL))

;; beta-bernoulli, two obs (Polya urn)
(def bb-model
  (gen []
    (let [th (trace :th (dist/beta-dist 2.0 3.0))]
      (trace :b1 (dist/bernoulli th))
      (trace :b2 (dist/bernoulli th))
      th)))

(let [k (rng/fresh-key 3)
      obs (cmv {:b1 1.0 :b2 0.0})
      r (p/generate (dyn/with-key bb-model k) [] obs)
      oracle (js/Math.log (* (/ 2.0 5.0) (/ 3.0 6.0)))]
  (assert-close "BB: generate weight = Polya-urn marginal"
                oracle (mx/item (:weight r)) TOL))

;; gamma-poisson (negative-binomial marginal, shape=2 rate=1 k=3 → 0.125)
(def gp-model
  (gen []
    (let [lam (trace :lam (dist/gamma-dist 2.0 1.0))]
      (trace :k (dist/poisson lam))
      lam)))

(let [k (rng/fresh-key 4)
      obs (cmv {:k 3.0})
      r (p/generate (dyn/with-key gp-model k) [] obs)]
  (assert-close "GP: generate weight = negative-binomial marginal"
                (js/Math.log 0.125) (mx/item (:weight r)) TOL))

;; gamma-exponential (Lomax marginal)
(def ge-model
  (gen []
    (let [lam (trace :lam (dist/gamma-dist 2.0 1.0))]
      (trace :x (dist/exponential lam))
      lam)))

(let [k (rng/fresh-key 5)
      obs (cmv {:x 0.7})
      r (p/generate (dyn/with-key ge-model k) [] obs)
      oracle (- (js/Math.log 2.0) (* 3.0 (js/Math.log 1.7)))]
  (assert-close "GE: generate weight = Lomax marginal"
                oracle (mx/item (:weight r)) TOL))

;; mvn-normal (2-d, diagonal: marginal N(m0, S0+R))
(def mvn-model
  (gen []
    (let [mu (trace :mu (dist/multivariate-normal
                         (mx/array [0.0 0.0])
                         (mx/array [[1.0 0.0] [0.0 1.0]])))]
      (trace :y (dist/multivariate-normal mu (mx/array [[0.5 0.0] [0.0 0.5]])))
      mu)))

(let [k (rng/fresh-key 6)
      obs (cm/set-value cm/EMPTY :y (mx/array [1.0 -1.0]))
      r (p/generate (dyn/with-key mvn-model k) [] obs)
      oracle (mvn2-lpdf [1.0 -1.0] [0.0 0.0] [[1.5 0.0] [0.0 1.5]])]
  (assert-close "MVN: generate weight = N(m0, S0+R) marginal"
                oracle (mx/item (:weight r)) TOL))

;; Kalman chain (full obs): joint MVN oracle over (y0, y1)
(def chain-model
  (gen []
    (let [z0 (trace :z0 (dist/gaussian 0.0 1.0))
          z1 (trace :z1 (dist/gaussian z0 1.0))]
      (trace :y0 (dist/gaussian z0 0.5))
      (trace :y1 (dist/gaussian z1 0.5))
      z1)))

(let [k (rng/fresh-key 7)
      obs (cmv {:y0 0.8 :y1 -0.4})
      r (p/generate (dyn/with-key chain-model k) [] obs)
      oracle (mvn2-lpdf [0.8 -0.4] [0.0 0.0] [[1.25 1.0] [1.0 2.25]])]
  (assert-true "chain: trace labeled :marginal" (marginal-trace? (:trace r)))
  (assert-close "chain: generate weight = joint MVN oracle"
                oracle (mx/item (:weight r)) TOL))

;; Linear-Gaussian block (full obs): joint MVN oracle, regression for the gates
(def linreg2
  (gen []
    (let [s (trace :s (dist/gaussian 0.0 1.0))
          i (trace :i (dist/gaussian 0.0 1.0))]
      (trace :y0 (dist/gaussian (mx/add (mx/multiply s (mx/scalar 1.0)) i) 0.5))
      (trace :y1 (dist/gaussian (mx/add (mx/multiply s (mx/scalar 2.0)) i) 0.5))
      s)))

(let [k (rng/fresh-key 8)
      obs (cmv {:y0 0.6 :y1 1.4})
      r (p/generate (dyn/with-key linreg2 k) [] obs)
      oracle (mvn2-lpdf [0.6 1.4] [0.0 0.0] [[2.25 3.0] [3.0 5.25]])]
  (assert-true "LG block: trace labeled :marginal" (marginal-trace? (:trace r)))
  (assert-close "LG block: generate weight = joint MVN oracle"
                oracle (mx/item (:weight r)) TOL))

;; ===========================================================================
;; SECTION 3 — Affine deps on non-NN families are declined
;; ===========================================================================

(println "\n== Section 3: non-NN affine deps decline ==")

(def gp-affine-model
  (gen []
    (let [lam (trace :lam (dist/gamma-dist 2.0 1.0))]
      (trace :k (dist/poisson (mx/multiply (mx/scalar 2.0) lam)))
      lam)))

(let [schema (:schema gp-affine-model)]
  (assert-true "GP affine: no conjugate pair detected"
               (empty? (:conjugate-pairs schema)))
  (assert-true "GP affine: no analytical handlers"
               (empty? (:auto-handlers schema))))

(let [k (rng/fresh-key 9)
      obs (cmv {:k 3.0})
      w (gen-weight gp-affine-model obs k)
      w-handler (gen-weight (strip-l3 gp-affine-model) obs k)
      ;; what the scalar path would produce by DROPPING the coefficient
      wrong (js/Math.log 0.125)]
  (assert-close "GP affine: weight matches handler path (same key)" w-handler w TOL)
  (assert-true "GP affine: weight is NOT the coefficient-dropping marginal"
               (> (js/Math.abs (- w wrong)) 1e-3)))

;; NN affine pairs must STILL be detected (kalman/LG paths handle them)
(let [schema (:schema linreg2)]
  (assert-true "NN affine: pairs still detected for normal-normal"
               (pos? (count (:conjugate-pairs schema)))))

;; ===========================================================================
;; SECTION 4 — Dirichlet–Categorical: bare logits NOT conjugate; log-link IS
;; (genmlx-cf0d). GenMLX's categorical is logit-parameterized, so ONLY the log
;; link (dist/categorical (mx/log theta)) equals Categorical(theta) over the
;; simplex and is conjugate to a Dirichlet prior. Bare (dist/categorical theta)
;; is raw logit space — its marginal E[softmax(theta)_k] has no closed form, so
;; it must be DECLINED (the false-positive guard this file is about). The
;; log-link form is the genuine pair and routes to exact.
;; ===========================================================================

(println "\n== Section 4: dirichlet-categorical — bare declined, log-link exact ==")

;; (a) BARE logits — NOT conjugate to Dirichlet; detection must NOT fire.
(def dc-bare-model
  (gen []
    (let [th (trace :th (dist/dirichlet (mx/array [1.0 1.0 1.0])))]
      (trace :c (dist/categorical th))
      th)))

(let [schema (:schema dc-bare-model)
      sel-result (ms/select-method dc-bare-model (cmv {:c 1.0}))]
  (assert-true "DC bare: NO conjugate pair (logit space is not conjugate)"
               (empty? (:conjugate-pairs schema)))
  (assert-true "DC bare: no analytical handlers installed"
               (empty? (:auto-handlers schema)))
  (assert-true "DC bare: method selection does NOT claim :exact"
               (not= :exact (:method sel-result))))

(let [k (rng/fresh-key 10)
      obs (cmv {:c 1.0})
      w (gen-weight dc-bare-model obs k)
      w-handler (gen-weight (strip-l3 dc-bare-model) obs k)
      tr (gen-trace dc-bare-model obs k)]
  (assert-true "DC bare: trace NOT labeled :marginal" (not (marginal-trace? tr)))
  (assert-close "DC bare: weight matches handler path (same key)" w-handler w TOL))

;; (b) LOG-LINK — the genuine Dirichlet–Categorical: fires, eliminates, exact.
(def dc-loglink-model
  (gen []
    (let [th (trace :th (dist/dirichlet (mx/array [1.0 1.0 1.0])))]
      (trace :c (dist/categorical (mx/log th)))
      th)))

(let [schema (:schema dc-loglink-model)
      eliminated (get-in schema [:analytical-plan :rewrite-result :eliminated])
      sel-result (ms/select-method dc-loglink-model (cmv {:c 1.0}))]
  (assert-true "DC log-link: conjugate pair detected"
               (pos? (count (:conjugate-pairs schema))))
  (assert-true "DC log-link: family is :dirichlet-categorical"
               (= :dirichlet-categorical (:family (first (:conjugate-pairs schema)))))
  (assert-true "DC log-link: prior :th eliminated"
               (contains? (or eliminated #{}) :th))
  (assert-true "DC log-link: analytical handlers installed for :th and :c"
               (and (contains? (:auto-handlers schema) :th)
                    (contains? (:auto-handlers schema) :c)))
  (assert-true "DC log-link: method selection claims :exact"
               (= :exact (:method sel-result))))

;; Single-obs marginal: P(c=1) = alpha_1 / sum(alpha) = 1/3  ->  log(1/3).
;; Independent closed-form oracle (host-side), exact path matches to float floor.
(let [k (rng/fresh-key 10)
      obs (cmv {:c 1.0})
      w (gen-weight dc-loglink-model obs k)
      tr (gen-trace dc-loglink-model obs k)]
  (assert-true "DC log-link: trace labeled :marginal" (marginal-trace? tr))
  (assert-close "DC log-link: exact weight == closed form log(1/3)"
                (js/Math.log (/ 1.0 3.0)) w TOL))

;; ===========================================================================
;; SECTION 5 — Constraint checks: constrained prior, partial obs
;; ===========================================================================

(println "\n== Section 5: constraint gates ==")

;; Two independent NN pairs; constrain pair-1's PRIOR and both obs.
(def two-pair-model
  (gen []
    (let [m1 (trace :m1 (dist/gaussian 0.0 1.0))
          m2 (trace :m2 (dist/gaussian 0.0 2.0))]
      (trace :y1 (dist/gaussian m1 0.5))
      (trace :y2 (dist/gaussian m2 0.5))
      m1)))

(let [k (rng/fresh-key 11)
      obs (cmv {:m1 0.7 :y1 1.0 :y2 -0.5})
      r (p/generate (dyn/with-key two-pair-model k) [] obs)
      w (mx/item (:weight r))
      tr (:trace r)
      ;; pair 1 joint (prior constrained) + pair 2 marginal
      oracle (+ (norm-lpdf 0.7 0.0 1.0)
                (norm-lpdf 1.0 0.7 0.25)
                (norm-lpdf -0.5 0.0 4.25))]
  (assert-close "constrained prior: weight = joint(pair1) + marginal(pair2)"
                oracle w TOL)
  (assert-close "constrained prior: :m1 keeps its constrained value (not posterior mean)"
                0.7 (choice-val tr :m1) 1e-6))

;; One prior, two obs, only one constrained → pair must decline entirely.
(let [k (rng/fresh-key 12)
      obs (cmv {:y1 1.0})
      w (gen-weight nn-model obs k)
      w-handler (gen-weight (strip-l3 nn-model) obs k)
      tr (gen-trace nn-model obs k)]
  (assert-true "partial obs: trace NOT labeled :marginal" (not (marginal-trace? tr)))
  (assert-close "partial obs: weight matches handler path (same key)" w-handler w TOL)
  (assert-close "partial obs: weight = p(y1 | sampled mu) from trace"
                (norm-lpdf 1.0 (choice-val tr :mu) 0.25) w TOL))

;; Kalman chain with partial obs → chain declines.
(let [k (rng/fresh-key 13)
      obs (cmv {:y0 0.8})
      w (gen-weight chain-model obs k)
      w-handler (gen-weight (strip-l3 chain-model) obs k)
      tr (gen-trace chain-model obs k)]
  (assert-true "chain partial obs: trace NOT labeled :marginal" (not (marginal-trace? tr)))
  (assert-close "chain partial obs: weight matches handler path (same key)" w-handler w TOL))

;; Kalman chain with a constrained LATENT → chain declines.
(let [k (rng/fresh-key 14)
      obs (cmv {:z0 0.3 :y0 0.8 :y1 -0.4})
      w (gen-weight chain-model obs k)
      w-handler (gen-weight (strip-l3 chain-model) obs k)
      tr (gen-trace chain-model obs k)]
  (assert-true "chain constrained latent: NOT :marginal" (not (marginal-trace? tr)))
  (assert-close "chain constrained latent: weight matches handler (same key)" w-handler w TOL)
  (assert-close "chain constrained latent: :z0 keeps constrained value"
                0.3 (choice-val tr :z0) 1e-6))

;; LG block with partial obs → block declines.
(let [k (rng/fresh-key 15)
      obs (cmv {:y0 0.6})
      w (gen-weight linreg2 obs k)
      w-handler (gen-weight (strip-l3 linreg2) obs k)
      tr (gen-trace linreg2 obs k)]
  (assert-true "LG partial obs: NOT :marginal" (not (marginal-trace? tr)))
  (assert-close "LG partial obs: weight matches handler (same key)" w-handler w TOL))

;; LG block with a constrained latent → block declines.
(let [k (rng/fresh-key 16)
      obs (cmv {:s 0.5 :y0 0.6 :y1 1.4})
      w (gen-weight linreg2 obs k)
      w-handler (gen-weight (strip-l3 linreg2) obs k)
      tr (gen-trace linreg2 obs k)]
  (assert-true "LG constrained latent: NOT :marginal" (not (marginal-trace? tr)))
  (assert-close "LG constrained latent: weight matches handler (same key)" w-handler w TOL)
  (assert-close "LG constrained latent: :s keeps constrained value"
                0.5 (choice-val tr :s) 1e-6))

;; ===========================================================================
;; SECTION 6 — Multi-parent obs filter (unit)
;; ===========================================================================

(println "\n== Section 6: multi-parent pair filter ==")

(let [pairs [{:prior-addr :a :obs-addr :y :family :gamma-poisson}
             {:prior-addr :b :obs-addr :y :family :gamma-poisson}
             {:prior-addr :c :obs-addr :z :family :gamma-poisson}]
      kept (conj/drop-multi-parent-pairs pairs)]
  (assert-true "multi-parent: both pairs claiming :y dropped"
               (= [:z] (mapv :obs-addr kept)))
  (assert-true "multi-parent: single-parent pair kept"
               (= [:c] (mapv :prior-addr kept)))
  (assert-true "multi-parent: no-op when all single-parent"
               (= 1 (count (conj/drop-multi-parent-pairs
                            [{:prior-addr :c :obs-addr :z}])))))

;; ===========================================================================
;; SECTION 7 — Kalman/EKF obs-handler guards (unit)
;; ===========================================================================

(println "\n== Section 7: kalman/ekf obs guards ==")

(let [h (:kalman-obs (kal/make-kalman-dispatch :z))
      params {:base-mean (mx/zeros [1]) :loading (mx/ones [1])
              :noise-std (mx/ones [1]) :mask (mx/ones [1])}
      fake-dist {:params params}
      belief {:mean (mx/zeros [1]) :var (mx/ones [1])}
      with-obs (cm/set-value cm/EMPTY :o (mx/ones [1]))]
  (assert-true "kalman obs: nil belief → fall through"
               (nil? (h {:constraints with-obs :kalman-n 1} :o fake-dist)))
  (assert-true "kalman obs: unconstrained → fall through"
               (nil? (h {:constraints cm/EMPTY :kalman-belief belief :kalman-n 1}
                        :o fake-dist)))
  (assert-true "kalman obs: belief + constraint → handled"
               (some? (h {:constraints with-obs :kalman-belief belief :kalman-n 1}
                         :o fake-dist))))

(let [h (:ekf-obs (ekf/make-ekf-dispatch :z))
      fake-dist {:params {:obs-fn identity :noise-std (mx/ones [1]) :mask (mx/ones [1])}}
      with-obs (cm/set-value cm/EMPTY :o (mx/ones [1]))]
  (assert-true "ekf obs: nil belief → fall through"
               (nil? (h {:constraints with-obs :ekf-n 1} :o fake-dist)))
  (assert-true "ekf obs: unconstrained → fall through"
               (nil? (h {:constraints cm/EMPTY
                         :ekf-belief {:mean (mx/zeros [1]) :var (mx/ones [1])}
                         :ekf-n 1}
                        :o fake-dist))))

(let [h (:ekf-nd-obs (ekfnd/make-multi-ekf-dispatch [:z0]))
      fake-dist {:params {:obs-fn identity :noise-std (mx/ones [1]) :mask (mx/ones [1])}}
      with-obs (cm/set-value cm/EMPTY :o (mx/ones [1]))]
  (assert-true "ekf-nd obs: nil means/covs → fall through"
               (nil? (h {:constraints with-obs :ekf-nd-n 1} :o fake-dist))))

;; ===========================================================================
;; SECTION 8 — LG regenerate: selected obs re-opens the block
;; ===========================================================================

(println "\n== Section 8: lg regenerate obs selection ==")

(let [k (rng/fresh-key 17)
      obs (cmv {:y0 0.6 :y1 1.4})
      tr (gen-trace linreg2 obs k)
      _ (assert-true "regen setup: generate trace is :marginal" (marginal-trace? tr))
      r (p/regenerate (dyn/with-key linreg2 (rng/fresh-key 18)) tr (sel/select :y0))
      new-tr (:trace r)]
  (assert-true "regen obs-selected: returns a trace" (some? new-tr))
  (assert-true "regen obs-selected: weight finite"
               (js/isFinite (mx/item (:weight r))))
  (assert-true "regen obs-selected: y0 actually resampled"
               (> (js/Math.abs (- (choice-val new-tr :y0) 0.6)) 1e-6))
  (assert-true "regen obs-selected: re-opened trace NOT labeled :marginal"
               (not (marginal-trace? new-tr))))

;; Unselected regenerate on a marginal trace still works (no over-decline).
(let [k (rng/fresh-key 19)
      obs (cmv {:y0 0.6 :y1 1.4})
      tr (gen-trace linreg2 obs k)
      r (p/regenerate (dyn/with-key linreg2 (rng/fresh-key 20)) tr (sel/select :none))]
  (assert-true "regen no-block-selection: stays :marginal"
               (marginal-trace? (:trace r))))

;; ===========================================================================
;; SECTION N — latent-dependent OBS NOISE: conjugacy must decline (genmlx-5ytq)
;; ===========================================================================
;;
;; conjugacy/classify-dependency inspected ONLY the natural-parameter argument
;; (:natural-param-idx, 0 for all seven families). When the eliminated latent
;; ALSO appeared in the observation's noise argument, the pair was still
;; detected and the closed form evaluated with sigma frozen at a point
;; estimate — a wrong marginal likelihood published as :exact.
;;
;; Measured before the guard: weight -1.51551 (exactly log N(3; 2, 2), i.e.
;; sigma pinned at the prior mean mu=2) against a true marginal of -1.79917
;; (2e6-point trapezoid over the exact 1-D integral). 0.284 nats, tagged
;; :marginal. The ORACLE below is that same independent quadrature.

(println "\n== Section N: latent-dependent obs noise declines ==")

(def cv-noise-model
  ;; constant-coefficient-of-variation likelihood: sigma = 0.5 * mu
  (gen []
    (let [mu (trace :mu (dist/gaussian 2.0 1.0))]
      (trace :y (dist/gaussian mu (mx/multiply 0.5 mu)))
      mu)))

(assert-true "cv-noise: no conjugate pair detected (latent in the sigma arg)"
             (empty? (conj/detect-conjugate-pairs (:schema cv-noise-model))))

(let [k (rng/fresh-key 41)
      tr (gen-trace cv-noise-model (cmv {:y 3.0}) k)]
  (assert-true "cv-noise: trace is :joint, not :marginal"
               (not (marginal-trace? tr))))

;; Same-key parity against the stripped handler path: the analytical route must
;; not merely be relabelled, it must be GONE.
(let [k (rng/fresh-key 42)]
  (assert-close "cv-noise: weight equals the stripped handler path"
                (gen-weight (strip-l3 cv-noise-model) (cmv {:y 3.0}) k)
                (gen-weight cv-noise-model (cmv {:y 3.0}) k)
                1e-6))

;; The independent oracle: whatever we return, it must NOT be the frozen-sigma
;; closed form (-1.51551), which is what the defect produced.
(let [frozen-sigma-closed-form (norm-lpdf 3.0 2.0 (* 2.0 2.0))]
  (assert-true "cv-noise: the frozen-sigma closed form is no longer returned"
               (> (Math/abs (- (gen-weight cv-noise-model (cmv {:y 3.0}) (rng/fresh-key 43))
                               frozen-sigma-closed-form))
                  1e-4)))

;; Control: the SAME family with a constant sigma must still be eliminated —
;; this guard is false-negatives-only and must not cost a real conjugate pair.
(def const-noise-model
  (gen []
    (let [mu (trace :mu (dist/gaussian 2.0 1.0))]
      (trace :y (dist/gaussian mu 0.5))
      mu)))

(assert-true "control: constant sigma still detects a conjugate pair"
             (seq (conj/detect-conjugate-pairs (:schema const-noise-model))))

(let [k (rng/fresh-key 44)
      tr (gen-trace const-noise-model (cmv {:y 3.0}) k)]
  (assert-true "control: constant sigma still yields a :marginal trace"
               (marginal-trace? tr))
  ;; normal-normal marginal: y ~ N(mu0, sigma0^2 + sigma^2)
  (assert-close "control: marginal matches the closed form"
                (norm-lpdf 3.0 2.0 (+ (* 1.0 1.0) (* 0.5 0.5)))
                (gen-weight const-noise-model (cmv {:y 3.0}) k)
                1e-4))

;; ===========================================================================
;; SECTION O — joint affinity probed in >1 DIRECTION (genmlx-su6q)
;; ===========================================================================
;;
;; probe-jointly-affine? validated a p-DIMENSIONAL affinity claim with two probe
;; points that BOTH lay on the diagonal ray s·(1,…,1). A mean that is nonlinear
;; OFF-diagonal but linear ALONG that ray was accepted, and the h recovered by
;; 0/1 coordinate probing was then published as an exact design row.
;;
;; Two adversaries, each a POSITIVE oracle for one of the added directions:
;;   a² − b²    quadratic terms cancel at every point of the diagonal
;;              (the bean's example) → needs the alternating direction
;;   a³b − ab³  = ab(a−b)(a+b): zero at every (s, s) AND every (s, −s)
;;              → needs the distinct-magnitude pseudo-random direction
;; Both are first shown to SATISFY the old diagonal-only test, so the section
;; cannot pass by accident.

(println "\n== Section O: off-diagonal nonlinearity declines ==")

(def ^:private sq  (fn [x] (mx/multiply x x)))
(def ^:private sc  mx/scalar)

;; mean forms, as raw (env, args) fns — the same level Section 1 probes at.
(def ^:private mean-diff-sq
  (fn [env _] (mx/subtract (sq (get env :a)) (sq (get env :b)))))
(def ^:private mean-quartic
  (fn [env _] (let [a (get env :a) b (get env :b)]
                (mx/subtract (mx/multiply (sq a) (mx/multiply a b))
                             (mx/multiply (mx/multiply a b) (sq b))))))
(def ^:private mean-affine2
  (fn [env _] (mx/add (mx/subtract (mx/multiply (sc 2.0) (get env :a))
                                   (mx/multiply (sc 3.0) (get env :b)))
                      (sc 1.0))))
(def ^:private mean-affine3
  (fn [env _] (-> (mx/multiply (sc 1.0) (get env :a))
                  (mx/add (mx/multiply (sc 2.0) (get env :b)))
                  (mx/subtract (mx/multiply (sc 0.5) (get env :c)))
                  (mx/add (sc 3.0)))))
(def ^:private mean-affine1
  (fn [env _] (mx/add (mx/multiply (sc 3.0) (get env :a)) (sc 1.0))))

;; The OLD oracle, recomputed here host-side: mean(s·1) ?= c + s·Σh, s ∈ {1,2}.
;; If an adversary failed THIS, the section would prove nothing about directions.
(defn- old-diagonal-probe-accepts? [mean-fn latents]
  (let [ev (fn [vals] (mx/item (mean-fn (zipmap latents (map sc vals)) nil)))
        p  (count latents)
        c  (ev (repeat p 0.0))
        hs (mapv (fn [i] (- (ev (map #(if (= % i) 1.0 0.0) (range p))) c)) (range p))
        sum-h (reduce + hs)]
    (every? (fn [s] (< (js/Math.abs (- (ev (repeat p s)) (+ c (* s sum-h)))) 1e-5))
            [1.0 2.0])))

(assert-true "a²−b²: the OLD diagonal-only probe accepts it (this is why the bug existed)"
             (old-diagonal-probe-accepts? mean-diff-sq [:a :b]))
(assert-true "a³b−ab³: accepted by the diagonal AND by the alternating ray"
             (and (old-diagonal-probe-accepts? mean-quartic [:a :b])
                  ;; alternating: mean(s,−s) = 0 for all s, and h·(1,−1) = 0 too
                  (let [ev (fn [a b] (mx/item (mean-quartic {:a (sc a) :b (sc b)} nil)))]
                    (and (< (js/Math.abs (ev 1.0 -1.0)) 1e-6)
                         (< (js/Math.abs (ev 2.0 -2.0)) 1e-6)))))

;; The decision under test: does the block's latent handler commit (some?) or
;; fall through to the ground-truth handler path (nil)?
(let [mk-state (fn [] {:model-args [] :constraints (cmv {:y 2.0})
                       :choices cm/EMPTY :score (mx/scalar 0.0)
                       :weight (mx/scalar 0.0)})
      fake-dist {:params {:mu (mx/scalar 0.0) :sigma (mx/scalar 1.0)}}
      mk-block (fn [latents mean-fn]
                 {:id latents :latents latents :p (count latents)
                  :latent-index (into {} (map-indexed (fn [i a] [a i])) latents)
                  :obs [{:addr :y :mean-fn mean-fn :sigma-fn (fn [_ _] (mx/scalar 1.0))}]
                  :obs-addrs [:y] :latent-addrs latents
                  :noise-latents #{} :all-addrs (into #{:y} latents)})
      eliminates? (fn [latents mean-fn]
                    (let [h (get (lg/make-lg-handlers (mk-block latents mean-fn))
                                 (first latents))]
                      (some? (h (mk-state) (first latents) fake-dist))))]

  (assert-true "probe: a²−b² (diagonal-only cancellation) is DECLINED"
               (not (eliminates? [:a :b] mean-diff-sq)))
  (assert-true "probe: a³b−ab³ (both ±1 rays cancel) is DECLINED"
               (not (eliminates? [:a :b] mean-quartic)))

  ;; False-negatives-only: genuinely affine means must still be eliminated.
  (assert-true "control: 2a−3b+1 still eliminates (p=2)"
               (eliminates? [:a :b] mean-affine2))
  (assert-true "control: a+2b−0.5c+3 still eliminates (p=3)"
               (eliminates? [:a :b :c] mean-affine3))
  (assert-true "control: 3a+1 still eliminates (p=1, alternating ray collapses)"
               (eliminates? [:a] mean-affine1))

  ;; The pseudo-random direction is SEEDED by the block id, never by entropy:
  ;; repeated probes of the same block must give the same verdict, or a block
  ;; would be eliminated in one particle and not the next.
  (assert-true "probe verdicts are deterministic across repeated probes"
               (and (= (repeatedly 3 #(eliminates? [:a :b] mean-quartic)) [false false false])
                    (= (repeatedly 3 #(eliminates? [:a :b] mean-affine2)) [true true true]))))

;; Whole-stack: the same mean written as a model. The static layer declines it
;; too (mx/square is not affine), so no block is claimed and the weight is the
;; plain joint — defense in depth, both layers must agree.
(def diff-sq-model
  (gen []
    (let [a (trace :a (dist/gaussian 1.0 1.0))
          b (trace :b (dist/gaussian 2.0 1.0))]
      (trace :y (dist/gaussian (mx/subtract (mx/multiply a a) (mx/multiply b b)) 1.0))
      a)))

(assert-true "a²−b² model: no linear-Gaussian block claimed"
             (empty? (:linear-gaussian-blocks (:schema diff-sq-model))))
(let [k (rng/fresh-key 51)
      obs (cmv {:y 2.0})
      r (p/generate (dyn/with-key diff-sq-model k) [] obs)
      tr (:trace r)
      a (choice-val tr :a) b (choice-val tr :b)]
  (assert-true "a²−b² model: trace NOT labeled :marginal" (not (marginal-trace? tr)))
  (assert-close "a²−b² model: weight = joint p(y|a,b) from the trace values"
                (norm-lpdf 2.0 (- (* a a) (* b b)) 1.0) (mx/item (:weight r)) TOL)
  (assert-close "a²−b² model: weight matches the stripped handler path (same key)"
                (gen-weight (strip-l3 diff-sq-model) obs k) (mx/item (:weight r)) TOL))

;; ===========================================================================
(println "\n==========================================")
(println (str "  l3-false-positive: " @*pass* " passed, " @*fail* " failed"))
(println "==========================================")
(when (pos? @*fail*) (js/process.exit 1))
