;; @tier medium
(ns genmlx.dirichlet-categorical-test
  "Dirichlet–Categorical exact analytical evidence (genmlx-cf0d).

   theta ~ Dirichlet(alpha), x ~ Categorical(theta). GenMLX's categorical is
   LOGIT-parameterized, so the conjugate obs is written with the log link
   (dist/categorical (mx/log theta)): softmax(log theta) = theta over the
   simplex, hence log p(x=k) = log theta_k. Bare (dist/categorical theta) is raw
   logit space and NOT conjugate — it is declined (covered in
   l3_false_positive_test Section 4).

   Ground truth here is THREE independent sources that must all agree:
     1. closed form    — host-side lgamma Dirichlet-multinomial sequence marginal
     2. exact          — the analytical handler (function under test)
     3. IS             — a self-written SHARED-theta importance sampler that
                         draws one theta per particle and scores all obs against
                         it (captures the obs correlation; NOT the ke9i trap of
                         summing independent per-obs marginals)
   Numbers were independently confirmed by the math-verifier (Lanczos lgamma).

   Run: bun run --bun nbb test/genmlx/dirichlet_categorical_test.cljs"
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

;; ---------------------------------------------------------------------------
;; Assertion harness
;; ---------------------------------------------------------------------------

(def ^:private pass (volatile! 0))
(def ^:private fail (volatile! 0))

(defn- assert-true [desc pred]
  (if pred
    (do (vswap! pass inc) (println (str "  PASS: " desc)))
    (do (vswap! fail inc) (println (str "  FAIL: " desc)))))

(defn- assert-close [desc expected actual tol]
  (let [ok (and (number? actual) (js/isFinite actual)
                (<= (js/Math.abs (- expected actual)) tol))]
    (if ok
      (do (vswap! pass inc)
          (println (str "  PASS: " desc " (" (.toFixed actual 6) " ~ " (.toFixed expected 6)
                        ", |Δ|=" (.toExponential (js/Math.abs (- expected actual)) 2) ")")))
      (do (vswap! fail inc)
          (println (str "  FAIL: " desc " expected " expected " got " actual))))))

;; ---------------------------------------------------------------------------
;; Independent host-side oracles (never the function under test)
;; ---------------------------------------------------------------------------

(defn- lgamma [x] (mx/item (mx/lgamma (mx/scalar (double x)))))

(defn- closed-form-log-marginal
  "Ordered Dirichlet-multinomial sequence marginal (NO multinomial coefficient):
   sum_k [lgamma(a_k + n_k) - lgamma(a_k)] + lgamma(sum a) - lgamma(sum a + n).
   alpha: vector of concentrations; obs: ordered seq of category indices."
  [alpha obs]
  (let [d (count alpha)
        counts (reduce (fn [c k] (update c k inc)) (vec (repeat d 0)) obs)
        n (count obs)
        sum-a (reduce + alpha)
        per-k (reduce + (map (fn [a nk] (- (lgamma (+ a nk)) (lgamma a))) alpha counts))]
    (+ per-k (lgamma sum-a) (- (lgamma (+ sum-a n))))))

(defn- counts-of [alpha obs]
  (let [d (count alpha)]
    (reduce (fn [c k] (update c k inc)) (vec (repeat d 0)) obs)))

(defn- is-log-marginal
  "Self-written SHARED-theta importance estimate of log P(obs). Draws N
   theta ~ Dirichlet(alpha) ([N,D]); each particle's weight is
   prod_i theta[k_i] = sum_k counts_k * log theta[:,k]; returns
   logsumexp(log w) - log N. One theta per particle scores ALL obs jointly,
   so the shared-theta correlation is preserved (this is the discriminating,
   non-circular oracle for the multi-obs case)."
  [alpha obs n key]
  (let [counts (mapv double (counts-of alpha obs))
        theta (dc/dist-sample-n (dist/dirichlet (mx/array (mapv double alpha))) key n) ;; [N,D]
        log-w (mx/sum (mx/multiply (mx/log theta) (mx/array counts)) [-1])             ;; [N]
        lse (mx/logsumexp log-w)]
    (mx/item (mx/subtract lse (mx/log (mx/scalar (double n)))))))

;; ---------------------------------------------------------------------------
;; Models + GFI helpers
;; ---------------------------------------------------------------------------

(defn- obs-cm [m]
  (reduce-kv (fn [c k v] (cm/set-value c k v)) cm/EMPTY m))

(defn- exact-weight [model obs key]
  (mx/item (:weight (p/generate (dyn/with-key model key) [] (obs-cm obs)))))

(defn- exact-trace [model obs key]
  (:trace (p/generate (dyn/with-key model key) [] (obs-cm obs))))

;; Single-obs models
(def ^:private m-123
  (gen [] (let [th (trace :th (dist/dirichlet (mx/array [1.0 2.0 3.0])))]
            (trace :x (dist/categorical (mx/log th))) th)))
(def ^:private m-22
  (gen [] (let [th (trace :th (dist/dirichlet (mx/array [2.0 2.0])))]
            (trace :x (dist/categorical (mx/log th))) th)))

;; Multi-obs (ordered) models — three / five iid obs sharing one theta.
(def ^:private m-D
  (gen [] (let [th (trace :th (dist/dirichlet (mx/array [1.0 1.0 1.0])))]
            (trace :x0 (dist/categorical (mx/log th)))
            (trace :x1 (dist/categorical (mx/log th)))
            (trace :x2 (dist/categorical (mx/log th)))
            th)))
(def ^:private m-E
  (gen [] (let [th (trace :th (dist/dirichlet (mx/array [2.0 1.0 1.0 1.0])))]
            (trace :x0 (dist/categorical (mx/log th)))
            (trace :x1 (dist/categorical (mx/log th)))
            (trace :x2 (dist/categorical (mx/log th)))
            (trace :x3 (dist/categorical (mx/log th)))
            (trace :x4 (dist/categorical (mx/log th)))
            th)))

(def TOL 2e-4)   ;; analytical vs closed-form: float32 lgamma floor

;; ===========================================================================
;; 1 — detection fires on the log link
;; ===========================================================================

(println "\n== 1: detection (log link fires, family is :dirichlet-categorical) ==")

(doseq [[nm model n] [["single" m-123 1] ["multi-D" m-D 3] ["multi-E" m-E 5]]]
  (let [pairs (:conjugate-pairs (:schema model))]
    (assert-true (str nm ": one pair per obs (" n ")") (= n (count pairs)))
    (assert-true (str nm ": every pair is :dirichlet-categorical")
                 (every? #(= :dirichlet-categorical (:family %)) pairs))
    (assert-true (str nm ": every pair via :log-link dependency")
                 (every? #(= :log-link (:type (:dependency-type %))) pairs))))

;; ===========================================================================
;; 2 — single-obs marginal: exact == closed form  (A, B, C)
;; ===========================================================================

(println "\n== 2: single-obs marginal == closed form (independent lgamma) ==")

(let [k (rng/fresh-key 1)]
  ;; A: alpha=[1,2,3], x=2  -> log(3/6) = log(1/2)
  (assert-close "A: alpha=[1,2,3] x=2 exact == closed form"
                (closed-form-log-marginal [1.0 2.0 3.0] [2]) (exact-weight m-123 {:x 2} k) TOL)
  (assert-close "A: == hand value -0.693147" -0.693147 (exact-weight m-123 {:x 2} k) TOL)
  ;; B: alpha=[1,2,3], x=0  -> log(1/6)
  (assert-close "B: alpha=[1,2,3] x=0 exact == closed form"
                (closed-form-log-marginal [1.0 2.0 3.0] [0]) (exact-weight m-123 {:x 0} k) TOL)
  (assert-close "B: == hand value -1.791759" -1.791759 (exact-weight m-123 {:x 0} k) TOL)
  ;; C: alpha=[2,2], x=1  -> log(2/4) = log(1/2)
  (assert-close "C: alpha=[2,2] x=1 exact == closed form"
                (closed-form-log-marginal [2.0 2.0] [1]) (exact-weight m-22 {:x 1} k) TOL)
  (assert-close "C: == hand value -0.693147" -0.693147 (exact-weight m-22 {:x 1} k) TOL))

;; ===========================================================================
;; 3 — multi-obs (ordered) marginal == closed form  (D, E)
;;     This is the chain-rule fold over shared theta; correlation matters.
;; ===========================================================================

(println "\n== 3: multi-obs marginal == closed form (Dirichlet-multinomial) ==")

(let [k (rng/fresh-key 2)
      wD (exact-weight m-D {:x0 0 :x1 0 :x2 1} k)
      wE (exact-weight m-E {:x0 0 :x1 1 :x2 0 :x3 2 :x4 0} k)]
  (assert-close "D: alpha=[1,1,1] seq[0,0,1] exact == closed form"
                (closed-form-log-marginal [1.0 1.0 1.0] [0 0 1]) wD TOL)
  (assert-close "D: == hand value -3.401197" -3.401197 wD TOL)
  (assert-close "E: alpha=[2,1,1,1] seq[0,1,0,2,0] exact == closed form"
                (closed-form-log-marginal [2.0 1.0 1.0 1.0] [0 1 0 2 0]) wE TOL)
  (assert-close "E: == hand value -6.445720" -6.445720 wE TOL))

;; ===========================================================================
;; 4 — posterior mean of theta is stored on the latent site
;; ===========================================================================

(println "\n== 4: latent :th holds the posterior mean (alpha+counts)/(sum+n) ==")

(let [k (rng/fresh-key 3)
      thD (mx/->clj (cm/get-value (cm/get-submap (:choices (exact-trace m-D {:x0 0 :x1 0 :x2 1} k)) :th)))
      thE (mx/->clj (cm/get-value (cm/get-submap (:choices (exact-trace m-E {:x0 0 :x1 1 :x2 0 :x3 2 :x4 0} k)) :th)))]
  ;; D posterior alpha=[3,2,1], sum=6 -> mean [0.5, 0.3333, 0.1667]
  (assert-close "D: theta[0] == 0.5" 0.5 (nth thD 0) 1e-4)
  (assert-close "D: theta[1] == 0.3333" (/ 1.0 3.0) (nth thD 1) 1e-4)
  (assert-close "D: theta[2] == 0.1667" (/ 1.0 6.0) (nth thD 2) 1e-4)
  ;; E posterior alpha=[5,2,2,1], sum=10 -> mean [0.5, 0.2, 0.2, 0.1]
  (assert-close "E: theta[0] == 0.5" 0.5 (nth thE 0) 1e-4)
  (assert-close "E: theta[1] == 0.2" 0.2 (nth thE 1) 1e-4)
  (assert-close "E: theta[3] == 0.1" 0.1 (nth thE 3) 1e-4))

;; ===========================================================================
;; 5 — exact == IS  (independent shared-theta Monte Carlo), and the IS
;;     DISCRIMINATES the correct marginal from the ke9i independence trap.
;; ===========================================================================

(println "\n== 5: exact == IS (shared-theta), discriminates the independence trap ==")

(let [k (rng/fresh-key 12345)
      n 200000
      ;; single obs A: IS over 1 obs
      isA (is-log-marginal [1.0 2.0 3.0] [2] n k)
      exactA (exact-weight m-123 {:x 2} (rng/fresh-key 1))
      ;; multi obs D: 3 correlated obs
      isD (is-log-marginal [1.0 1.0 1.0] [0 0 1] n (rng/fresh-key 999))
      exactD (exact-weight m-D {:x0 0 :x1 0 :x2 1} (rng/fresh-key 2))
      ;; the WRONG independent-per-obs estimate (ke9i trap): sum of marginals
      ;; treating each obs as independent => 3 * log(1/3) = -3.295837
      trapD -3.295837
      trueD -3.401197]
  (assert-close "A: IS == exact (single obs)" exactA isA 0.03)
  (assert-close "D: IS == exact (3 shared-theta obs)" exactD isD 0.03)
  (assert-true (str "D: IS (" (.toFixed isD 4) ") closer to true marginal ("
                    trueD ") than to independence trap (" trapD ")")
               (< (js/Math.abs (- isD trueD)) (js/Math.abs (- isD trapD)))))

;; ===========================================================================
;; 6 — regenerate is wired: analytical regenerate parity with handler path
;; ===========================================================================

(println "\n== 6: regenerate wired — analytical == handler (same key) ==")

(defn- strip-analytical [model]
  (dyn/->DynamicGF (:body-fn model) (:source model)
                   (dissoc (:schema model)
                           :auto-handlers :auto-regenerate-transition
                           :auto-regenerate-handlers :analytical-plan
                           :conjugate-pairs :has-conjugate?)))

(let [k (rng/fresh-key 4)
      ;; ONE marginal base trace; regenerate it via BOTH the analytical model and
      ;; the stripped (pure-handler) model, selecting the latent :th. Selecting
      ;; :th re-opens the pair, so the analytical regenerate falls through to the
      ;; handler path — with the SAME base + SAME key the retained-only weights
      ;; must coincide. (Same base trace is essential: the retained weight depends
      ;; on the OLD :th value, so a different base would legitimately differ.)
      base (exact-trace m-D {:x0 0 :x1 0 :x2 1} k)
      seln (sel/select :th)
      ra (p/regenerate (dyn/with-key m-D k) base seln)
      rh (p/regenerate (dyn/with-key (strip-analytical m-D) k) base seln)
      thA (mx/->clj (cm/get-value (cm/get-submap (:choices (:trace ra)) :th)))
      thBase (mx/->clj (cm/get-value (cm/get-submap (:choices base) :th)))]
  (assert-true "regen: analytical path returns a trace" (some? (:trace ra)))
  (assert-true "regen: analytical weight finite" (js/isFinite (mx/item (:weight ra))))
  (assert-true "regen: :th actually resampled (changed from posterior mean)"
               (> (js/Math.abs (- (first thA) (first thBase))) 1e-6))
  (assert-close "regen: analytical weight == handler weight (same base+key)"
                (mx/item (:weight rh)) (mx/item (:weight ra)) TOL))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n==========================================\n"
              "  dirichlet-categorical: " @pass " passed, " @fail " failed\n"
              "=========================================="))
(when (pos? @fail) (js/process.exit 1))
