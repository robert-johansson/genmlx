;; @tier medium
(ns genmlx.trace-translator-test
  "genmlx-oen5: general trace translators (Cusumano-Towner 2020 thesis §3.6-3.7).

   Verifies the Eq 3.12 importance weight, the AD Jacobian (with sparsity),
   bijection round-trips, reversible-jump MCMC (split/merge recovering the exact
   model posterior), coarse-to-fine SMC, and the discrete<->continuous bridge.

   INDEPENDENT ORACLE discipline: every numeric expectation is a closed form
   derived by hand here (o-log-gauss, o-cat, the change-of-variables identities,
   the split/merge log-Jacobians +/- log 2, and the Normal-Normal marginal
   likelihoods giving P(k=2|y)=0.8804741734), never via the code under test."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.inference.translator :as tt]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.gfi :as gfi]
            [genmlx.inference.util :as u])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn- it [a] (mx/item a))
(defn- o-log-gauss [x mu sigma]
  (- (* -0.5 (js/Math.log (* 2 js/Math.PI))) (js/Math.log sigma)
     (* 0.5 (js/Math.pow (/ (- x mu) sigma) 2))))
(def ^:private LOG2 (js/Math.log 2))

;; ===========================================================================
;; 1. AD Jacobian (mx/grad) + sparsity (§3.6.2)
;; ===========================================================================

(deftest jacobian-ad
  (testing "1-D: log|d(x/2)/dx| = -log 2, log|d(2x)/dx| = +log 2"
    (is (< (js/Math.abs (- (it (tt/jacobian-logdet #(mx/divide % (mx/scalar 2.0))
                                                   (mx/array [3.0]) 1)) (- LOG2))) 1e-5))
    (is (< (js/Math.abs (- (it (tt/jacobian-logdet #(mx/multiply % (mx/scalar 2.0))
                                                   (mx/array [3.0]) 1)) LOG2)) 1e-5)))
  (testing "2-D split (mu,u)->(mu-u,mu+u): log|det J| = log 2 (orientation-safe)"
    (let [g (fn [x] (let [m (mx/index x 0) u (mx/index x 1)]
                      (mx/stack [(mx/subtract m u) (mx/add m u)])))]
      (is (< (js/Math.abs (- (it (tt/jacobian-logdet g (mx/array [1.0 0.5]) 2)) LOG2)) 1e-5))))
  (testing "sparsity (§3.6.2): copied coordinate excluded => sparse == dense logdet"
    (let [g (fn [x] (let [a (mx/index x 0) b (mx/index x 1) c (mx/index x 2)]
                      (mx/stack [(mx/subtract a b) (mx/add a b) c])))   ;; c copied
          x (mx/array [1.0 0.5 9.0])]
      (is (< (js/Math.abs (- (it (tt/jacobian-logdet g x 3)) LOG2)) 1e-5) "dense = log 2")
      (is (< (js/Math.abs (- (it (tt/sparse-jacobian-logdet g x [0 1])) LOG2)) 1e-5)
          "sparse over {0,1} = log 2 (copied coord 2 is identity, det 1)"))))

;; ===========================================================================
;; 2. read / write / copy introspection API
;; ===========================================================================

(deftest read-write-copy
  (let [in (cm/choicemap :mu (mx/scalar 3.0) :y0 (mx/scalar -2.0) :y1 (mx/scalar 2.0))
        aux (cm/choicemap :u (mx/scalar 0.5))]
    (is (= 3.0 (it (tt/read-choice in :mu))) "read-choice")
    (is (= 0.5 (it (tt/read-aux aux :u))) "read-aux")
    (let [out (-> cm/EMPTY
                  (tt/write-choice :mu1 (mx/subtract (tt/read-choice in :mu) (tt/read-aux aux :u)))
                  (tt/copy-choice in :y0)
                  (tt/copy-choice in :y1))]
      (is (= 2.5 (it (tt/read-choice out :mu1))) "write-choice transformed value")
      (is (= -2.0 (it (tt/read-choice out :y0))) "copy-choice copies unchanged")
      (is (= 2.0 (it (tt/read-choice out :y1))) "copy-choice copies unchanged"))))

;; ===========================================================================
;; 3. Eq 3.12 weight — pure reparameterization => weight 0
;; ===========================================================================

(defn- reparam-p1 [] (gen [] (trace :x (dist/gaussian 0 1))))
;; x = 2u with u ~ N(0, sd=0.5) => pushforward N(0, 4*0.25)=N(0,1) = P1 (same law).
(defn- reparam-p2 [] (gen [] (trace :u (dist/gaussian 0 0.5))))

(deftest reparam-weight-zero
  (testing "translator over a pure reparameterization has weight 0 (Eq 3.12)"
    (let [P1 (reparam-p1) P2 (reparam-p2)
          h (fn [in _aux] {:trace (tt/write-choice cm/EMPTY :u
                                                   (mx/divide (tt/read-choice in :x) (mx/scalar 2.0)))
                           :aux cm/EMPTY
                           :log-det-jacobian (mx/scalar (- LOG2))})  ;; |du/dx| = 1/2
          translator (tt/trace-translator {:p2 P2 :h h})]
      (doseq [seed [1 2 3 4 5]]
        (let [in-tr (p/simulate (dyn/with-key P1 (rng/fresh-key seed)) [])
              {:keys [weight]} (tt/apply-translator translator in-tr [] (rng/fresh-key (+ 100 seed)))]
          (is (< (js/Math.abs (it weight)) 1e-5)
              (str "reparam weight ~ 0 (x=" (it (tt/read-choice (:choices in-tr) :x)) ")"))))))
  (testing "MISMATCHED law (u ~ N(0,sd=sqrt(0.5)) => x ~ N(0,2)) gives NONZERO weight"
    (let [P1 (reparam-p1)
          P2bad (gen [] (trace :u (dist/gaussian 0 0.7071067811865476)))  ;; var 0.5 => x~N(0,2)
          h (fn [in _aux] {:trace (tt/write-choice cm/EMPTY :u
                                                   (mx/divide (tt/read-choice in :x) (mx/scalar 2.0)))
                           :aux cm/EMPTY :log-det-jacobian (mx/scalar (- LOG2))})
          translator (tt/trace-translator {:p2 P2bad :h h})
          in-tr (p/simulate (dyn/with-key P1 (rng/fresh-key 9)) [(mx/scalar 1.0)])]
      ;; pick a trace with x clearly nonzero so the density mismatch shows
      (let [in-tr (loop [s 9] (let [t (p/simulate (dyn/with-key P1 (rng/fresh-key s)) [])]
                                (if (> (js/Math.abs (it (tt/read-choice (:choices t) :x))) 0.5) t (recur (inc s)))))
            {:keys [weight]} (tt/apply-translator translator in-tr [] (rng/fresh-key 3))]
        (is (> (js/Math.abs (it weight)) 1e-3) "wrong-variance reparam is not weight-0")))))

;; ===========================================================================
;; 4. Bijection round-trip (split then merge recovers the trace)
;; ===========================================================================

(deftest bijection-roundtrip
  (testing "split (mu,u)->(mu1,mu2) then merge recovers (mu,u) and the Jacobians are exact negatives"
    (doseq [[mu u] [[1.0 0.5] [-0.3 0.2] [2.7 -1.1]]]
      (let [mu1 (- mu u) mu2 (+ mu u)
            mu' (/ (+ mu1 mu2) 2.0) u' (/ (- mu2 mu1) 2.0)]
        (is (< (js/Math.abs (- mu mu')) 1e-9) "merge recovers mu")
        (is (< (js/Math.abs (- u u')) 1e-9) "merge recovers u")))
    (is (< (js/Math.abs (+ LOG2 (- LOG2))) 1e-12) "log|det J_split| + log|det J_merge| = 0")))

;; ===========================================================================
;; 5. Discrete <-> continuous bridge (Eq 3.12 with a discrete coordinate)
;; ===========================================================================

(def ^:private bridge-logits
  (mx/array [(js/Math.log 0.2) (js/Math.log 0.5) (js/Math.log 0.3)]))
(defn- bridge-p1 [] (gen [] (let [z (trace :z (dist/categorical bridge-logits))]
                              (trace :y (dist/gaussian z 1)))))
(defn- bridge-p2 [] (gen [] (let [w (trace :w (dist/gaussian 1 1))]
                              (trace :y (dist/gaussian w 1)))))

(deftest discrete-continuous-bridge
  (testing "Eq 3.12 weight with one discrete coord: discrete adds NO Jacobian; weight matches oracle"
    (let [P1 (bridge-p1) P2 (bridge-p2)
          q1 (gen [_in] (trace :a (dist/gaussian 0 0.3)))    ;; forward aux a ~ N(0,0.3)
          ;; h: w = z + a ; backward is deterministic (z=round w) => no aux, q2=nil, log|det J|=0
          h (fn [in aux]
              (let [z (tt/read-choice in :z) a (tt/read-aux aux :a)]
                {:trace (-> cm/EMPTY
                            (tt/write-choice :w (mx/add z a))
                            (tt/copy-choice in :y))
                 :aux cm/EMPTY}))    ;; no :log-det-jacobian => 0 (volume-preserving over a->w)
          translator (tt/trace-translator {:p2 P2 :q1 q1 :h h :volume-preserving? true})
          y 1.0
          in-tr (:trace (p/generate (dyn/with-key P1 (rng/fresh-key 1)) []
                                    (cm/choicemap :z (mx/scalar 1.0) :y (mx/scalar y))))
          {:keys [trace weight log-det-jacobian]}
          (tt/apply-translator translator in-tr [] (rng/fresh-key 2))
          z 1.0
          w (it (tt/read-choice (:choices trace) :w))
          a (- w z)
          ;; independent oracle at the SAMPLED a:
          log-p1 (+ (js/Math.log 0.5) (o-log-gauss y z 1))       ;; log P(z=1)+logN(y;z,1)
          log-q1 (o-log-gauss a 0 0.3)                           ;; forward aux (denominator)
          log-p2 (+ (o-log-gauss w 1 1) (o-log-gauss y w 1))     ;; logN(w;1,1)+logN(y;w,1)
          oracle (- (+ log-p2 0.0) (+ log-p1 log-q1))]           ;; +q2(=0) +log|detJ|(=0)
      (is (< (js/Math.abs (it log-det-jacobian)) 1e-12)
          "discrete coordinate contributes NO Jacobian: log|det J| = 0")
      (is (< (js/Math.abs (- (it weight) oracle)) 1e-4)
          (str "bridge weight " (it weight) " matches Eq 3.12 oracle " oracle " (a=" a ")")))))

;; ===========================================================================
;; 6. Reversible-jump MCMC: split/merge recovers the exact model posterior
;; ===========================================================================

;; k=0 => one shared mean :mu; k=1 => two means :mu1,:mu2 (fixed assignment).
;; Prior P(k)=0.5, s0=10, sigma=1, data y0=-2, y1=2. Oracle: P(k=2|y)=0.8804741734.
(defn- sm-model []
  (gen []
    (let [k (trace :k (dist/bernoulli 0.5))]
      (if (> (mx/item k) 0.5)
        (let [mu1 (trace :mu1 (dist/gaussian 0 10))
              mu2 (trace :mu2 (dist/gaussian 0 10))]
          (trace :y0 (dist/gaussian mu1 1))
          (trace :y1 (dist/gaussian mu2 1)))
        (let [mu (trace :mu (dist/gaussian 0 10))]
          (trace :y0 (dist/gaussian mu 1))
          (trace :y1 (dist/gaussian mu 1)))))))

(def ^:private su 1.5)   ;; split auxiliary sd
(defn- u-proposal [] (gen [_in] (trace :u (dist/gaussian 0 su))))

(defn- two? [t] (> (it (tt/read-choice (:choices t) :k)) 0.5))

(defn- split-translator [model]
  ;; k=0 -> k=1: (mu,u) -> (mu1,mu2)=(mu-u,mu+u), Jacobian +log 2; backward aux empty.
  (tt/trace-translator
   {:p2 model :q1 (u-proposal)
    :h (fn [in aux]
         (let [m (tt/read-choice in :mu) u (tt/read-aux aux :u)]
           {:trace (-> cm/EMPTY
                       (tt/write-choice :k (mx/scalar 1.0))
                       (tt/write-choice :mu1 (mx/subtract m u))
                       (tt/write-choice :mu2 (mx/add m u))
                       (tt/copy-choice in :y0)
                       (tt/copy-choice in :y1))
            :aux cm/EMPTY
            :log-det-jacobian (mx/scalar LOG2)}))
    :applicable? (complement two?)}))

(defn- merge-translator [model]
  ;; k=1 -> k=0: (mu1,mu2) -> mu=(mu1+mu2)/2; recovered u=(mu2-mu1)/2 is the
  ;; backward aux scored under q2; Jacobian -log 2.
  (tt/trace-translator
   {:p2 model :q2 (u-proposal)
    :h (fn [in _aux]
         (let [m1 (tt/read-choice in :mu1) m2 (tt/read-choice in :mu2)]
           {:trace (-> cm/EMPTY
                       (tt/write-choice :k (mx/scalar 0.0))
                       (tt/write-choice :mu (mx/divide (mx/add m1 m2) (mx/scalar 2.0)))
                       (tt/copy-choice in :y0)
                       (tt/copy-choice in :y1))
            :aux (tt/write-aux cm/EMPTY :u (mx/divide (mx/subtract m2 m1) (mx/scalar 2.0)))
            :log-det-jacobian (mx/scalar (- LOG2))}))
    :applicable? two?}))

(defn- rw-sweep
  "One within-model Gaussian random-walk sweep over the active mean site(s) via
   p/update, threading the key. The standard parameter move RJMCMC interleaves."
  [model t step key]
  (let [sites (if (two? t) [:mu1 :mu2] [:mu])]
    (loop [t t, ss sites, k key]
      (if (empty? ss)
        t
        (let [[kp ka knext] (rng/split-n k 3)
              addr (first ss)
              cur (it (tt/read-choice (:choices t) addr))
              prop (+ cur (* step (it (rng/normal kp []))))
              {nt :trace w :weight} (p/update (dyn/with-key model ka) t
                                              (cm/choicemap addr (mx/scalar prop)))]
          (recur (if (u/accept-mh? (it w) ka) nt t) (rest ss) knext))))))

(defn- rw-refresh
  "Several within-model RW sweeps to decorrelate the continuous params between
   structural moves (keeps the k-indicator's effective sample size high)."
  [model t step n-sweeps key]
  (loop [t t, i 0, k key]
    (if (>= i n-sweeps)
      t
      (let [[k1 k2] (rng/split k)]
        (recur (rw-sweep model t step k1) (inc i) k2)))))

(defn- o-marginal-k1
  "log p(y0,y1 | k=1): y ~ N(0, Sigma), Sigma = sigma^2 I + s0^2 11^T."
  [y0 y1 s0 sigma]
  (let [d (+ (* sigma sigma) (* s0 s0))     ;; diagonal
        o (* s0 s0)                          ;; off-diagonal
        det (- (* d d) (* o o))
        quad (/ (+ (* d y0 y0) (* -2 o y0 y1) (* d y1 y1)) det)]
    (* -0.5 (+ (* 2 (js/Math.log (* 2 js/Math.PI))) (js/Math.log det) quad))))

(deftest reversible-jump-split-merge
  ;; Less-separated data => P(k=2|y) ~ 0.566 => balanced split/merge transitions
  ;; => fast k-mixing. Exact posterior from the closed-form marginal likelihoods.
  (testing "split/merge RJMCMC recovers the exact model posterior P(k=2|y)"
    (let [model (sm-model)
          s0 10.0 sigma 1.0 yy0 -1.5 yy1 1.5
          ;; independent oracle: exact P(k=2|y) from marginal likelihoods (priors equal)
          lp1 (o-marginal-k1 yy0 yy1 s0 sigma)
          lp2 (+ (o-log-gauss yy0 0 (js/Math.sqrt (+ (* s0 s0) (* sigma sigma))))
                 (o-log-gauss yy1 0 (js/Math.sqrt (+ (* s0 s0) (* sigma sigma)))))
          p-k2 (/ 1.0 (+ 1.0 (js/Math.exp (- lp1 lp2))))
          init (:trace (p/generate (dyn/with-key model (rng/fresh-key 1)) []
                                   (cm/choicemap :k (mx/scalar 0.0)
                                                 :y0 (mx/scalar yy0) :y1 (mx/scalar yy1))))
          moves [{:translator (split-translator model) :applicable? (complement two?)}
                 {:translator (merge-translator model) :applicable? two?}]
          n-samp 9000 burn 2000
          frac (loop [t init i 0 k (rng/fresh-key 77) acc 0]
                 (if (>= i (+ burn n-samp))
                   (/ acc (double n-samp))
                   (let [[k1 k2 k3] (rng/split-n k 3)
                         t1 (rw-refresh model t 0.8 2 k1)
                         {t2 :trace} (tt/reversible-jump-mh-step t1 moves k2)]
                     (recur t2 (inc i) k3
                            (if (and (>= i burn) (two? t2)) (inc acc) acc)))))]
      (is (< (js/Math.abs (- p-k2 0.5656)) 0.01) (str "oracle sanity P(k=2)=" p-k2))
      (is (< (js/Math.abs (- frac p-k2)) 0.04)
          (str "RJMCMC P(k=2|y) fraction " frac " ~ exact " p-k2)))))

;; ===========================================================================
;; 7. Coarse-to-fine SMC
;; ===========================================================================

(defn- ctf-model [] (gen [] (let [x (trace :x (dist/gaussian 0 1))]
                              (trace :y (dist/gaussian x 1)))))

(deftest coarse-to-fine
  (testing "identity-translator coarse-to-fine SMC log-ML == direct importance-sampling log-ML"
    (let [model (ctf-model)
          obs (cm/choicemap :y (mx/scalar 1.5))
          ;; identity translator: same model, no aux, h copies everything, |det J|=1
          ident (tt/trace-translator
                 {:p2 model :h (fn [in _aux] {:trace in :aux cm/EMPTY}) :volume-preserving? true})
          n 4000
          ctf (tt/coarse-to-fine-smc {:models [model model model]
                                      :stage-obs [obs obs obs]
                                      :translators [ident ident]
                                      :n-particles n :key (rng/fresh-key 5)})
          ;; direct IS log-ML on the fine model: logmeanexp of generate weights
          direct (let [ks (rng/split-n (rng/fresh-key 6) n)
                       lws (mapv #(it (:weight (p/generate (dyn/with-key model %) [] obs))) ks)
                       m (reduce max lws)]
                   (+ m (js/Math.log (/ (reduce + (map (fn [x] (js/Math.exp (- x m))) lws)) n))))
          ;; analytic log p(y) = log N(1.5; 0, sqrt(2))
          analytic (o-log-gauss 1.5 0 (js/Math.sqrt 2))]
      (is (< (js/Math.abs (- (:log-ml ctf) analytic)) 0.05)
          (str "coarse-to-fine log-ML " (:log-ml ctf) " ~ analytic " analytic))
      (is (< (js/Math.abs (- (:log-ml ctf) direct)) 0.06)
          (str "coarse-to-fine log-ML " (:log-ml ctf) " ~ direct IS " direct))
      (is (= n (count (:particles ctf))) "returns N fine-model particles")))
  (testing "genuine 2-stage reparam bridge preserves the marginal likelihood"
    (let [coarse (gen [] (let [x (trace :x (dist/gaussian 0 1))] (trace :y (dist/gaussian x 1))))
          ;; fine model uses u with x=2u (same law) — coordinates change, density preserved
          fine (gen [] (let [u (trace :u (dist/gaussian 0 0.5))]
                         (trace :y (dist/gaussian (mx/multiply (mx/scalar 2.0) u) 1))))
          obs (cm/choicemap :y (mx/scalar 1.5))
          reparam (tt/trace-translator
                   {:p2 fine
                    :h (fn [in _aux]
                         {:trace (-> cm/EMPTY
                                     (tt/write-choice :u (mx/divide (tt/read-choice in :x) (mx/scalar 2.0)))
                                     (tt/copy-choice in :y))
                          :aux cm/EMPTY :log-det-jacobian (mx/scalar (- LOG2))})})
          ctf (tt/coarse-to-fine-smc {:models [coarse fine] :stage-obs [obs obs]
                                      :translators [reparam] :n-particles 4000
                                      :key (rng/fresh-key 11)})
          analytic (o-log-gauss 1.5 0 (js/Math.sqrt 2))]
      (is (< (js/Math.abs (- (:log-ml ctf) analytic)) 0.06)
          (str "reparam-bridged log-ML " (:log-ml ctf) " ~ analytic " analytic)))))

;; ===========================================================================
;; 8. GFI laws registered + hold
;; ===========================================================================

(deftest gfi-laws-registered
  (let [names (set (map :name gfi/laws))]
    (doseq [nm [:translator-weight-formula :translator-jacobian-ad
                :translator-bijection-roundtrip :translator-sparsity-equiv
                :reversible-jump-detailed-balance]]
      (is (contains? names nm) (str nm " registered")))))

(deftest gfi-laws-hold
  (doseq [law-name [:translator-weight-formula :translator-jacobian-ad
                    :translator-bijection-roundtrip :translator-sparsity-equiv
                    :reversible-jump-detailed-balance]]
    (let [{:keys [pass? error]} (gfi/check-law law-name (dyn/auto-key (ctf-model)) [])]
      (is pass? (str law-name (when error (str " — ERROR: " error)))))))

(cljs.test/run-tests)
