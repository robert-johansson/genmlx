(ns genmlx.inference.ekf-nd
  "Multi-dimensional Extended Kalman Filter middleware.

   Generalizes the 1D EKF middleware (ekf.cljs) to N latent dimensions with
   full cross-covariance tracking via scalar decomposition. State uses
   N means + N*(N+1)/2 covariance entries, all [P]-shaped arrays — no
   matrix ops, fully element-wise on GPU.

   Two levels of API (matching Kalman/EKF):

   1. Pure building blocks — ekf-nd-predict-one, ekf-nd-update.
      Use directly for explicit control.

   2. Handler middleware — make-multi-ekf-dispatch + ekf-nd-generate + ekf-nd-fold.
      The cognitive architecture uses ekf-nd-latent and ekf-nd-obs trace sites.
      Same gen function works under standard handlers (sampling)
      or ND EKF handler (analytical marginalization).

   Composes with wrap-analytical/compose-middleware."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.handler :as h]
            [genmlx.runtime :as rt]
            [genmlx.inference.analytical :as ana]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.dist.macros :refer [defdist]]))

;; ---------------------------------------------------------------------------
;; Distributions
;; ---------------------------------------------------------------------------

(defdist ekf-nd-latent
  "One component of multi-dim EKF latent dynamics.
   transition-fn: (fn [z] -> z') — differentiable, element-wise.
   prev-value:    previous latent value (used by standard handler).
   process-noise: std dev of process noise.

   Under standard handler: samples from N(f(prev), noise).
   Under ND EKF handler: linearizes f at belief mean, predict step."
  [transition-fn prev-value process-noise]
  (sample [key]
    (dc/dist-sample
      (dist/gaussian (transition-fn prev-value) process-noise)
      key))
  (log-prob [v]
    (dc/dist-log-prob
      (dist/gaussian (transition-fn prev-value) process-noise)
      v)))

(defdist ekf-nd-obs
  "Observation depending on multiple latents.
   obs-fn:        (fn [latent-map] -> predicted-obs) where latent-map = {addr -> value}.
   latent-values: {addr -> value} map (used by standard handler fallback).
   noise-std:     observation noise std dev.
   mask:          1=observed, 0=missing.

   Under standard handler: samples/scores N(h(values), noise).
   Under ND EKF handler: linearizes h via mx/grad, ND Kalman update."
  [obs-fn latent-values noise-std mask]
  (sample [key]
    (dc/dist-sample
      (dist/gaussian (obs-fn latent-values) noise-std)
      key))
  (log-prob [v]
    (mx/multiply mask
      (dc/dist-log-prob
        (dist/gaussian (obs-fn latent-values) noise-std)
        v))))

;; --- Analytical Jacobian variants (no nested autodiff) ---

(defdist ekf-nd-latent-j
  "Like ekf-nd-latent but with analytical Jacobian.
   jacobian-fn: (fn [z] -> df/dz) — returns scalar derivative.
   Eliminates mx/grad in predict step."
  [transition-fn jacobian-fn prev-value process-noise]
  (sample [key]
    (dc/dist-sample
      (dist/gaussian (transition-fn prev-value) process-noise)
      key))
  (log-prob [v]
    (dc/dist-log-prob
      (dist/gaussian (transition-fn prev-value) process-noise)
      v)))

(defdist ekf-nd-obs-j
  "Like ekf-nd-obs but with analytical Jacobian.
   jacobian-fn: (fn [latent-map] -> {addr -> dh/dz_addr}).
   Missing addrs default to zero derivative.
   Eliminates N mx/grad calls per observation update."
  [obs-fn jacobian-fn latent-values noise-std mask]
  (sample [key]
    (dc/dist-sample
      (dist/gaussian (obs-fn latent-values) noise-std)
      key))
  (log-prob [v]
    (mx/multiply mask
      (dc/dist-log-prob
        (dist/gaussian (obs-fn latent-values) noise-std)
        v))))

;; ---------------------------------------------------------------------------
;; State initialization
;; ---------------------------------------------------------------------------

(defn- make-zero-means [addrs n]
  (reduce (fn [m addr] (assoc m addr (mx/zeros [n]))) {} addrs))

(defn- make-zero-covs
  "Initialize N*(N+1)/2 covariance entries to zeros.
   Keys are [addr-i addr-j] ordered by position in addrs."
  [addrs n]
  (let [N (count addrs)]
    (reduce
      (fn [m i]
        (reduce (fn [m j]
                  (assoc m [(nth addrs i) (nth addrs j)] (mx/zeros [n])))
                m (range i N)))
      {} (range N))))

(defn- cov-key
  "Canonical covariance key for a pair of latent addresses.
   Uses position in addrs for consistent ordering."
  [addrs a b]
  (let [ia (.indexOf addrs a)
        ib (.indexOf addrs b)]
    (if (<= ia ib) [a b] [b a])))

;; ---------------------------------------------------------------------------
;; Pure ND EKF operations
;; ---------------------------------------------------------------------------

(def ^:private LOG-2PI 1.8378770664093453)

(defn- predict-one-core
  "Shared predict logic given pre-computed f(z0) and Jacobian A."
  [addrs addr means covs f-z0 A q]
  (let [new-means (assoc means addr f-z0)
        new-covs (reduce
                   (fn [cs other]
                     (let [k (cov-key addrs addr other)
                           p (get cs k)]
                       (if (= addr other)
                         (assoc cs k (mx/add (mx/multiply A (mx/multiply A p))
                                             (mx/multiply q q)))
                         (assoc cs k (mx/multiply A p)))))
                   covs addrs)]
    [f-z0 new-means new-covs]))

(defn ekf-nd-predict-one
  "EKF predict for one latent component (auto-diff Jacobian).

   Linearizes transition-fn at the current belief mean, updates this
   component's mean and all covariance entries involving it. Sequential
   prediction across all components produces the correct full predict:
   P_ij final = A_i * A_j * P_ij_orig + Q_i^2 * delta_ij.

   Returns [predicted-mean new-means new-covs]."
  [addrs addr means covs f q]
  (let [z0 (get means addr)]
    (predict-one-core addrs addr means covs
                      (f z0)
                      ((mx/grad (fn [z] (mx/sum (f z)))) z0)
                      q)))

(defn ekf-nd-predict-one-j
  "EKF predict with analytical Jacobian. No mx/grad.
   jacobian-fn: (fn [z] -> df/dz), returns [P]-shaped derivative."
  [addrs addr means covs f jacobian-fn q]
  (let [z0 (get means addr)]
    (predict-one-core addrs addr means covs
                      (f z0) (jacobian-fn z0) q)))

(defn- update-core
  "Shared ND EKF update given pre-computed Jacobian vector H.
   H: vector of [P]-shaped derivatives, one per latent in addrs order.
   Uses scalar-decomposed Joseph form: P_ij -= mask * PH_i * PH_j / S."
  [addrs means covs obs pred H noise-std mask]
  (let [N (count addrs)
        innov (mx/subtract obs pred)
        ;; PH_i = sum_j P_ij * H_j
        PH (mapv (fn [i]
                   (reduce
                     (fn [acc j]
                       (let [k (cov-key addrs (nth addrs i) (nth addrs j))]
                         (mx/add acc (mx/multiply (get covs k) (nth H j)))))
                     (mx/scalar 0.0)
                     (range N)))
                 (range N))
        ;; Innovation variance: S = sum_i H_i * PH_i + R^2
        S (reduce (fn [acc i] (mx/add acc (mx/multiply (nth H i) (nth PH i))))
                  (mx/multiply noise-std noise-std)
                  (range N))
        ;; Mean update: mu_i += mask * (PH_i / S) * innov
        mi (mx/multiply mask innov)
        new-means (reduce
                    (fn [m i]
                      (let [addr (nth addrs i)]
                        (assoc m addr
                          (mx/add (get m addr)
                                  (mx/multiply (mx/divide (nth PH i) S) mi)))))
                    means (range N))
        ;; Covariance update: P_ij -= mask * PH_i * PH_j / S
        new-covs (reduce
                   (fn [cs i]
                     (reduce
                       (fn [cs j]
                         (let [k [(nth addrs i) (nth addrs j)]
                               p (get cs k)]
                           (assoc cs k
                             (mx/subtract p
                               (mx/multiply mask
                                 (mx/divide (mx/multiply (nth PH i) (nth PH j)) S))))))
                       cs (range i N)))
                   covs (range N))
        ;; Marginal LL: -0.5 * (log(2pi) + log(S) + innov^2/S)
        ll (mx/multiply mask
             (mx/multiply (mx/scalar -0.5)
               (mx/add (mx/scalar LOG-2PI)
                 (mx/add (mx/log S)
                   (mx/divide (mx/multiply innov innov) S)))))]
    {:means new-means :covs new-covs :ll ll}))

(defn ekf-nd-update
  "ND EKF update with auto-diff Jacobian (one mx/grad call per latent).
   Returns {:means :covs :ll}."
  [addrs means covs obs obs-fn noise-std mask]
  (let [H (mapv (fn [addr]
                  (let [g (mx/grad (fn [zi] (mx/sum (obs-fn (assoc means addr zi)))))]
                    (g (get means addr))))
               addrs)]
    (update-core addrs means covs obs (obs-fn means) H noise-std mask)))

(defn ekf-nd-update-j
  "ND EKF update with analytical Jacobian. No mx/grad — no nested autodiff.
   jacobian-fn: (fn [latent-map] -> {addr -> dh/dz_addr}).
   Missing addrs default to zero derivative.
   Returns {:means :covs :ll}."
  [addrs means covs obs obs-fn jacobian-fn noise-std mask]
  (let [H-map (jacobian-fn means)
        H (mapv (fn [addr] (or (get H-map addr) (mx/scalar 0.0))) addrs)]
    (update-core addrs means covs obs (obs-fn means) H noise-std mask)))

;; ---------------------------------------------------------------------------
;; Handler middleware
;; ---------------------------------------------------------------------------

(defn- make-latent-handler
  "Shared latent handler logic for both auto-diff and analytical variants."
  [latent-addrs addr-set predict-fn]
  (fn [state addr dist]
    (if (contains? addr-set addr)
      (let [n (:ekf-nd-n state)
            means (or (:ekf-nd-means state) (make-zero-means latent-addrs n))
            covs (or (:ekf-nd-covs state) (make-zero-covs latent-addrs n))
            [val new-means new-covs] (predict-fn latent-addrs addr means covs dist)]
        [val (-> state
                 (assoc :ekf-nd-means new-means
                        :ekf-nd-covs new-covs)
                 (update :choices cm/set-value addr val))])
      nil)))

(defn- make-obs-handler
  "Shared obs handler logic for both auto-diff and analytical variants."
  [latent-addrs update-fn]
  (fn [state addr dist]
    (let [n (:ekf-nd-n state)
          means (:ekf-nd-means state)
          covs (:ekf-nd-covs state)
          constraint (cm/get-submap (:constraints state) addr)
          obs (cm/get-value constraint)
          result (update-fn latent-addrs means covs obs dist)]
      [obs (-> state
               (assoc :ekf-nd-means (:means result)
                      :ekf-nd-covs (:covs result))
               (update :choices cm/set-value addr obs)
               (update :ekf-nd-ll
                 #(mx/add (or % (mx/zeros [n])) (:ll result))))])))

(defn make-multi-ekf-dispatch
  "Create multi-dim EKF dispatch map for use with wrap-analytical.

   latent-addrs: vector of keyword addresses, e.g. [:z0 :z1 :z2]

   Handles both auto-diff types (ekf-nd-latent, ekf-nd-obs) and
   analytical Jacobian types (ekf-nd-latent-j, ekf-nd-obs-j).

   State keys:
   - :ekf-nd-means  {addr -> [P]-shaped mean}
   - :ekf-nd-covs   {[addr-i addr-j] -> [P]-shaped covariance entry}
   - :ekf-nd-ll     [P]-shaped accumulated marginal LL
   - :ekf-nd-n      number of elements"
  [latent-addrs]
  (let [addr-set (set latent-addrs)]
    {:ekf-nd-latent
     (make-latent-handler latent-addrs addr-set
       (fn [addrs addr means covs dist]
         (let [{:keys [transition-fn process-noise]} (:params dist)]
           (ekf-nd-predict-one addrs addr means covs
                               transition-fn process-noise))))

     :ekf-nd-latent-j
     (make-latent-handler latent-addrs addr-set
       (fn [addrs addr means covs dist]
         (let [{:keys [transition-fn jacobian-fn process-noise]} (:params dist)]
           (ekf-nd-predict-one-j addrs addr means covs
                                 transition-fn jacobian-fn process-noise))))

     :ekf-nd-obs
     (make-obs-handler latent-addrs
       (fn [addrs means covs obs dist]
         (let [{:keys [obs-fn noise-std mask]} (:params dist)]
           (ekf-nd-update addrs means covs obs obs-fn noise-std mask))))

     :ekf-nd-obs-j
     (make-obs-handler latent-addrs
       (fn [addrs means covs obs dist]
         (let [{:keys [obs-fn jacobian-fn noise-std mask]} (:params dist)]
           (ekf-nd-update-j addrs means covs obs obs-fn jacobian-fn
                            noise-std mask))))}))
(defn make-multi-ekf-transition
  "Handler middleware: wraps generate-transition for multi-dim EKF."
  [latent-addrs]
  (ana/wrap-analytical h/generate-transition
                       (make-multi-ekf-dispatch latent-addrs)))

(defn ekf-nd-generate
  "Run a gen function under the multi-dim EKF handler.

   gf:           DynamicGF with ekf-nd-latent/ekf-nd-obs trace sites
   args:         gen function arguments
   constraints:  choicemap with observation constraints
   latent-addrs: vector of latent address keywords
   n:            number of elements (e.g. units)
   key:          PRNG key

   opts (map):
   - :param-store  parameter store for param sites
   - :init-means   initial means {addr -> [P]-shaped}
   - :init-covs    initial covs {[a b] -> [P]-shaped}

   Returns handler result with :ekf-nd-means, :ekf-nd-covs, :ekf-nd-ll."
  [gf args constraints latent-addrs n key & [opts]]
  (let [{:keys [param-store init-means init-covs]} opts
        transition (make-multi-ekf-transition latent-addrs)
        init-state (cond-> {:choices cm/EMPTY
                            :score (mx/scalar 0.0)
                            :weight (mx/scalar 0.0)
                            :key key
                            :constraints constraints
                            :ekf-nd-n n
                            :ekf-nd-means (or init-means
                                              (make-zero-means latent-addrs n))
                            :ekf-nd-covs (or init-covs
                                             (make-zero-covs latent-addrs n))}
                     param-store (assoc :param-store param-store))]
    (rt/run-handler transition init-state
      (fn [rt] (apply (:body-fn gf) rt args)))))

(defn ekf-nd-fold
  "Fold a per-step gen function over T timesteps under multi-dim EKF handler.

   step-fn:      gen function with ekf-nd-latent/ekf-nd-obs trace sites
   latent-addrs: vector of latent address keywords
   n:            number of elements
   T:            number of timesteps
   context-fn:   (fn [t] -> {:args [step-fn-args], :constraints choicemap})

   Initial state: all zeros. Handler always predicts — at t=0,
   predict({0,0}, f, q) gives the correct N(f(0), q^2) prior.

   Returns {:ll [P]-shaped total LL, :means final means, :covs final covs}."
  [step-fn latent-addrs n T context-fn]
  (loop [t 0
         means (make-zero-means latent-addrs n)
         covs (make-zero-covs latent-addrs n)
         acc-ll (mx/zeros [n])]
    (if (>= t T)
      {:ll acc-ll :means means :covs covs}
      (let [{:keys [args constraints]} (context-fn t)
            result (ekf-nd-generate
                     step-fn args constraints latent-addrs n
                     (rng/fresh-key t)
                     {:init-means means :init-covs covs})
            step-ll (or (:ekf-nd-ll result) (mx/zeros [n]))]
        (recur (inc t)
               (:ekf-nd-means result)
               (:ekf-nd-covs result)
               (mx/add acc-ll step-ll))))))
