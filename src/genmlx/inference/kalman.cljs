commit 04ea8410306a54a1e6f0d0163ccb1bf7a450cc81
Author: Robert Johansson <robert@Mac-Mini-KI.lan>
Date:   Mon Mar 9 00:21:07 2026 +0100

    Full Kalman middleware: kalman-fold temporal combinator + Level 2 C3
    
    The cognitive architecture IS the gen function with latent trace sites.
    kalman-fold runs it over T timesteps under the Kalman handler, analytically
    marginalizing latent states. Level 2 C3b verified: exact match (-16454.79).
    
    - kalman-obs now carries mask (5th param) for missing data
    - make-kalman-transition accumulates [P]-shaped LL in :kalman-ll
    - {mean:0, var:0} initial belief: predict always runs, giving prior at t=0
    - kalman-fold: temporal combinator over T steps
    - 35 tests, all pass
    
    Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

diff --git a/src/genmlx/inference/kalman.cljs b/src/genmlx/inference/kalman.cljs
index e5f525f..29382f3 100644
--- a/src/genmlx/inference/kalman.cljs
+++ b/src/genmlx/inference/kalman.cljs
@@ -10,14 +10,14 @@
    1. Pure building blocks — kalman-predict, kalman-update, kalman-sequential-update.
       Use these directly in gen function bodies for explicit control.
 
-   2. Handler middleware — make-kalman-transition + kalman-generate.
-      Wraps generate-transition to intercept kalman-latent and kalman-obs
-      trace sites, running Kalman filtering transparently. The cognitive
-      architecture stays clean (latent states as trace sites), and the
-      handler does the math.
-
-   Both levels produce identical results. Level 1 is more explicit,
-   Level 2 is more composable."
+   2. Handler middleware — make-kalman-transition + kalman-generate + kalman-fold.
+      The cognitive architecture is a gen function using kalman-latent and
+      kalman-obs trace sites. kalman-fold runs it over T timesteps under
+      the Kalman handler, analytically marginalizing latent states.
+      Same gen function works under standard handlers (sampling instead
+      of marginalizing).
+
+   Both levels produce identical results."
   (:require [genmlx.mlx :as mx]
             [genmlx.dist :as dist]
             [genmlx.dist.core :as dc]
@@ -48,17 +48,20 @@
 (defdist kalman-obs
   "Observation with linear-Gaussian structure:
    x ~ N(base-mean + loading * latent-value, noise-std).
-   Under standard handler: samples from Gaussian.
-   Under Kalman handler: provides loading + noise for update step."
-  [base-mean loading latent-value noise-std]
+   mask: 1=observed, 0=missing. Masked observations contribute 0 to LL.
+
+   Under standard handler: masked Gaussian log-prob.
+   Under Kalman handler: provides loading + noise + mask for update step."
+  [base-mean loading latent-value noise-std mask]
   (sample [key]
     (dc/dist-sample
       (dist/gaussian (mx/add base-mean (mx/multiply loading latent-value)) noise-std)
       key))
   (log-prob [v]
-    (dc/dist-log-prob
-      (dist/gaussian (mx/add base-mean (mx/multiply loading latent-value)) noise-std)
-      v)))
+    (mx/multiply mask
+      (dc/dist-log-prob
+        (dist/gaussian (mx/add base-mean (mx/multiply loading latent-value)) noise-std)
+        v))))
 
 ;; ---------------------------------------------------------------------------
 ;; Pure Kalman operations (Level 1)
@@ -141,23 +144,32 @@
 ;; ---------------------------------------------------------------------------
 ;; Handler middleware (Level 2)
 ;; ---------------------------------------------------------------------------
+;;
+;; The handler intercepts kalman-latent and kalman-obs trace sites.
+;; Observation LL accumulates in :kalman-ll ([P]-shaped, per-element),
+;; NOT in :score/:weight (which stay at 0). This keeps the LL structure
+;; intact for the caller to aggregate (sum, mean, etc.).
+;;
+;; Key design: initial belief = {mean: zeros, var: zeros}. The handler
+;; ALWAYS predicts on kalman-latent. At t=0, predict({0,0}, rho, q)
+;; gives {0, q²} — the correct prior. No skip-predict flag needed.
 
 (defn make-kalman-transition
   "Handler middleware: wraps generate-transition for Kalman filtering.
 
    The cognitive architecture is a gen function that uses:
    - (trace :z (kalman-latent rho z-prev noise)) for latent dynamics
-   - (trace :obs (kalman-obs base-mean loading z noise-std)) for observations
+   - (trace :obs (kalman-obs base-mean loading z noise-std mask)) for observations
 
    Under this handler:
    - kalman-latent sites: Kalman predict, return belief mean
-   - kalman-obs sites: Kalman update with constraint, accumulate marginal LL
+   - kalman-obs sites: Kalman update with constraint, accumulate LL in :kalman-ll
    - Other sites: delegate to generate-transition
 
    Handler state additions:
    - :kalman-belief {:mean :var}
    - :kalman-n      number of elements
-   - :kalman-masks  {addr -> mask-array} for missing data"
+   - :kalman-ll     [P]-shaped accumulated marginal LL"
   [latent-addr]
   (fn [state addr dist]
     (cond
@@ -165,7 +177,8 @@
       (and (= addr latent-addr) (= (:type dist) :kalman-latent))
       (let [{:keys [transition-coeff process-noise]} (:params dist)
             belief (or (:kalman-belief state)
-                       (kalman-init (:kalman-n state)))
+                       {:mean (mx/zeros [(:kalman-n state)])
+                        :var  (mx/zeros [(:kalman-n state)])})
             new-belief (kalman-predict belief transition-coeff process-noise)]
         [(:mean new-belief)
          (-> state
@@ -174,19 +187,17 @@
 
       ;; Observation site: Kalman update
       (= (:type dist) :kalman-obs)
-      (let [{:keys [base-mean loading noise-std]} (:params dist)
+      (let [{:keys [base-mean loading noise-std mask]} (:params dist)
             belief (:kalman-belief state)
             constraint (cm/get-submap (:constraints state) addr)
             obs (cm/get-value constraint)
-            mask (get (:kalman-masks state) addr
-                      (mx/ones (mx/shape (:mean belief))))
             {:keys [belief ll]} (kalman-update belief obs base-mean loading noise-std mask)
-            total-ll (mx/sum ll)]
+            n (:kalman-n state)]
         [obs (-> state
                  (assoc :kalman-belief belief)
                  (update :choices cm/set-value addr obs)
-                 (update :score #(mx/add % total-ll))
-                 (update :weight #(mx/add % total-ll)))])
+                 (update :kalman-ll
+                   #(mx/add (or % (mx/zeros [n])) ll)))])
 
       ;; Standard site: delegate
       :else
@@ -203,14 +214,13 @@
    key:         PRNG key
 
    opts (map):
-   - :masks        {addr -> mask-array} for missing data
    - :param-store  parameter store for param sites
-   - :init-belief  custom initial belief (default: N(0,1))
+   - :init-belief  initial belief {:mean :var} (default: {zeros, zeros})
 
-   Returns handler result map with :retval, :weight, :score, :choices,
-   plus :kalman-belief (final belief state)."
+   Returns handler result map with :retval, :choices, :score, :weight,
+   plus :kalman-belief and :kalman-ll ([P]-shaped marginal LL)."
   [gf args constraints latent-addr n key & [opts]]
-  (let [{:keys [masks param-store init-belief]} opts
+  (let [{:keys [param-store init-belief]} opts
         transition (make-kalman-transition latent-addr)
         init-state (cond-> {:choices cm/EMPTY
                             :score (mx/scalar 0.0)
@@ -218,8 +228,39 @@
                             :key key
                             :constraints constraints
                             :kalman-n n
-                            :kalman-belief (or init-belief (kalman-init n))
-                            :kalman-masks (or masks {})}
+                            :kalman-belief (or init-belief
+                                              {:mean (mx/zeros [n])
+                                               :var  (mx/zeros [n])})}
                      param-store (assoc :param-store param-store))]
     (rt/run-handler transition init-state
       (fn [rt] (apply (:body-fn gf) rt args)))))
+
+(defn kalman-fold
+  "Fold a per-step gen function over T timesteps under the Kalman handler.
+
+   step-fn:     gen function with kalman-latent and kalman-obs trace sites
+   latent-addr: keyword address of the latent state
+   n:           number of elements (patients)
+   T:           number of timesteps
+   context-fn:  (fn [t] -> {:args [step-fn-args], :constraints choicemap})
+                builds per-timestep args and observation constraints.
+
+   Uses {mean: zeros, var: zeros} initial belief. The handler always
+   predicts — at t=0 this gives the N(0, q²) prior.
+
+   Returns [P]-shaped total marginal LL (not summed — caller aggregates)."
+  [step-fn latent-addr n T context-fn]
+  (loop [t 0
+         belief {:mean (mx/zeros [n]) :var (mx/zeros [n])}
+         acc-ll (mx/zeros [n])]
+    (if (>= t T)
+      acc-ll
+      (let [{:keys [args constraints]} (context-fn t)
+            result (kalman-generate
+                     step-fn args constraints latent-addr n
+                     (rng/fresh-key t)
+                     {:init-belief belief})
+            step-ll (or (:kalman-ll result) (mx/zeros [n]))]
+        (recur (inc t)
+               (:kalman-belief result)
+               (mx/add acc-ll step-ll))))))
