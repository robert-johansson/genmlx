(ns genmlx.agents.belief
  "Tensor belief-update kernel (bean genmlx-kpuo): the pure-MLX observation-Bayes
   filter that replaces the host Clojure-map belief filters in
   genmlx.agents.pomdp/update-belief and genmlx.agents.biased-planners/bayes-update.

   CORRECTION TO THE BEAN FORMULA. The swxr epic specifies the textbook
   state-belief filter `b' = normalize(O[.,.,a] ⊙ (T^T·b))`, which assumes a
   latent that TRANSITIONS. This codebase's POMDP latent is a STATIC world
   identity w (which goal pays / which restaurants are open) that never changes
   for the episode (pomdp.cljs:5-13, pomdp_env.cljs:108-110, biased_planners.cljs:484).
   The physical state s transitions separately (agent/sample-next); the BELIEF
   update is pure observation-Bayes with NO `T^T·b` prediction step:

       b'(w) ∝ b(w) · P(obs | world_w, loc)
             = b(w) · [observe(w, loc) = o]        (deterministic reveal)

   which is exactly what both host filters compute:
     - pomdp.cljs:76-83   normalize-logs over {log b(w) + (0 if match else -Inf)}
     - biased_planners.cljs:483-491  raw[w]=(if match b[w] 0) ; b'=raw/Σraw

   So the kernel is a [W]-vector elementwise multiply + sum + divide:

       raw = b ⊙ L                                 ; [W]
       z   = Σ raw                                 ; []
       b'  = where(z > eps, raw/z, b)              ; [W], defensive identity at z≈0

   The likelihood L is DATA (built host-side from the host `observe` geometry,
   then `mx/array`'d once into a [W] vector); only b carries gradient. There is no
   per-step `mx/item` and no per-step host-map arithmetic, and b stays a
   differentiable MLX [W] array — which is what the downstream differentiable
   POMDP work (genmlx-5x3f) needs.

   The default agent paths stay host-side (the seam threads Clojure maps/vectors);
   this kernel is opt-in via the agents' `:update-belief-tensor` and
   simulate-pomdp's `:belief-mode :tensor`."
  (:require [genmlx.mlx :as mx]))

(def ^:private eps
  "Defensive normalization floor. Host filters keep b unchanged when the observed
   o is impossible under the current belief (Σraw = 0). With float32 {0,1}
   likelihoods, Σraw is the exact sum of the surviving belief masses — never a
   denormal — so any eps below the smallest representable belief mass reproduces
   the host `(pos? z)` guard. 1e-30 is far below any demo belief mass."
  1e-30)

(defn obs-likelihood-vec
  "Build the per-world observation likelihood L : [W] float32, where
   L[w] = 1.0 if (observe w loc) equals o, else 0.0. Geometry stays host-side:
   `observe` is the host model and the comparison is Clojure `=` on the obs value
   (a goal keyword, or a vector of [restaurant open?] pairs, or nil), so the SAME
   semantics as the host filters. `worlds` is the FIXED world ordering the belief
   vector is aligned to."
  [observe worlds loc o]
  (mx/array (clj->js (mapv (fn [w] (if (= (observe w loc) o) 1.0 0.0)) worlds))
            mx/float32))

(defn filter-step
  "Pure differentiable belief filter core: b' = where(Σ(b⊙L) > eps, (b⊙L)/Σ, b).
   Both b and L are [W] MLX arrays; returns a [W] MLX array. No mx/item, no host
   arithmetic — the gradient flows through b (L is a constant likelihood).
   Safe-where: the division uses a guarded denominator (1 where z ≤ eps) so the
   untaken branch never computes raw/0 — with autograd, a NaN/Inf in the untaken
   where branch poisons the gradient even though the forward value is fine
   (genmlx-xpbm; prerequisite for the 5x3f differentiable-planning path)."
  [b L]
  (let [raw    (mx/multiply b L)
        z      (mx/sum raw)
        ok     (mx/greater z eps)
        z-safe (mx/where ok z (mx/scalar 1.0))]
    (mx/where ok (mx/divide raw z-safe) b)))

(defn tensor-update-belief
  "One Bayes filtering step as pure tensor ops. `b` is a [W] MLX belief aligned to
   `worlds`; returns the updated [W] MLX belief. o = nil (uninformative cell) is an
   explicit identity fast-path that bit-matches the host short-circuit and skips
   building L (every world is consistent with nil, so L would be all-ones anyway)."
  [observe worlds b loc o]
  (if (nil? o)
    b
    (filter-step b (obs-likelihood-vec observe worlds loc o))))

;; -- belief <-> vector seam (host map/vector callers <-> [W] MLX) --------------

(defn belief->vec
  "{world -> prob} map  ->  [W] MLX float32 aligned to `worlds` (missing -> 0.0)."
  [worlds belief]
  (mx/array (clj->js (mapv (fn [w] (double (get belief w 0.0))) worlds)) mx/float32))

(defn vec->belief
  "[W] MLX (or clj vector) aligned to `worlds`  ->  {world -> prob} map."
  [worlds v]
  (zipmap worlds (if (mx/array? v) (mx/->clj v) v)))

(defn update-belief-map
  "Map-in / map-out drop-in for pomdp.cljs's host `:update-belief`: routes a
   {world -> prob} belief through the tensor kernel. Used as the agent's
   `:update-belief-tensor`."
  [observe worlds belief loc o]
  (vec->belief worlds (tensor-update-belief observe worlds (belief->vec worlds belief) loc o)))

(defn update-belief-vec
  "Clj-vector-in / clj-vector-out drop-in for biased_planners' `:update-belief`
   (a prob vector aligned to `worlds`): routes through the tensor kernel."
  [observe worlds bvec loc o]
  (let [b (mx/array (clj->js (mapv double bvec)) mx/float32)]
    (vec (mx/->clj (tensor-update-belief observe worlds b loc o)))))
