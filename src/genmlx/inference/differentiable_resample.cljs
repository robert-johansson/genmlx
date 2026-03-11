(ns genmlx.inference.differentiable-resample
  "Differentiable resampling for compiled SMC.

   Two modes:
   - gumbel-top-k:     Hard resampling via Gumbel-top-k trick. Exact categorical
                        distribution, all-GPU, non-differentiable (argsort breaks grad).
                        Use for forward-pass correctness (Gate 3).
   - gumbel-softmax:   Soft resampling via Gumbel-softmax relaxation. Approximate,
                        fully differentiable through mx/grad. Temperature tau controls
                        bias-variance tradeoff. Use for gradient-through-SMC (Gate 4).

   Both modes consume pre-generated Gumbel(0,1) noise, enabling deterministic
   sweeps and compiled execution."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]))

;; =========================================================================
;; Gumbel noise generation
;; =========================================================================

(defn generate-gumbel-noise
  "Pre-generate Gumbel(0,1) noise for T resampling steps with N particles.
   Returns [T,N,N] MLX array — one [N,N] matrix per step.

   Each step needs [N,N] noise: N output particles, each drawing independently
   from categorical(log-weights) via the Gumbel-max trick.

   Gumbel(0,1) = -log(-log(U)) where U ~ Uniform(0,1).
   Clamps U to [1e-7, 1-1e-7] for numerical stability."
  [key T N]
  (let [uniforms (rng/uniform key [T N N])
        ;; Clamp to (eps, 1-eps) for numerical stability
        eps (mx/scalar 1e-7)
        clamped (mx/clip uniforms eps (mx/scalar (- 1.0 1e-7)))]
    (mx/negative (mx/log (mx/negative (mx/log clamped))))))

;; =========================================================================
;; Mode 1: Gumbel-top-k (hard, exact, non-differentiable)
;; =========================================================================

(defn gumbel-top-k
  "Hard resampling via Gumbel-max trick (with replacement).

   log-weights: [N] unnormalized log-weights
   gumbel-noise: [N,N] pre-generated Gumbel(0,1) noise

   Returns {:particles [N,K] :ancestors [N] int32}.

   Each output particle independently draws from categorical(log-weights):
   add independent [N] Gumbel noise to log-weights, take argmax.
   With [N,N] noise (one row per output particle), this gives N independent
   draws WITH REPLACEMENT — high-weight particles are duplicated, low-weight
   particles are eliminated.

   NOTE: argmax returns integer indices — no gradient flows through this op.
   Use gumbel-softmax for differentiable resampling."
  [particles log-weights gumbel-noise]
  (let [N (first (mx/shape log-weights))
        ;; Broadcast log-weights [N] → [1,N], add [N,N] noise → [N,N]
        perturbed (mx/add (mx/reshape log-weights [1 N]) gumbel-noise)
        ;; argmax per row → [N] indices WITH replacement
        ancestors (mx/astype (mx/argmax perturbed 1) mx/int32)
        ;; Gather particles by ancestor indices
        resampled (mx/take-idx particles ancestors 0)]
    {:particles resampled :ancestors ancestors}))

;; =========================================================================
;; Mode 2: Gumbel-softmax (soft, approximate, differentiable)
;; =========================================================================

(defn gumbel-softmax
  "Soft differentiable resampling via Gumbel-softmax relaxation.

   log-weights: [N] unnormalized log-weights
   gumbel-noise: [N,N] pre-generated Gumbel(0,1) noise
   tau: temperature scalar (lower = sharper ≈ hard resampling, higher = smoother)

   Returns {:particles [N,K]} where each output particle is a weighted
   combination of all input particles.

   Each output particle i gets independent perturbation:
     soft_weights[i] = softmax((log_w + gumbel[i,:]) / tau)
   Then: new_particles = soft_weights @ particles
   Shape: [N,N] @ [N,K] → [N,K]

   This is the differentiable analog of with-replacement resampling:
   each output particle independently draws soft weights.

   Memory: O(N²) for the [N,N] weight matrix. Fine for N≤1000,
   document limit for N>1000.

   Fully differentiable through mx/grad — gradient flows through softmax and matmul."
  [particles log-weights gumbel-noise tau]
  (let [N (first (mx/shape log-weights))
        ;; Broadcast log-weights [N] → [1,N], add [N,N] noise → [N,N]
        ;; Each row i: log_w + gumbel[i,:] — independent perturbation per output particle
        perturbed (mx/add (mx/reshape log-weights [1 N]) gumbel-noise)
        ;; Soft weights per output particle: softmax along axis 1 → [N,N]
        ;; Each row sums to 1
        weight-matrix (mx/softmax (mx/divide perturbed tau) 1)
        ;; Weighted combination: [N,N] @ [N,K] → [N,K]
        resampled (mx/matmul weight-matrix particles)]
    {:particles resampled}))

;; =========================================================================
;; Soft resampling (simpler alternative)
;; =========================================================================

(defn soft-resample
  "Simple soft resampling: convex combination of weighted and uniform resampling.

   log-weights: [N] unnormalized log-weights
   alpha: mixing coefficient in [0,1]. alpha=1 → pure categorical, alpha=0 → uniform.

   Returns {:particles [N,K]}.

   Differentiable. Lower variance than gumbel-softmax but always biased."
  [particles log-weights alpha]
  (let [N (first (mx/shape log-weights))
        ;; Categorical weights
        cat-weights (mx/softmax log-weights)
        ;; Uniform weights
        uniform-w (mx/divide (mx/ones [N]) (mx/scalar N))
        ;; Mix
        mixed (mx/add (mx/multiply (mx/scalar alpha) cat-weights)
                       (mx/multiply (mx/scalar (- 1.0 alpha)) uniform-w))
        ;; [N] → [N,N] → @ [N,K] → [N,K]
        weight-matrix (mx/broadcast-to (mx/reshape mixed [1 N]) [N N])
        resampled (mx/matmul weight-matrix particles)]
    {:particles resampled}))

;; =========================================================================
;; 1D convenience: soft resample for per-particle state
;; =========================================================================

(defn gumbel-softmax-1d
  "Gumbel-softmax for 1D values (e.g., per-particle state).

   values: [N] array
   log-weights: [N] unnormalized log-weights
   gumbel-noise: [N,N] pre-generated Gumbel(0,1) noise
   tau: temperature scalar

   Returns [N] array where each element is an independently weighted average.
   Equivalent to gumbel-softmax on [N,1] particles, squeezed back to [N]."
  [values log-weights gumbel-noise tau]
  (let [N (first (mx/shape values))
        ;; Reshape values [N] → [N,1] to use as single-column particles
        particles-2d (mx/reshape values [N 1])
        ;; Reuse full gumbel-softmax: [N,N] noise → [N,1] result
        {:keys [particles]} (gumbel-softmax particles-2d log-weights gumbel-noise tau)]
    ;; Reshape [N,1] → [N]
    (mx/reshape particles [N])))
