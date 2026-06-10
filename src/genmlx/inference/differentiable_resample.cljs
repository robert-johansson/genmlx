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
    ;; Gumbel(0,1) = -log(-log(clamped))
    (-> clamped mx/log mx/negative mx/log mx/negative)))

(defn- perturb
  "Broadcast [N] log-weights to [1,N] and add the [N,N] Gumbel noise, giving a
   [N,N] perturbation matrix — one independent perturbed row per output particle."
  [log-weights gumbel-noise]
  (let [N (first (mx/shape log-weights))]
    (mx/add (mx/reshape log-weights [1 N]) gumbel-noise)))

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
  (let [perturbed (perturb log-weights gumbel-noise)
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
  (let [;; Each row i: log_w + gumbel[i,:] — independent perturbation per output particle
        perturbed (perturb log-weights gumbel-noise)
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
  "Soft resampling (Karkus et al. 2018, Particle Filter Networks): sample
   ancestor indices from the mixture q(i) = alpha·w_i + (1-alpha)/N, then
   importance-reweight each survivor by w_{a_k}/q(a_k). The hard gather is
   not differentiated; gradients flow through the RETURNED log-weights,
   whose ratio depends on the pre-resample weights — that is the method's
   point. alpha < 1 keeps q's support full so the ratio stays finite;
   the importance correction makes the scheme unbiased for alpha in (0,1].

   (The previous implementation broadcast one mixed row to all N output
   particles — every output was the same convex combination and the
   ensemble collapsed to its weighted mean; genmlx-7sqe.)

   particles: [N,K] array
   log-weights: [N] unnormalized log-weights
   alpha: mixing coefficient in (0,1]
   key: PRNG key for the categorical draw

   Returns {:particles [N,K] :log-weights [N] :indices [N]}."
  [particles log-weights alpha key]
  (let [N (first (mx/shape log-weights))
        ;; Normalized log categorical weights
        log-cat (mx/subtract log-weights (mx/logsumexp log-weights))
        ;; Mixture q(i) = alpha*w_i + (1-alpha)/N
        mix-w (mx/add (mx/multiply (mx/scalar alpha) (mx/exp log-cat))
                      (mx/scalar (/ (- 1.0 alpha) N)))
        log-mix (mx/log mix-w)
        ;; N independent draws from categorical(q)
        indices (rng/categorical (rng/ensure-key key)
                                 (mx/broadcast-to (mx/reshape log-mix [1 N])
                                                  [N N]))
        resampled (mx/take-idx particles indices)
        ;; Importance correction: log w'_k = log w_{a_k} - log q(a_k)
        new-log-w (mx/subtract (mx/take-idx log-cat indices)
                               (mx/take-idx log-mix indices))]
    {:particles resampled
     :log-weights new-log-w
     :indices indices}))

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
