;; @tier fast core
(ns genmlx.vectorized-mask-merge-test
  "genmlx-3nme: vsmc rejuvenation merges accepted/rejected particles with an
   [N] boolean mask via (mx/where mask proposed current). For a VECTOR-valued
   latent leaf shaped [N,D] (dist/gaussian-vec / broadcasted-normal / dirichlet)
   MLX right-aligned broadcasting aligned the [N] mask against the trailing D
   axis: N != D crashed through NAPI; N == D silently broadcast to [N,N] and
   applied the accept/reject decision along the FEATURE axis (mixing components
   within a particle). The fix reshapes the [N] mask to [N,1,...] so it selects
   whole particles along the leading axis and broadcasts over latent dims —
   matching reindex-choicemap's take-idx gather semantics.

   ORACLE: a host-side per-particle select (row i from proposed if mask[i] else
   from current), independent of the code under test."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.mlx :as mx]
            [genmlx.choicemap :as cm]
            [genmlx.vectorized :as vz]))

(defn- ->clj [x] (mx/eval! x) (mx/->clj x))

(defn- mask-from [bits] (mx/greater (mx/array (mapv double bits)) (mx/scalar 0.5)))

(defn- host-merge
  "Reference: row i = proposed-rows[i] if bits[i] else current-rows[i]."
  [bits current-rows proposed-rows]
  (mapv (fn [b c p] (if (pos? b) p c)) bits current-rows proposed-rows))

(defn- leaf [merged addr] (->clj (cm/get-value (cm/get-submap merged addr))))

(defn- vtrace [choices score retval]
  {:choices choices :score score :retval retval :n-particles (first (mx/shape score))})

;; ── [N,D] vector latent, N != D : crashed before the fix ────────────────────

(deftest vector-latent-mask-merge-N-not-D
  (let [bits [1 0 1 0]                           ; N=4
        cur-rows [[0.0 0.0 0.0] [0.0 0.0 0.0] [0.0 0.0 0.0] [0.0 0.0 0.0]]  ; D=3
        prop-rows [[1.0 1.0 1.0] [1.0 1.0 1.0] [1.0 1.0 1.0] [1.0 1.0 1.0]]
        cur  (vtrace (cm/set-choice cm/EMPTY [:mu] (mx/array cur-rows))  (mx/array [0.0 0.0 0.0 0.0]) nil)
        prop (vtrace (cm/set-choice cm/EMPTY [:mu] (mx/array prop-rows)) (mx/array [1.0 1.0 1.0 1.0]) nil)
        merged (vz/merge-vtraces-by-mask cur prop (mask-from bits))]
    (testing "no NAPI broadcast crash, whole-particle selection"
      (is (= (host-merge bits cur-rows prop-rows) (leaf (:choices merged) :mu))
          "row i taken whole from proposed iff mask[i]")
      (is (= [4 3] (mx/shape (cm/get-value (cm/get-submap (:choices merged) :mu))))
          "leaf shape preserved [N,D]"))))

;; ── [N,D] vector latent, N == D : silently feature-mixed before the fix ─────

(deftest vector-latent-mask-merge-N-equals-D
  (let [bits [1 0 1]                              ; N=3
        cur-rows [[0.0 0.0 0.0] [0.0 0.0 0.0] [0.0 0.0 0.0]]  ; D=3
        prop-rows [[1.0 1.0 1.0] [1.0 1.0 1.0] [1.0 1.0 1.0]]
        cur  (vtrace (cm/set-choice cm/EMPTY [:mu] (mx/array cur-rows))  (mx/array [0.0 0.0 0.0]) nil)
        prop (vtrace (cm/set-choice cm/EMPTY [:mu] (mx/array prop-rows)) (mx/array [1.0 1.0 1.0]) nil)
        merged (vz/merge-vtraces-by-mask cur prop (mask-from bits))]
    (testing "no feature-axis mixing — each row is wholly current or proposed"
      (is (= [[1.0 1.0 1.0] [0.0 0.0 0.0] [1.0 1.0 1.0]] (leaf (:choices merged) :mu))
          "accept along the particle axis, not the feature axis"))))

;; ── retval (merge-state-by-mask) with [N,D] arrays ──────────────────────────

(deftest vector-retval-mask-merge
  (let [bits [0 1 1]
        cur-rows [[0.0 0.0] [0.0 0.0] [0.0 0.0]]
        prop-rows [[1.0 1.0] [1.0 1.0] [1.0 1.0]]
        cur  (vtrace cm/EMPTY (mx/array [0.0 0.0 0.0]) (mx/array cur-rows))
        prop (vtrace cm/EMPTY (mx/array [1.0 1.0 1.0]) (mx/array prop-rows))
        merged (vz/merge-vtraces-by-mask cur prop (mask-from bits))]
    (is (= (host-merge bits cur-rows prop-rows) (->clj (:retval merged)))
        "retval [N,D] merged whole-particle")))

;; ── scalar [N] latent : reshape is a no-op, behaviour unchanged ─────────────

(deftest scalar-latent-mask-merge-unchanged
  (let [bits [1 0 1 1]
        cur-v  [10.0 20.0 30.0 40.0]
        prop-v [11.0 21.0 31.0 41.0]
        cur  (vtrace (cm/set-choice cm/EMPTY [:x] (mx/array cur-v))  (mx/array [0.0 0.0 0.0 0.0]) nil)
        prop (vtrace (cm/set-choice cm/EMPTY [:x] (mx/array prop-v)) (mx/array [1.0 1.0 1.0 1.0]) nil)
        merged (vz/merge-vtraces-by-mask cur prop (mask-from bits))]
    (is (= [11.0 20.0 31.0 41.0] (leaf (:choices merged) :x))
        "scalar [N] leaf still selected element-wise by the mask")
    (is (= [1.0 0.0 1.0 1.0] (->clj (:score merged)))
        "[N] score selected by the same mask")))

(cljs.test/run-tests)
