;; @tier fast core
(ns genmlx.gdn-scan-contract-test
  "Membrane contract for mx/gated-delta-scan (genmlx-ps8a) — the fused
   chunk-parallel GDN-prefill primitive (mlx-core gated_delta_chunked_ops,
   BT=64 WY form, the native CUDA prefill algorithm).

   The reference is the token-serial per-step recurrence — the same math as
   qwen35_forward's gdn-layer inner reduce (and Rust gated_delta_step),
   generalized over batch. The two paths differ only in reduction order, so
   parity is tolerance-based: max-abs-diff < 1e-2 at production dims (the
   Rust unit-test bound, gated_delta.rs). The Rust parity tests cannot run
   on Thor/CUDA (genmlx-gr51 exit-SIGABRT), so THIS file is the parity gate
   on this host.

   Pins, per the spec on bean genmlx-ps8a:
   - parity across T ∈ {1 2 48 64 65 127 128 256} (sub-chunk / exact /
     off-by-one / multi-chunk zero-padding cases)
   - the strong-decay regime stays finite (the log-space g_log contract:
     callers pass -exp(A_log)*softplus(a+dt_bias) DIRECTLY, never log(g))
   - chain law: two chained T=64 scans ≡ one T=128 scan (state carry)
   - B=2 smoke (batch-ready for genmlx-9uyg)
   - bit-determinism (scan twice → identical; no gather_mm inside)
   - MLX_GDN_KERNEL env has NO effect on the export (it bypasses routing)
   - shape validation rejects malformed inputs with a friendly error"
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]))

;; Production dims (Ornith 35B GDN): Hv=32, Dk=Dv=128.
(def ^:private Hv 32)
(def ^:private Dk 128)
(def ^:private Dv 128)

(defn- slice-t
  "Contiguous [start, stop) slice along axis 1 (mx/slice is axis-0 only)."
  [a start stop]
  (mx/take-idx a (mx/arange start stop) 1))

(defn- per-step-ref
  "Token-serial gated-delta recurrence — mirrors qwen35_forward.cljs
   gdn-layer's inner reduce / Rust gated_delta_step, generalized over B.
   `gg` is the EXP-space gate. Returns [y state']."
  [q k v gg beta state]
  (let [[b T _ _] (mx/shape q)
        [st outs]
        (reduce
         (fn [[st outs] t]
           (let [qt (mx/squeeze (slice-t q t (inc t)) [1])    ; [B Hv Dk]
                 kt (mx/squeeze (slice-t k t (inc t)) [1])    ; [B Hv Dk]
                 vt (mx/squeeze (slice-t v t (inc t)) [1])    ; [B Hv Dv]
                 gt (mx/squeeze (slice-t gg t (inc t)) [1])   ; [B Hv]
                 bt (mx/squeeze (slice-t beta t (inc t)) [1]) ; [B Hv]
                 st (mx/multiply st (mx/reshape gt [b Hv 1 1]))
                 k4 (mx/reshape kt [b Hv 1 Dk])
                 kv-mem (mx/sum (mx/multiply st k4) [3] false)      ; [B Hv Dv]
                 delta  (mx/multiply (mx/subtract vt kv-mem)
                                     (mx/reshape bt [b Hv 1]))
                 st (mx/add st (mx/multiply k4 (mx/reshape delta [b Hv Dv 1])))
                 q4 (mx/reshape qt [b Hv 1 Dk])
                 yt (mx/sum (mx/multiply st q4) [3] false)]          ; [B Hv Dv]
             [st (conj outs (mx/reshape yt [b 1 Hv Dv]))]))
         [state []] (range T))]
    [(mx/concatenate outs 1) st]))

(defn- rand-inputs
  "Well-conditioned random GDN inputs at production dims: q/k carry the
   upstream Dk^-0.5 RMS-norm scaling (k.k ~ O(0.1), the WY system stays
   well-conditioned — same regime as the Rust parity test)."
  [seed b T]
  (let [[k1 k2 k3 k4 k5] (rng/split-n (h/deterministic-key seed) 5)
        qk-scale (/ 1.0 (js/Math.sqrt Dk))
        q    (mx/multiply (rng/normal k1 [b T Hv Dk]) qk-scale)
        k    (mx/multiply (rng/normal k2 [b T Hv Dk]) qk-scale)
        v    (rng/normal k3 [b T Hv Dv])
        gg   (mx/sigmoid (rng/normal k4 [b T Hv]))   ; g in (0,1), no underflow
        beta (mx/sigmoid (rng/normal k5 [b T Hv]))
        st0  (mx/zeros [b Hv Dv Dk] mx/float32)]
    {:q q :k k :v v :gg gg :g-log (mx/log gg) :beta beta :state st0}))

(defn- max-abs-diff [a b]
  (mx/item (mx/amax (mx/abs (mx/subtract a b)))))

(def ^:private TOL 1e-2)  ;; the Rust chunked-vs-per-step bound at these dims

;; ---------------------------------------------------------------------------

(deftest parity-across-chunk-boundaries
  (doseq [T [1 2 48 64 65 127 128 256]]
    (testing (str "T=" T " parity vs per-step reference")
      (let [{:keys [q k v gg g-log beta state]} (rand-inputs (+ 100 T) 1 T)
            [y-ref s-ref] (per-step-ref q k v gg beta state)
            [y-new s-new] (mx/gated-delta-scan q k v g-log beta state)]
        (is (= [1 T Hv Dv] (mx/shape y-new)) (str "T=" T " y shape"))
        (is (= [1 Hv Dv Dk] (mx/shape s-new)) (str "T=" T " state shape"))
        (is (< (max-abs-diff y-ref y-new) TOL) (str "T=" T " output parity"))
        (is (< (max-abs-diff s-ref s-new) TOL) (str "T=" T " state parity"))))))

(deftest strong-decay-stays-finite
  ;; The MoE regime that motivates the log-space contract: decay strong
  ;; enough that exp-space g underflows f32 to 0 (g-log ~ -100). Passing
  ;; g-log DIRECTLY must stay finite and match per-step (whose gg is exactly
  ;; 0 here); log(exp-space g) would be -inf and NaN the in-chunk decay-diff.
  (let [T 128
        {:keys [q k v beta state]} (rand-inputs 7 1 T)
        g-log (mx/full [1 T Hv] -100.0)
        gg    (mx/exp g-log)                      ; underflows to exactly 0
        [y-ref _] (per-step-ref q k v gg beta state)
        [y-new s-new] (mx/gated-delta-scan q k v g-log beta state)]
    (is (h/finite? (mx/item (mx/amax (mx/abs y-new)))) "y finite under strong decay")
    (is (h/finite? (mx/item (mx/amax (mx/abs s-new)))) "state finite under strong decay")
    (is (< (max-abs-diff y-ref y-new) TOL) "strong-decay output parity")))

(deftest chain-law-state-carry
  ;; Two chained T=64 scans must equal one T=128 scan: the state carried out
  ;; of the first call is a full substitute for having seen the prefix.
  (let [{:keys [q k v g-log beta state]} (rand-inputs 11 1 128)
        [y-full s-full] (mx/gated-delta-scan q k v g-log beta state)
        half (fn [a] [(slice-t a 0 64) (slice-t a 64 128)])
        [qa qb] (half q) [ka kb] (half k) [va vb] (half v)
        [ga gb] (half g-log) [ba bb] (half beta)
        [ya s-mid] (mx/gated-delta-scan qa ka va ga ba state)
        [yb s-end] (mx/gated-delta-scan qb kb vb gb bb s-mid)
        y-chained (mx/concatenate [ya yb] 1)]
    (is (< (max-abs-diff y-full y-chained) TOL) "chained outputs ≡ whole scan")
    (is (< (max-abs-diff s-full s-end) TOL) "chained final state ≡ whole scan")))

(deftest batch-dim-smoke
  ;; B=2: the primitive is batch-ready by construction (genmlx-9uyg). Each
  ;; batch row must independently match the per-step reference.
  (let [T 96
        {:keys [q k v gg g-log beta state]} (rand-inputs 13 2 T)
        [y-ref s-ref] (per-step-ref q k v gg beta state)
        [y-new s-new] (mx/gated-delta-scan q k v g-log beta state)]
    (is (= [2 T Hv Dv] (mx/shape y-new)) "B=2 y shape")
    (is (< (max-abs-diff y-ref y-new) TOL) "B=2 output parity")
    (is (< (max-abs-diff s-ref s-new) TOL) "B=2 state parity")))

(deftest bit-determinism
  ;; Same inputs twice → bit-identical y and state (no gather_mm inside;
  ;; preserves the dense-9B reproducibility-vehicle property).
  (let [{:keys [q k v g-log beta state]} (rand-inputs 17 1 128)
        [y1 s1] (mx/gated-delta-scan q k v g-log beta state)
        [y2 s2] (mx/gated-delta-scan q k v g-log beta state)]
    (is (zero? (max-abs-diff y1 y2)) "y bit-identical across runs")
    (is (zero? (max-abs-diff s1 s2)) "state bit-identical across runs")))

(deftest env-var-independence
  ;; MLX_GDN_KERNEL steers the NATIVE forward's routing; the membrane export
  ;; calls chunked_ops directly and must not observe it.
  (let [{:keys [q k v g-log beta state]} (rand-inputs 19 1 96)
        [y-plain _] (mx/gated-delta-scan q k v g-log beta state)
        env (.-env js/process)
        _ (set! (.-MLX_GDN_KERNEL env) "perstep")
        [y-env _] (try (mx/gated-delta-scan q k v g-log beta state)
                       (finally (js-delete env "MLX_GDN_KERNEL")))]
    (is (zero? (max-abs-diff y-plain y-env))
        "MLX_GDN_KERNEL=perstep does not change the export's result")))

(deftest shape-validation
  (let [{:keys [q k v g-log beta state]} (rand-inputs 23 1 64)]
    (is (thrown-with-msg? js/Error #"gated_delta_scan"
          (mx/gated-delta-scan q k v (mx/expand-dims g-log 3) beta state))
        "4-d g_log rejected")
    (is (thrown-with-msg? js/Error #"gated_delta_scan"
          (mx/gated-delta-scan q k v g-log beta
                               (mx/zeros [1 Hv Dv (dec Dk)] mx/float32)))
        "state with wrong Dk rejected")))

(cljs.test/run-tests)
