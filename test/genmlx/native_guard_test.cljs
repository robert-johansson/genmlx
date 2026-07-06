;; @tier fast core
(ns genmlx.native-guard-test
  "Regression guard for the native exception escape path (bean genmlx-8w48).

   MLX C++ exceptions (allocation failure under Metal buffer-count pressure,
   eager validation throws) must surface as CATCHABLE JS errors, never as
   process death. Before the guard sweep completed, an exception thrown inside
   a void/out-param shim (mlx_random_split — called on EVERY GenMLX sample)
   unwound through the extern \"C\" frame to std::terminate -> SIGTRAP,
   killing overnight SBC runs (conjugate-linreg-elim x smc,
   multivariate-regression-5d x hmc).

   These tests drive the same C++ frames with eager-validation throws (a
   malformed PRNG key, a 1-D matrix into QR), which exercise the exact same
   guard + null-out-param + check_handle path as the memory-pressure throws
   that cannot be triggered deterministically (the Metal resource limit is a
   device constant). If a guard regresses, these tests don't fail — the
   process dies, which the test runner reports as a crash."
  (:require [cljs.test :refer [deftest is]]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]))

(defonce ^:private core (js/require "@genmlx/core"))

(defn- caught-message
  "Run f, return the caught error message, or ::no-throw."
  [f]
  (try (f) ::no-throw
       (catch :default e (.-message e))))

(deftest random-split-malformed-key-is-catchable
  ;; rng::split validates key dtype/shape eagerly inside mlx_random_split.
  ;; Unguarded, this throw aborted the process (exit 133). The membrane's
  ;; rng/split pre-checks keys, so call the NAPI export directly.
  (let [msg (caught-message #(.randomSplit core (mx/scalar 1.5)))]
    (is (not= ::no-throw msg) "malformed key must throw, not return")
    (is (re-find #"\[bits\] Expected key type uint32" (str msg))
        "the MLX exception detail is surfaced in the JS error")))

(deftest linalg-qr-bad-input-is-catchable
  ;; Same escape-path class through the void out-param decomposition shims.
  ;; qr is a MODULE-level export ((.qr core a)), not an MxArray instance
  ;; method — the old call form threw SCI's "Could not find instance method"
  ;; before ever reaching the native guard, so this assertion never actually
  ;; exercised the escape path (genmlx-ne1q).
  (let [msg (caught-message #(.qr core (mx/array #js [1.0 2.0 3.0])))]
    (is (not= ::no-throw msg) "1-D input to QR must throw, not return")
    (is (re-find #"MLX error in qr_q" (str msg))
        "null out-param surfaces as a catchable napi error with context")))

(deftest native-error-channel-recovers
  ;; After a caught native error the thread-local error slot must be clear
  ;; and normal operation resume — a wedged channel would mislabel the next
  ;; unrelated call as failed.
  (caught-message #(.randomSplit core (mx/scalar 1.5)))
  (let [[k1 k2] (rng/split (rng/fresh-key))]
    (is (some? k1) "split works after a caught native error")
    (is (some? k2))
    (is (= [2] (mx/shape k1)) "k1 is a real [2] uint32 key")
    (is (= [2] (mx/shape k2)) "k2 is a real [2] uint32 key")))

(cljs.test/run-tests)
