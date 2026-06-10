;; @tier fast core
(ns genmlx.combinator-regen-weight-test
  "genmlx-v740: combinator regenerate weights must stay correct when the
   per-step/per-element score metadata (::step-scores / ::element-scores) is
   lost — which happens on every regenerate THROUGH a splice (execute-sub
   reconstructs the child trace without it) and after serialize round-trips.

   Oracles (independent of the code under test):
   1. Empty selection => weight must be exactly 0 (nothing resampled, all
      values kept, same args => identical scores, zero proposal ratio).
      Pre-fix, metadata loss made this S_total (the entire old joint score).
   2. Single-site selection through a splice => weight must equal the
      hand-computed local density ratio from the trace values."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.combinators :as comb]
            [genmlx.diff :as diff]
            [genmlx.vectorized :as vz]
            [genmlx.vmap :as vmap]
            [genmlx.mlx.random :as rng])
  (:require-macros [genmlx.gen :refer [gen]]))

(defn- num [x] (mx/eval! x) (mx/item x))

(defn- strip-meta [trace] (with-meta trace {}))

(def select-nothing (sel/select :no-such-address))

;; ── Unfold ────────────────────────────────────────────────────────────────

(def hmm-kernel
  (gen [t prev]
       (let [x (trace :x (dist/gaussian prev 1))]
         (trace :y (dist/gaussian x 0.5))
         x)))

(def hmm-unfold (comb/unfold-combinator hmm-kernel))

(deftest unfold-empty-selection-weight-zero-after-meta-loss
  (let [tr (p/simulate (dyn/with-key hmm-unfold (rng/fresh-key 3))
                       [3 (mx/scalar 0.0)])]
    (testing "with metadata intact"
      (let [{w :weight} (p/regenerate hmm-unfold tr select-nothing)]
        (is (< (js/Math.abs (num w)) 1e-5) "empty selection => weight 0")))
    (testing "with metadata stripped (splice/serialize loss)"
      (let [{w :weight} (p/regenerate hmm-unfold (strip-meta tr) select-nothing)]
        (is (< (js/Math.abs (num w)) 1e-4)
            (str "empty selection => weight 0 even without ::step-scores, got "
                 (num w)))))))

;; ── Through a splice (the SBC unfold-hmm repro) ───────────────────────────

(def spliced-model
  (dyn/auto-key (gen []
                     (splice :chain hmm-unfold 3 (mx/scalar 0.0)))))

(defn- chain-val [choices t addr]
  (num (cm/get-value (reduce cm/get-submap choices [:chain t addr]))))

(deftest spliced-unfold-empty-selection-weight-zero
  (let [tr0 (p/simulate spliced-model [])
        obs (reduce (fn [o t]
                      (cm/set-choice o [:chain t :y]
                                     (mx/scalar (chain-val (:choices tr0) t :y))))
                    cm/EMPTY (range 3))
        {tr :trace} (p/generate spliced-model [] obs)
        {w :weight} (p/regenerate spliced-model tr
                                  (sel/hierarchical :chain select-nothing))]
    (is (< (js/Math.abs (num w)) 1e-4)
        (str "regenerate through splice, empty selection => weight 0, got "
             (num w)))))

(deftest spliced-unfold-single-site-weight-matches-local-ratio
  ;; Regenerate ONLY [:chain 1 :x]. The exact MH weight for a prior-proposal
  ;; single-site regenerate is the local density ratio:
  ;;   logp(y1|x1') - logp(y1|x1) + logp(x2|x1') - logp(x2|x1)
  ;; (the x1-prior terms cancel against the proposal).
  (let [tr0 (p/simulate spliced-model [])
        obs (reduce (fn [o t]
                      (cm/set-choice o [:chain t :y]
                                     (mx/scalar (chain-val (:choices tr0) t :y))))
                    cm/EMPTY (range 3))
        {tr :trace} (p/generate spliced-model [] obs)
        x1 (chain-val (:choices tr) 1 :x)
        x2 (chain-val (:choices tr) 2 :x)
        y1 (chain-val (:choices tr) 1 :y)
        {tr' :trace w :weight}
        (p/regenerate spliced-model tr
                      (sel/hierarchical :chain (sel/hierarchical 1 (sel/select :x))))
        x1' (chain-val (:choices tr') 1 :x)
        lp (fn [d v] (num (dist/log-prob d (mx/scalar v))))
        expected (+ (- (lp (dist/gaussian (mx/scalar x1') 0.5) y1)
                       (lp (dist/gaussian (mx/scalar x1) 0.5) y1))
                    (- (lp (dist/gaussian (mx/scalar x1') 1) x2)
                       (lp (dist/gaussian (mx/scalar x1) 1) x2)))]
    (is (not= x1 x1') "x1 was resampled")
    (is (< (js/Math.abs (chain-val (:choices tr') 2 :x) ) 1e9) "sanity")
    (is (< (js/Math.abs (- (num w) expected)) 1e-3)
        (str "spliced single-site regenerate weight " (num w)
             " == local density ratio " expected))))

;; ── Map ───────────────────────────────────────────────────────────────────

(def map-kernel
  (gen [x]
       (let [m (trace :m (dist/gaussian x 1))]
         (trace :o (dist/gaussian m 0.5))
         m)))

(def mapped (comb/map-combinator map-kernel))

(deftest map-empty-selection-weight-zero-after-meta-loss
  (let [tr (p/simulate (dyn/with-key mapped (rng/fresh-key 11))
                       [[(mx/scalar 0.0) (mx/scalar 1.0) (mx/scalar 2.0)]])]
    (testing "metadata intact"
      (let [{w :weight} (p/regenerate mapped tr select-nothing)]
        (is (< (js/Math.abs (num w)) 1e-5))))
    (testing "metadata stripped"
      (let [{w :weight} (p/regenerate mapped (strip-meta tr) select-nothing)]
        (is (< (js/Math.abs (num w)) 1e-4)
            (str "Map: empty selection => weight 0 without ::element-scores, got "
                 (num w)))))))

;; ── Scan ──────────────────────────────────────────────────────────────────

(def scan-kernel
  (gen [carry input]
       (let [s (trace :s (dist/gaussian carry 1))]
         (trace :o (dist/gaussian s 0.5))
         [s s])))

(def scanned (comb/scan-combinator scan-kernel))

(deftest scan-empty-selection-weight-zero-after-meta-loss
  (let [tr (p/simulate (dyn/with-key scanned (rng/fresh-key 17))
                       [(mx/scalar 0.0)
                        [(mx/scalar 0.0) (mx/scalar 1.0) (mx/scalar 2.0)]])]
    (testing "metadata intact"
      (let [{w :weight} (p/regenerate scanned tr select-nothing)]
        (is (< (js/Math.abs (num w)) 1e-5))))
    (testing "metadata stripped"
      (let [{w :weight} (p/regenerate scanned (strip-meta tr) select-nothing)]
        (is (< (js/Math.abs (num w)) 1e-4)
            (str "Scan: empty selection => weight 0 without ::step-scores, got "
                 (num w)))))))

;; ── Map update discard indexing (genmlx-v740 item 1) ─────────────────────
;; Constraining ONLY element 2 must record the displaced old value under
;; discard index [2] — positional reassembly after a filter put it at [0].

(deftest map-update-discard-keeps-element-index
  (let [tr (p/simulate (dyn/with-key mapped (rng/fresh-key 23))
                       [[(mx/scalar 0.0) (mx/scalar 1.0) (mx/scalar 2.0)]])
        old-m2 (num (cm/get-value (reduce cm/get-submap (:choices tr) [2 :m])))
        {d :discard} (p/update mapped tr (cm/set-choice cm/EMPTY [2 :m]
                                                        (mx/scalar 9.0)))]
    (testing "discard lands under the original element index"
      (is (cm/has-value? (reduce cm/get-submap d [2 :m]))
          "displaced value recorded under [2]")
      (is (= cm/EMPTY (cm/get-submap d 0))
          "nothing recorded under [0]")
      (is (< (js/Math.abs (- (num (cm/get-value (reduce cm/get-submap d [2 :m])))
                             old-m2))
             1e-6)
          "discard holds the displaced old value"))))

(deftest map-update-with-diffs-discard-keeps-element-index
  (let [xs [(mx/scalar 0.0) (mx/scalar 1.0) (mx/scalar 2.0)]
        tr (p/simulate (dyn/with-key mapped (rng/fresh-key 29)) [xs])
        old-m2 (num (cm/get-value (reduce cm/get-submap (:choices tr) [2 :m])))
        {d :discard} (p/update-with-diffs mapped tr
                                          (cm/set-choice cm/EMPTY [2 :m]
                                                         (mx/scalar 9.0))
                                          diff/no-change)]
    (testing "update-with-diffs discard lands under the original index"
      (is (cm/has-value? (reduce cm/get-submap d [2 :m]))
          "displaced value recorded under [2]")
      (is (= cm/EMPTY (cm/get-submap d 0))
          "nothing recorded under [0]")
      (is (< (js/Math.abs (- (num (cm/get-value (reduce cm/get-submap d [2 :m])))
                             old-m2))
             1e-6)))))

;; ── Vectorized retval pairing (genmlx-v740 item 4) ───────────────────────
;; resample-vtrace and merge-vtraces-by-mask must permute/merge :retval with
;; the same indices/mask as the choices — otherwise particle i's choices
;; pair with ancestor j's return value.

(def identity-model
  ;; retval IS the traced value, so retval[i] must equal choices :x [i]
  ;; after any permutation/merge.
  (gen []
       (trace :x (dist/gaussian 0 1))))

(deftest resample-vtrace-permutes-retval
  (let [n 8
        vt (dyn/vsimulate identity-model [] n (rng/fresh-key 31))
        ;; weight mass concentrated on particle 0 => resample maps all to it
        vt (assoc vt :weight (mx/array (into [100.0] (repeat (dec n) -100.0))))
        rs (vz/resample-vtrace vt (rng/fresh-key 32))
        xs (mx/->clj (cm/get-value (cm/get-submap (:choices rs) :x)))
        rv (mx/->clj (:retval rs))]
    (is (= (count (set (mapv #(.toFixed % 5) xs))) 1)
        "all particles resampled to the dominant ancestor")
    (is (every? true? (mapv #(< (js/Math.abs (- %1 %2)) 1e-6) xs rv))
        "retval permuted with the same ancestor indices as choices")))

(deftest merge-vtraces-by-mask-merges-retval
  (let [n 4
        cur (dyn/vsimulate identity-model [] n (rng/fresh-key 41))
        prop (dyn/vsimulate identity-model [] n (rng/fresh-key 42))
        mask (mx/greater (mx/array [1.0 0.0 1.0 0.0]) (mx/scalar 0.5))
        merged (vz/merge-vtraces-by-mask cur prop mask)
        xs (mx/->clj (cm/get-value (cm/get-submap (:choices merged) :x)))
        rv (mx/->clj (:retval merged))]
    (is (every? true? (mapv #(< (js/Math.abs (- %1 %2)) 1e-6) xs rv))
        "retval merged with the same mask as choices")))

;; ── vmap fast path param-store (genmlx-v740 item 5) ──────────────────────
;; The batched fast path must see the kernel's trained params; pre-fix it
;; ran the body with param DEFAULTS.

(def param-kernel
  (gen []
       (let [mu (param :mu 0.0)]
         (trace :x (dist/gaussian mu 0.01)))))

(deftest vmap-fast-path-sees-param-store
  (let [store {:params {:mu (mx/scalar 50.0)} :version 0}
        vmapped (vmap/vmap-gf
                 (vary-meta param-kernel assoc
                            :genmlx.dynamic/param-store store)
                 :axis-size 4)
        tr (p/simulate (dyn/with-key vmapped (rng/fresh-key 51)) [])
        xs (mx/->clj (cm/get-value (cm/get-submap (:choices tr) :x)))]
    (is (every? #(> % 40.0) xs)
        (str "fast-path samples centered on the TRAINED mu=50, got " xs))))

;; ── Mix :component-idx regenerate weight (genmlx-v740 item 6) ────────────
;; Selecting :component-idx resamples the ENTIRE Mix subtree from the
;; prior, so the regenerate MH weight is exactly 0 (score delta cancels
;; the prior-proposal ratio). Pre-fix it returned the raw score delta.

(def mix-gf
  (comb/mix-combinator
   [(gen [] (trace :v (dist/gaussian -3 0.5)))
    (gen [] (trace :v (dist/gaussian 3 0.5)))]
   (mx/array [(js/Math.log 0.5) (js/Math.log 0.5)])))

(deftest mix-idx-selected-regenerate-weight-zero
  (let [tr (p/simulate (dyn/with-key mix-gf (rng/fresh-key 61)) [])]
    (dotimes [i 5]
      (let [{w :weight} (p/regenerate mix-gf tr (sel/select :component-idx))]
        (is (< (js/Math.abs (num w)) 1e-5)
            (str "prior-resample regenerate weight must be 0, got " (num w)
                 " (attempt " i ")"))))))

(cljs.test/run-tests)
