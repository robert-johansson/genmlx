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

(cljs.test/run-tests)
