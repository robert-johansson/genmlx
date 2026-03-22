(ns genmlx.trace-immutability-test
  "Trace record: field access, immutability, type invariants."
  (:require [cljs.test :refer [deftest is are testing]]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.test-helpers :as h]))

(defn sample-trace
  "Build a representative trace for testing."
  []
  (tr/make-trace {:gen-fn :test-model
                  :args [1 2 3]
                  :choices (cm/choicemap :x (mx/scalar 1.0) :y (mx/scalar 2.0))
                  :retval (mx/scalar 42.0)
                  :score (mx/scalar -1.8379)}))

;; ---------------------------------------------------------------------------
;; Field access via keyword lookup
;; ---------------------------------------------------------------------------

(deftest all-fields-accessible-via-keywords
  (let [{:keys [gen-fn args choices retval score]} (sample-trace)]
    (is (= :test-model gen-fn))
    (is (= [1 2 3] args))
    (is (some? choices))
    (is (some? retval))
    (is (some? score))))

(deftest trace-fields-match-construction-args
  (let [gf :my-model
        args [:a :b]
        choices (cm/choicemap :z 99)
        retval "result"
        score (mx/scalar -2.5)
        t (tr/make-trace {:gen-fn gf :args args :choices choices
                          :retval retval :score score})]
    (is (identical? gf (:gen-fn t)))
    (is (identical? args (:args t)))
    (is (identical? choices (:choices t)))
    (is (identical? retval (:retval t)))
    (is (identical? score (:score t)))))

;; ---------------------------------------------------------------------------
;; Score is MLX scalar
;; ---------------------------------------------------------------------------

(deftest score-is-mlx-scalar
  (let [{:keys [score]} (sample-trace)]
    (is (mx/array? score)
        "score must be an MLX array, not a JS number")
    (is (= [] (vec (mx/shape score)))
        "score must be scalar (0-dimensional)")))

(deftest score-realizes-to-expected-value
  (let [{:keys [score]} (sample-trace)]
    (is (< (js/Math.abs (- -1.8379 (h/realize score))) 1e-4)
        "score realizes to the value used at construction")))

;; ---------------------------------------------------------------------------
;; Choices is a valid ChoiceMap
;; ---------------------------------------------------------------------------

(deftest choices-is-choicemap-node
  (let [{:keys [choices]} (sample-trace)]
    (is (instance? cm/Node choices)
        "choices is a ChoiceMap Node")
    (is (not (cm/has-value? choices))
        "choices root is not a leaf")))

(deftest choices-addresses-are-accessible
  (let [{:keys [choices]} (sample-trace)
        addrs (cm/addresses choices)]
    (is (= #{[:x] [:y]} (set addrs)))
    (are [path] (some? (cm/get-choice choices path))
      [:x]
      [:y])))

(deftest choices-values-are-mlx-arrays
  (let [{:keys [choices]} (sample-trace)]
    (are [path expected]
         (< (js/Math.abs (- expected (h/realize (cm/get-choice choices path)))) 1e-6)
      [:x] 1.0
      [:y] 2.0)))

;; ---------------------------------------------------------------------------
;; Trace is a record (supports assoc without mutation)
;; ---------------------------------------------------------------------------

(deftest trace-supports-non-destructive-assoc
  (let [t1 (sample-trace)
        t2 (assoc t1 :retval :modified)]
    (is (= :modified (:retval t2))
        "new trace has modified retval")
    (is (not= :modified (:retval t1))
        "original trace is unaffected")))

(deftest trace-supports-non-destructive-score-update
  (let [t1 (sample-trace)
        new-s (mx/scalar -999.0)
        t2 (assoc t1 :score new-s)]
    (is (identical? new-s (:score t2)))
    (is (not (identical? new-s (:score t1)))
        "original trace retains its score")))

;; ---------------------------------------------------------------------------
;; make-trace from map
;; ---------------------------------------------------------------------------

(deftest make-trace-from-partial-map
  (testing "make-trace with nil fields"
    (let [t (tr/make-trace {:gen-fn :m :args [] :choices cm/EMPTY
                            :retval nil :score nil})]
      (is (nil? (:retval t)))
      (is (nil? (:score t))))))

(deftest make-trace-produces-trace-record
  (is (instance? tr/Trace (sample-trace))))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(cljs.test/run-tests)
