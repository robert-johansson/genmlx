(ns genmlx.gfi-property-test
  "Property-based GFI contract tests using test.check.
   Verifies that GFI invariants hold across random models and inputs."
  (:require [cljs.test :as t]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel])
  (:require-macros [genmlx.gen :refer [gen]]
                   [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- choice-val
  "Extract a JS number from a choicemap at addr, or nil."
  [choices addr]
  (let [sub (cm/get-submap choices addr)]
    (when (and sub (cm/has-value? sub))
      (let [v (cm/get-value sub)]
        (mx/eval! v)
        (mx/item v)))))

(defn- trace-score [trace]
  (mx/eval! (:score trace))
  (mx/item (:score trace)))

(defn- eval-weight [w]
  (mx/eval! w)
  (mx/item w))

(defn- finite? [x]
  (and (number? x) (js/isFinite x)))

(defn- close? [a b tol]
  (and (finite? a) (finite? b) (<= (js/Math.abs (- a b)) tol)))

;; ---------------------------------------------------------------------------
;; Model generators
;; ---------------------------------------------------------------------------

(def single-gaussian
  {:model (dyn/auto-key (gen []
            (let [x (trace :x (dist/gaussian 0 1))]
              (mx/eval! x) (mx/item x))))
   :args []
   :addrs #{:x}
   :label "single-gaussian"})

(def two-gaussian
  {:model (dyn/auto-key (gen []
            (let [x (trace :x (dist/gaussian 0 1))
                  y (trace :y (dist/gaussian 0 1))]
              (mx/eval! x y)
              (+ (mx/item x) (mx/item y)))))
   :args []
   :addrs #{:x :y}
   :label "two-gaussian"})

(def gaussian-chain
  {:model (dyn/auto-key (gen []
            (let [x (trace :x (dist/gaussian 0 1))]
              (mx/eval! x)
              (let [y (trace :y (dist/gaussian (mx/item x) 1))]
                (mx/eval! y)
                (mx/item y)))))
   :args []
   :addrs #{:x :y}
   :label "gaussian-chain"})

(def three-site
  {:model (dyn/auto-key (gen []
            (let [a (trace :a (dist/gaussian 0 2))
                  b (trace :b (dist/gaussian 0 2))
                  c (trace :c (dist/gaussian 0 2))]
              (mx/eval! a b c)
              (+ (mx/item a) (mx/item b) (mx/item c)))))
   :args []
   :addrs #{:a :b :c}
   :label "three-site"})

(def uniform-model
  {:model (dyn/auto-key (gen []
            (let [x (trace :x (dist/uniform 0 1))]
              (mx/eval! x) (mx/item x))))
   :args []
   :addrs #{:x}
   :label "uniform"})

(def mixed-continuous
  {:model (dyn/auto-key (gen []
            (let [x (trace :x (dist/gaussian 0 1))
                  y (trace :y (dist/exponential 1))]
              (mx/eval! x y)
              (+ (mx/item x) (mx/item y)))))
   :args []
   :addrs #{:x :y}
   :label "mixed-continuous"})

(def model-pool
  [single-gaussian two-gaussian gaussian-chain
   three-site uniform-model mixed-continuous])

(def gen-model
  "Generator that picks a random model from the pool."
  (gen/elements model-pool))

(def gen-scalar
  (gen/fmap #(mx/scalar %) (gen/double* {:min -10.0 :max 10.0
                                          :NaN? false :infinite? false})))

(defn gen-addr-subset [addrs]
  (let [addr-vec (vec addrs)]
    (gen/fmap set (gen/not-empty (gen/vector (gen/elements addr-vec))))))

;; ---------------------------------------------------------------------------
;; Properties
;; ---------------------------------------------------------------------------

(defspec simulate-all-addresses-present 50
  (prop/for-all [m gen-model]
    (let [trace (p/simulate (:model m) (:args m))]
      (every? #(some? (choice-val (:choices trace) %))
              (:addrs m)))))

(defspec simulate-gen-fn-round-trips 50
  (prop/for-all [m gen-model]
    (let [trace (p/simulate (:model m) (:args m))]
      (= (:gen-fn trace) (:model m)))))

(defspec generate-empty-weight-approx-zero 50
  (prop/for-all [m gen-model]
    (let [{:keys [weight]} (p/generate (:model m) (:args m) cm/EMPTY)
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(defspec generate-full-weight-approx-score 50
  (prop/for-all [m gen-model]
    (let [trace (p/simulate (:model m) (:args m))
          {:keys [trace weight]} (p/generate (:model m) (:args m)
                                             (:choices trace))
          w (eval-weight weight)
          s (trace-score trace)]
      (close? s w 0.01))))

(defspec assess-weight-approx-generate-score 50
  (prop/for-all [m gen-model]
    (let [trace (p/simulate (:model m) (:args m))
          choices (:choices trace)
          {:keys [weight]} (p/assess (:model m) (:args m) choices)
          assess-w (eval-weight weight)
          {:keys [trace]} (p/generate (:model m) (:args m) choices)
          gen-s (trace-score trace)]
      (close? assess-w gen-s 0.01))))

(defspec update-same-weight-approx-zero 50
  (prop/for-all [m gen-model]
    (let [trace (p/simulate (:model m) (:args m))
          {:keys [weight]} (p/update (:model m) trace (:choices trace))
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(defspec update-round-trip-via-discard 50
  (prop/for-all [m gen-model]
    (let [trace (p/simulate (:model m) (:args m))
          orig-score (trace-score trace)
          first-addr (first (:addrs m))
          constraint (cm/choicemap first-addr (mx/scalar 42.0))
          {:keys [trace discard]} (p/update (:model m) trace constraint)
          {:keys [trace]} (p/update (:model m) trace discard)
          recovered-score (trace-score trace)]
      (close? orig-score recovered-score 0.01))))

(defspec update-weight-approx-score-diff 50
  (prop/for-all [m gen-model]
    (let [trace (p/simulate (:model m) (:args m))
          old-score (trace-score trace)
          trace2 (p/simulate (:model m) (:args m))
          {:keys [trace weight]} (p/update (:model m) trace (:choices trace2))
          w (eval-weight weight)
          new-score (trace-score trace)]
      (close? w (- new-score old-score) 0.1))))

(defspec regenerate-none-weight-approx-zero 50
  (prop/for-all [m gen-model]
    (let [trace (p/simulate (:model m) (:args m))
          {:keys [weight]} (p/regenerate (:model m) trace sel/none)
          w (eval-weight weight)]
      (close? 0.0 w 0.01))))

(defspec regenerate-none-choices-preserved 50
  (prop/for-all [m gen-model]
    (let [orig-trace (p/simulate (:model m) (:args m))
          orig-vals (into {} (map (fn [a] [a (choice-val (:choices orig-trace) a)])
                                  (:addrs m)))
          {:keys [trace]} (p/regenerate (:model m) orig-trace sel/none)]
      (every? (fn [addr]
                (let [new-v (choice-val (:choices trace) addr)]
                  (close? (get orig-vals addr) new-v 1e-10)))
              (:addrs m)))))

(defspec regenerate-unselected-preserved 50
  (prop/for-all [m (gen/such-that #(> (count (:addrs %)) 1) gen-model)]
    (let [addrs-vec (vec (:addrs m))
          selected-addr (first addrs-vec)
          unselected (rest addrs-vec)
          trace (p/simulate (:model m) (:args m))
          orig-vals (into {} (map (fn [a] [a (choice-val (:choices trace) a)])
                                  unselected))
          {:keys [trace]} (p/regenerate (:model m) trace
                                        (sel/select selected-addr))]
      (every? (fn [a]
                (let [new-v (choice-val (:choices trace) a)]
                  (close? (get orig-vals a) new-v 1e-6)))
              unselected))))

(defspec project-all-approx-score 50
  (prop/for-all [m gen-model]
    (let [trace (p/simulate (:model m) (:args m))
          s (trace-score trace)
          proj (eval-weight (p/project (:model m) trace sel/all))]
      (close? s proj 0.01))))

(defspec project-none-approx-zero 50
  (prop/for-all [m gen-model]
    (let [trace (p/simulate (:model m) (:args m))
          proj (eval-weight (p/project (:model m) trace sel/none))]
      (close? 0.0 proj 0.01))))

(defspec project-S-plus-complement-S-approx-score 50
  (prop/for-all [m (gen/such-that #(> (count (:addrs %)) 1) gen-model)]
    (let [addrs-vec (vec (:addrs m))
          sel-addr (first addrs-vec)
          s (sel/select sel-addr)
          trace (p/simulate (:model m) (:args m))
          score (trace-score trace)
          proj-s (eval-weight (p/project (:model m) trace s))
          proj-cs (eval-weight (p/project (:model m) trace (sel/complement-sel s)))]
      (close? score (+ proj-s proj-cs) 0.1))))

(defspec propose-weight-approx-generate-weight 50
  (prop/for-all [m gen-model]
    (let [{:keys [choices weight]} (p/propose (:model m) (:args m))
          propose-w (eval-weight weight)
          {:keys [trace weight]} (p/generate (:model m) (:args m) choices)
          gen-w (eval-weight weight)]
      (close? propose-w gen-w 0.01))))

(defspec simulate-score-approx-assess-weight 50
  (prop/for-all [m gen-model]
    (let [trace (p/simulate (:model m) (:args m))
          s (trace-score trace)
          {:keys [weight]} (p/assess (:model m) (:args m) (:choices trace))
          w (eval-weight weight)]
      (close? s w 0.01))))

(t/run-tests)
