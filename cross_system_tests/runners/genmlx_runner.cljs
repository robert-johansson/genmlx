(ns genmlx.cross-system-runner
  "GenMLX cross-system verification runner.
   Reads JSON from stdin, writes results to stdout."
  (:require [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.gen :refer [gen]]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            ["fs" :as fs]))

;; --- Read JSON from stdin ---

(def input-data
  (js->clj (js/JSON.parse (.toString (fs/readFileSync "/dev/stdin" "utf8")))))

(def test-type (get input-data "test_type"))

;; --- Distribution log-prob dispatch ---

(defn eval-logprob [spec]
  (let [dist-name (get spec "dist")
        value     (get spec "value")
        params    (get spec "params")]
    (try
      (let [v  (when (number? value) (mx/scalar value))
            lp (case dist-name
                 "normal"
                 (let [d (dist/gaussian (mx/scalar (get params "mu"))
                                        (mx/scalar (get params "sigma")))]
                   (mx/item (dist/log-prob d v)))

                 "uniform"
                 (let [d (dist/uniform (mx/scalar (get params "lo"))
                                       (mx/scalar (get params "hi")))]
                   (mx/item (dist/log-prob d v)))

                 "bernoulli"
                 (let [d (dist/bernoulli (mx/scalar (get params "p")))]
                   (mx/item (dist/log-prob d (mx/scalar value))))

                 "beta"
                 (let [d (dist/beta-dist (mx/scalar (get params "alpha"))
                                         (mx/scalar (get params "beta")))]
                   (mx/item (dist/log-prob d v)))

                 "gamma"
                 (let [d (dist/gamma-dist (mx/scalar (get params "shape"))
                                          (mx/scalar (get params "rate")))]
                   (mx/item (dist/log-prob d v)))

                 "exponential"
                 (let [d (dist/exponential (mx/scalar (get params "rate")))]
                   (mx/item (dist/log-prob d v)))

                 "laplace"
                 (let [d (dist/laplace (mx/scalar (get params "loc"))
                                       (mx/scalar (get params "scale")))]
                   (mx/item (dist/log-prob d v)))

                 "cauchy"
                 (let [d (dist/cauchy (mx/scalar (get params "loc"))
                                      (mx/scalar (get params "scale")))]
                   (mx/item (dist/log-prob d v)))

                 "poisson"
                 (let [d (dist/poisson (mx/scalar (get params "rate")))]
                   (mx/item (dist/log-prob d (mx/scalar value))))

                 "binomial"
                 (let [d (dist/binomial (mx/scalar (get params "n"))
                                        (mx/scalar (get params "p")))]
                   (mx/item (dist/log-prob d (mx/scalar value))))

                 "geometric"
                 (let [d (dist/geometric (mx/scalar (get params "p")))]
                   (mx/item (dist/log-prob d (mx/scalar value))))

                 "lognormal"
                 (let [d (dist/log-normal (mx/scalar (get params "mu"))
                                          (mx/scalar (get params "sigma")))]
                   (mx/item (dist/log-prob d v)))

                 "inv_gamma"
                 (let [d (dist/inv-gamma (mx/scalar (get params "shape"))
                                         (mx/scalar (get params "scale")))]
                   (mx/item (dist/log-prob d v)))

                 "student_t"
                 (let [d (dist/student-t (mx/scalar (get params "df"))
                                         (mx/scalar (get params "loc"))
                                         (mx/scalar (get params "scale")))]
                   (mx/item (dist/log-prob d v)))

                 "categorical"
                 (let [logits (mx/array (clj->js (get params "logits")))
                       d (dist/categorical logits)]
                   (mx/item (dc/dist-log-prob d (mx/scalar (int value) mx/int32))))

                 "dirichlet"
                 (let [alpha (mx/array (clj->js (get params "alpha")))
                       d (dist/dirichlet alpha)
                       val-arr (mx/array (clj->js value))]
                   (mx/item (dc/dist-log-prob d val-arr)))

                 "mvn"
                 (let [mu  (mx/array (clj->js (get params "mu")))
                       cov (mx/array (clj->js (get params "cov")))
                       d   (dist/multivariate-normal mu cov)
                       val-arr (mx/array (clj->js value))]
                   (mx/item (dc/dist-log-prob d val-arr)))

                 ;; default
                 (throw (js/Error. (str "unsupported dist: " dist-name))))]
        #js {"id" (get spec "id") "logprob" lp})
      (catch :default e
        #js {"id" (get spec "id") "error" (str e)}))))

;; --- GFI models ---

(def single-normal-model
  (gen []
    (let [x (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
      x)))

(def two-normals-model
  (gen []
    (let [mu (trace :mu (dist/gaussian (mx/scalar 0) (mx/scalar 10)))
          x  (trace :x  (dist/gaussian mu (mx/scalar 1)))]
      x)))

(def beta-bernoulli-model
  (gen []
    (let [p (trace :p (dist/beta-dist (mx/scalar 2) (mx/scalar 2)))
          x (trace :x (dist/bernoulli p))]
      x)))

(def linear-regression-model
  (gen [xs]
    (let [slope     (trace :slope     (dist/gaussian (mx/scalar 0) (mx/scalar 10)))
          intercept (trace :intercept (dist/gaussian (mx/scalar 0) (mx/scalar 10)))]
      (trace :y0 (dist/gaussian (mx/add (mx/multiply slope (mx/scalar (nth xs 0)))
                                        intercept) (mx/scalar 1)))
      (trace :y1 (dist/gaussian (mx/add (mx/multiply slope (mx/scalar (nth xs 1)))
                                        intercept) (mx/scalar 1)))
      (trace :y2 (dist/gaussian (mx/add (mx/multiply slope (mx/scalar (nth xs 2)))
                                        intercept) (mx/scalar 1)))
      slope)))

;; single_gaussian is the same model as single_normal (alias used in expanded specs)
(def single-gaussian-model single-normal-model)

(def mixed-model
  (gen []
    (let [coin (trace :coin (dist/bernoulli (mx/scalar 0.5)))]
      (if (pos? (mx/item coin))
        (trace :x (dist/gaussian (mx/scalar 10) (mx/scalar 1)))
        (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1)))))))

(def many-addresses-model
  (gen []
    (let [mu (trace :mu (dist/gaussian (mx/scalar 0) (mx/scalar 10)))]
      (doseq [i (range 10)]
        (trace (keyword (str "y" i)) (dist/gaussian mu (mx/scalar 1))))
      mu)))

(def linear-regression-5-model
  (gen [xs]
    (let [slope     (trace :slope     (dist/gaussian (mx/scalar 0) (mx/scalar 10)))
          intercept (trace :intercept (dist/gaussian (mx/scalar 0) (mx/scalar 10)))]
      (doseq [i (range 5)]
        (trace (keyword (str "y" i))
               (dist/gaussian (mx/add (mx/multiply slope (mx/scalar (nth xs i)))
                                      intercept)
                              (mx/scalar 1))))
      slope)))

(def model-lookup
  (into {}
    (map (fn [[k v]] [k (dyn/auto-key v)]))
    {"single_normal"         single-normal-model
     "single_gaussian"       single-gaussian-model
     "two_normals"           two-normals-model
     "beta_bernoulli"        beta-bernoulli-model
     "linear_regression"     linear-regression-model
     "linear_regression_5"   linear-regression-5-model
     "mixed"                 mixed-model
     "many_addresses"        many-addresses-model}))

(defn make-choicemap [choices-map]
  (reduce-kv
   (fn [cm k v]
     (cm/set-value cm (keyword k) (mx/scalar v)))
   (cm/choicemap)
   choices-map))

(defn make-args
  "Build the args vector for a model from the spec's args and model name.
   Falls back to default args when spec omits them for models that require args."
  [model-name raw-args]
  (if (seq raw-args)
    (case model-name
      ("linear_regression" "linear_regression_5")
      [(js->clj (first raw-args))]
      ;; default: pass through
      (vec raw-args))
    ;; No args in spec — use defaults for models that require them
    (case model-name
      "linear_regression"   [[1.0 2.0 3.0]]
      "linear_regression_5" [[1.0 2.0 3.0 4.0 5.0]]
      [])))

(defn eval-assess [spec]
  (let [model-name (get spec "model")
        model      (get model-lookup model-name)
        raw-args   (get spec "args" [])
        args       (make-args model-name raw-args)
        cm         (make-choicemap (get spec "choices"))]
    (try
      (let [result (p/assess model args cm)
            weight (mx/item (:weight result))]
        #js {"id" (get spec "id") "weight" weight})
      (catch :default e
        #js {"id" (get spec "id") "error" (str e)}))))

(defn eval-generate [spec]
  (let [model-name  (get spec "model")
        model       (get model-lookup model-name)
        raw-args    (get spec "args" [])
        args        (make-args model-name raw-args)
        constraints (make-choicemap (get spec "constraints"))]
    (try
      (let [result (p/generate model args constraints)
            weight (mx/item (:weight result))
            score  (mx/item (:score (:trace result)))]
        #js {"id" (get spec "id") "weight" weight "score" score})
      (catch :default e
        #js {"id" (get spec "id") "error" (str e)}))))

;; --- Score decomposition ---

(defn make-component-dist
  "Create a distribution from a component spec map."
  [comp-spec]
  (let [dist-name (get comp-spec "dist")
        params    (get comp-spec "params")]
    (case dist-name
      "normal"    (dist/gaussian (mx/scalar (get params "mu"))
                                 (mx/scalar (get params "sigma")))
      "beta"      (dist/beta-dist (mx/scalar (get params "alpha"))
                                   (mx/scalar (get params "beta")))
      "bernoulli" (dist/bernoulli (mx/scalar (get params "p")))
      (throw (js/Error. (str "unsupported component dist: " dist-name))))))

(defn eval-score-decomposition [spec]
  (let [model-name (get spec "model")
        model      (get model-lookup model-name)
        raw-args   (get spec "args" [])
        args       (make-args model-name raw-args)
        choices    (get spec "choices")
        cm         (make-choicemap choices)
        components (get spec "expected_components")]
    (try
      ;; Get total score via assess
      (let [result      (p/assess model args cm)
            total-score (mx/item (:weight result))
            ;; Compute per-component logprobs
            comp-scores
            (reduce-kv
             (fn [acc addr comp-spec]
               (let [d   (make-component-dist comp-spec)
                     v   (get choices addr)
                     lp  (mx/item (dc/dist-log-prob d (mx/scalar v)))]
                 (assoc acc addr lp)))
             {}
             components)]
        #js {"id"              (get spec "id")
             "total_score"    total-score
             "components"     (clj->js comp-scores)
             "sum_components" (reduce + (vals comp-scores))})
      (catch :default e
        #js {"id" (get spec "id") "error" (str e)}))))

;; --- Update ---

(defn eval-update [spec]
  (let [model-name (get spec "model")
        model      (get model-lookup model-name)
        raw-args   (get spec "args" [])
        args       (make-args model-name raw-args)
        init-cm    (make-choicemap (get spec "initial_choices"))
        upd-cm     (make-choicemap (get spec "update_constraints"))]
    (try
      ;; Step 1: create initial trace via fully-constrained generate
      (let [gen-result  (p/generate model args init-cm)
            init-trace  (:trace gen-result)
            old-score   (mx/item (:score init-trace))
            ;; Step 2: update trace with new constraints
            upd-result  (p/update model init-trace upd-cm)
            new-trace   (:trace upd-result)
            new-score   (mx/item (:score new-trace))
            weight      (mx/item (:weight upd-result))]
        #js {"id"        (get spec "id")
             "old_score" old-score
             "new_score" new-score
             "weight"    weight})
      (catch :default e
        #js {"id" (get spec "id") "error" (str e)}))))

;; --- MLX ops dispatch ---

(defn to-mlx
  "Convert JSON value to MLX array — scalar, vector, or matrix."
  [v]
  (cond
    (number? v)                          (mx/scalar v)
    (= v "NaN")                          (mx/scalar js/NaN)
    (= v "Inf")                          (mx/scalar js/Infinity)
    (= v "-Inf")                         (mx/scalar (- js/Infinity))
    (and (vector? v) (number? (first v))) (mx/array (clj->js v))
    (and (vector? v) (vector? (first v))) (mx/array (clj->js v))
    :else                                (mx/scalar v)))

(defn from-mlx
  "Convert MLX array back to JSON-safe value — scalar, vector, or nested vector."
  [a]
  (let [shape (js->clj (.-shape a))]
    (cond
      (= (count shape) 0) (mx/item a)
      (= (count shape) 1) (vec (js->clj (.tolist a)))
      :else (vec (map vec (js->clj (.tolist a)))))))

(defn eval-mlx-op [spec]
  (let [op   (get spec "op")
        args (get spec "args")]
    (try
      (let [result
            (case op
              "log"       (mx/item (mx/log (to-mlx (first args))))
              "exp"       (mx/item (mx/exp (to-mlx (first args))))
              "sqrt"      (mx/item (mx/sqrt (to-mlx (first args))))
              "abs"       (mx/item (mx/abs (to-mlx (first args))))
              "sin"       (mx/item (mx/sin (to-mlx (first args))))
              "cos"       (mx/item (mx/cos (to-mlx (first args))))
              "sigmoid"   (mx/item (mx/sigmoid (to-mlx (first args))))
              "lgamma"    (mx/item (mx/lgamma (to-mlx (first args))))
              "add"       (mx/item (mx/add (to-mlx (first args)) (to-mlx (second args))))
              "subtract"  (mx/item (mx/subtract (to-mlx (first args)) (to-mlx (second args))))
              "multiply"  (mx/item (mx/multiply (to-mlx (first args)) (to-mlx (second args))))
              "divide"    (mx/item (mx/divide (to-mlx (first args)) (to-mlx (second args))))
              "power"     (mx/item (mx/power (to-mlx (first args)) (to-mlx (second args))))
              "logsumexp" (mx/item (mx/logsumexp (to-mlx (first args))))
              "sum"       (mx/item (mx/sum (to-mlx (first args))))
              "mean"      (mx/item (mx/mean (to-mlx (first args))))
              "matmul"    (from-mlx (mx/matmul (to-mlx (first args)) (to-mlx (second args))))
              "cholesky"  (from-mlx (mx/cholesky (to-mlx (first args))))
              "det"       (mx/item (mx/det (to-mlx (first args))))
              "logdet"    (mx/item (mx/logdet (to-mlx (first args))))
              "trace"     (mx/item (mx/trace-mat (to-mlx (first args))))
              "nan_to_num" (mx/item (mx/nan-to-num (to-mlx (first args))))
              "cholesky_inv" (from-mlx (mx/cholesky-inv (mx/cholesky (to-mlx (first args)))))
              "logcumsumexp" (from-mlx (mx/logcumsumexp (to-mlx (first args))))
              "einsum"    (let [subscripts (first args)
                                arrays (mapv to-mlx (rest args))]
                            (from-mlx (apply mx/einsum subscripts arrays)))
              "slice"     (let [a     (to-mlx (first args))
                                start (second args)
                                stop  (nth args 2)]
                            (if (> (count args) 3)
                              (from-mlx (mx/slice a start stop (nth args 3)))
                              (from-mlx (mx/slice a start stop))))
              (throw (js/Error. (str "unsupported op: " op))))]
        #js {"id" (get spec "id") "result" (clj->js result)})
      (catch :default e
        #js {"id" (get spec "id") "error" (str e)}))))

;; --- JSON sanitization (JS JSON.stringify turns NaN/Infinity into null) ---

(defn sanitize-result [v]
  (cond
    (and (number? v) (js/isNaN v))             "NaN"
    (and (number? v) (not (js/isFinite v)) (pos? v)) "Inf"
    (and (number? v) (not (js/isFinite v)) (neg? v)) "-Inf"
    :else v))

(defn sanitize-js-obj [obj]
  (let [result (get (js->clj obj) "result")]
    (clj->js (assoc (js->clj obj) "result"
                     (if (sequential? result)
                       result
                       (sanitize-result result))))))

;; --- Main: dispatch on test_type, write stdout ---

(let [results (case test-type
                "logprob"              (mapv eval-logprob  (get input-data "tests"))
                "assess"               (mapv eval-assess   (get input-data "assess_tests"))
                "generate"             (mapv eval-generate (get input-data "generate_tests"))
                "score_decomposition"  (mapv eval-score-decomposition
                                             (get input-data "score_decomposition_tests"))
                "update"               (mapv eval-update   (get input-data "tests"))
                "mlx_ops"              (mapv (comp sanitize-js-obj eval-mlx-op) (get input-data "tests"))
                (do (.error js/console (str "Unknown test type: " test-type))
                    (js/process.exit 1)))
      output  #js {"system"    "genmlx"
                   "test_type" test-type
                   "results"   (clj->js results)}]
  (println (js/JSON.stringify output nil 2)))
