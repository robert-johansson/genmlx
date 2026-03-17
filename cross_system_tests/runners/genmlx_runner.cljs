(ns genmlx.cross-system-runner
  "GenMLX cross-system verification runner.
   Reads JSON from stdin, writes results to stdout."
  (:require [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.gen :refer [gen]]
            [genmlx.choicemap :as cm]
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
      (let [v  (mx/scalar value)
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

(def model-lookup
  {"single_normal"      single-normal-model
   "two_normals"        two-normals-model
   "beta_bernoulli"     beta-bernoulli-model
   "linear_regression"  linear-regression-model})

(defn make-choicemap [choices-map]
  (reduce-kv
   (fn [cm k v]
     (cm/set-value cm (keyword k) (mx/scalar v)))
   (cm/choicemap)
   choices-map))

(defn eval-assess [spec]
  (let [model-name (get spec "model")
        model      (get model-lookup model-name)
        raw-args   (get spec "args" [])
        args       (if (= model-name "linear_regression")
                     [(js->clj (first raw-args))]
                     [])
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
        args        (if (= model-name "linear_regression")
                      [(js->clj (first raw-args))]
                      [])
        constraints (make-choicemap (get spec "constraints"))]
    (try
      (let [result (p/generate model args constraints)
            weight (mx/item (:weight result))
            score  (mx/item (:score (:trace result)))]
        #js {"id" (get spec "id") "weight" weight "score" score})
      (catch :default e
        #js {"id" (get spec "id") "error" (str e)}))))

;; --- MLX ops dispatch ---

(defn to-mlx
  "Convert JSON value to MLX array — scalar, vector, or matrix."
  [v]
  (cond
    (number? v)                          (mx/scalar v)
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
              "det"       (let [L (mx/cholesky (to-mlx (first args)))
                                d (mx/diag L)]
                            (mx/item (mx/power (mx/prod d) (mx/scalar 2))))
              "logdet"    (let [L (mx/cholesky (to-mlx (first args)))
                                d (mx/diag L)]
                            (mx/item (mx/multiply (mx/scalar 2) (mx/sum (mx/log d)))))
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
                "logprob"  (mapv eval-logprob  (get input-data "tests"))
                "assess"   (mapv eval-assess   (get input-data "assess_tests"))
                "generate" (mapv eval-generate (get input-data "generate_tests"))
                "mlx_ops"  (mapv (comp sanitize-js-obj eval-mlx-op) (get input-data "tests"))
                (do (.error js/console (str "Unknown test type: " test-type))
                    (js/process.exit 1)))
      output  #js {"system"    "genmlx"
                   "test_type" test-type
                   "results"   (clj->js results)}]
  (println (js/JSON.stringify output nil 2)))
