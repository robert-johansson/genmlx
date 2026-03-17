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
            [genmlx.combinators :as comb]
            [genmlx.selection :as sel]
            [genmlx.inference.importance :as is]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.smc :as smc]
            [genmlx.inference.vi :as vi]
            [genmlx.inference.util :as u]
            [genmlx.mlx.random :as rng]
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

;; --- Selection helper ---

(defn make-selection [sel-spec]
  (case (get sel-spec "type")
    "addrs" (apply sel/select (map keyword (get sel-spec "addrs")))
    "all"   sel/all
    "none"  sel/none))

;; --- Project ---

(defn eval-project [spec]
  (let [model-name (get spec "model")
        model      (get model-lookup model-name)
        raw-args   (get spec "args" [])
        args       (make-args model-name raw-args)
        cm         (make-choicemap (get spec "choices"))
        sel        (make-selection (get spec "selection"))]
    (try
      (let [gen-result (p/generate model args cm)
            trace      (:trace gen-result)
            weight     (mx/item (p/project model trace sel))
            score      (mx/item (:score trace))]
        #js {"id" (get spec "id") "weight" weight "score" score})
      (catch :default e
        #js {"id" (get spec "id") "error" (str e)}))))

;; --- Regenerate ---

(defn eval-regenerate [spec]
  (let [model-name (get spec "model")
        model      (get model-lookup model-name)
        raw-args   (get spec "args" [])
        args       (make-args model-name raw-args)
        cm         (make-choicemap (get spec "choices"))
        sel        (make-selection (get spec "selection"))]
    (try
      (let [gen-result   (p/generate model args cm)
            trace        (:trace gen-result)
            old-score    (mx/item (:score trace))
            regen-result (p/regenerate model trace sel)
            new-score    (mx/item (:score (:trace regen-result)))
            weight       (mx/item (:weight regen-result))]
        #js {"id" (get spec "id") "old_score" old-score "new_score" new-score "weight" weight})
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

;; --- Combinator models ---

(def map-kernel
  (gen [x]
    (trace :y (dist/gaussian (mx/scalar x) (mx/scalar 1)))))

(def map-model (dyn/auto-key (comb/map-combinator (dyn/auto-key map-kernel))))

(def unfold-kernel
  (gen [t state]
    (let [x (trace :x (dist/gaussian (mx/scalar state) (mx/scalar 1)))]
      x)))

(def unfold-model (dyn/auto-key (comb/unfold-combinator (dyn/auto-key unfold-kernel))))

(def switch-branch-a
  (dyn/auto-key (gen [] (trace :x (dist/gaussian (mx/scalar 0) (mx/scalar 1))))))

(def switch-branch-b
  (dyn/auto-key (gen [] (trace :x (dist/gaussian (mx/scalar 10) (mx/scalar 1))))))

(def switch-model (comb/switch-combinator switch-branch-a switch-branch-b))

(defn make-hierarchical-choicemap
  "Convert {\"0\": {\"y\": 1.5}, \"1\": {\"y\": 2.5}} to hierarchical choicemap."
  [choices-map]
  (reduce-kv
   (fn [cm idx-str sub-map]
     (let [idx (js/parseInt idx-str)]
       (reduce-kv
        (fn [cm2 addr-str val]
          (cm/set-choice cm2 [idx (keyword addr-str)] (mx/scalar val)))
        cm sub-map)))
   (cm/choicemap)
   choices-map))

(defn make-combinator-args [comb-type raw-args]
  (case comb-type
    "map"    [(vec (first raw-args))]
    "unfold" [(int (first raw-args)) (second raw-args)]
    "switch" [(int (first raw-args))]))

(defn get-combinator-model [comb-type]
  (case comb-type
    "map"    map-model
    "unfold" unfold-model
    "switch" switch-model))

(defn make-combinator-choicemap [comb-type choices-map]
  (case comb-type
    ;; Switch choices are flat (not hierarchical)
    "switch" (make-choicemap choices-map)
    ;; Map and Unfold use hierarchical addressing
    (make-hierarchical-choicemap choices-map)))

(defn eval-combinator [spec]
  (let [comb-type (get spec "combinator_type")
        operation (get spec "operation")
        raw-args  (get spec "args")
        model     (get-combinator-model comb-type)
        args      (make-combinator-args comb-type raw-args)]
    (try
      (case operation
        "assess"
        (let [cm     (make-combinator-choicemap comb-type (get spec "choices"))
              result (p/assess model args cm)]
          #js {"id" (get spec "id") "weight" (mx/item (:weight result))})

        "score_decomposition"
        (let [cm         (make-combinator-choicemap comb-type (get spec "choices"))
              result     (p/assess model args cm)
              total      (mx/item (:weight result))
              components (get spec "expected_components")
              comp-scores
              (mapv (fn [comp]
                      (let [d  (case (get comp "dist")
                                 "normal" (dist/gaussian (mx/scalar (get-in comp ["params" "mu"]))
                                                         (mx/scalar (get-in comp ["params" "sigma"]))))
                            lp (mx/item (dc/dist-log-prob d (mx/scalar (get comp "value"))))]
                        {"index" (get comp "index") "logprob" lp}))
                    components)
              sum-comp (reduce + (map #(get % "logprob") comp-scores))]
          #js {"id"             (get spec "id")
               "total_score"    total
               "components"     (clj->js comp-scores)
               "sum_components" sum-comp})

        "generate"
        (let [constraints (make-combinator-choicemap comb-type (get spec "constraints"))
              result      (p/generate model args constraints)
              weight      (mx/item (:weight result))
              score       (mx/item (:score (:trace result)))]
          #js {"id" (get spec "id") "weight" weight "score" score})

        "update"
        (let [init-cm  (make-combinator-choicemap comb-type (get spec "initial_choices"))
              upd-cm   (make-combinator-choicemap comb-type (get spec "update_choices"))
              gen-r    (p/generate model args init-cm)
              old-tr   (:trace gen-r)
              old-score (mx/item (:score old-tr))
              upd-r    (p/update model old-tr upd-cm)
              new-score (mx/item (:score (:trace upd-r)))
              weight   (mx/item (:weight upd-r))]
          #js {"id" (get spec "id") "old_score" old-score "new_score" new-score "weight" weight}))
      (catch :default e
        #js {"id" (get spec "id") "error" (str e)}))))

;; --- Gradient dispatch ---

(defn make-dist-from-spec
  "Create a distribution from a dist name + params map (all MLX scalars)."
  [dist-name params]
  (case dist-name
    "normal"      (dist/gaussian (mx/scalar (get params "mu"))
                                 (mx/scalar (get params "sigma")))
    "beta"        (dist/beta-dist (mx/scalar (get params "alpha"))
                                   (mx/scalar (get params "beta")))
    "gamma"       (dist/gamma-dist (mx/scalar (get params "shape"))
                                    (mx/scalar (get params "rate")))
    "exponential" (dist/exponential (mx/scalar (get params "rate")))
    "laplace"     (dist/laplace (mx/scalar (get params "loc"))
                                 (mx/scalar (get params "scale")))
    "cauchy"      (dist/cauchy (mx/scalar (get params "loc"))
                                (mx/scalar (get params "scale")))
    "lognormal"   (dist/log-normal (mx/scalar (get params "mu"))
                                    (mx/scalar (get params "sigma")))
    "student_t"   (dist/student-t (mx/scalar (get params "df"))
                                   (mx/scalar (get params "loc"))
                                   (mx/scalar (get params "scale")))
    (throw (js/Error. (str "unsupported gradient dist: " dist-name)))))

(defn eval-gradient [spec]
  (let [dist-name (get spec "dist")
        params    (get spec "params")
        grad-wrt  (get spec "grad_wrt")
        value     (get spec "value")]
    (try
      (let [grad-val
            (case grad-wrt
              "value"
              (let [grad-fn (mx/grad
                              (fn [v]
                                (dc/dist-log-prob
                                 (make-dist-from-spec dist-name params) v)))]
                (mx/item (grad-fn (mx/scalar value))))

              "mu"
              (let [grad-fn (mx/grad
                              (fn [mu]
                                (dc/dist-log-prob
                                 (case dist-name
                                   "normal"   (dist/gaussian mu (mx/scalar (get params "sigma")))
                                   "lognormal" (dist/log-normal mu (mx/scalar (get params "sigma"))))
                                 (mx/scalar value))))]
                (mx/item (grad-fn (mx/scalar (get params "mu")))))

              "sigma"
              (let [grad-fn (mx/grad
                              (fn [sigma]
                                (dc/dist-log-prob
                                 (case dist-name
                                   "normal"   (dist/gaussian (mx/scalar (get params "mu")) sigma)
                                   "lognormal" (dist/log-normal (mx/scalar (get params "mu")) sigma))
                                 (mx/scalar value))))]
                (mx/item (grad-fn (mx/scalar (get params "sigma")))))

              "alpha"
              (let [grad-fn (mx/grad
                              (fn [alpha]
                                (dc/dist-log-prob
                                 (dist/beta-dist alpha (mx/scalar (get params "beta")))
                                 (mx/scalar value))))]
                (mx/item (grad-fn (mx/scalar (get params "alpha")))))

              "shape"
              (let [grad-fn (mx/grad
                              (fn [shape]
                                (dc/dist-log-prob
                                 (dist/gamma-dist shape (mx/scalar (get params "rate")))
                                 (mx/scalar value))))]
                (mx/item (grad-fn (mx/scalar (get params "shape")))))

              (throw (js/Error. (str "unsupported grad_wrt: " grad-wrt))))]
        #js {"id" (get spec "id") "gradient" grad-val})
      (catch :default e
        #js {"id" (get spec "id") "error" (str e)}))))

;; --- Inference quality models ---

(def nn-inference-model
  (dyn/auto-key
    (gen [observations]
      (let [mu (trace :mu (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
        (doseq [[i obs] (map-indexed vector observations)]
          (trace (keyword (str "x" i)) (dist/gaussian mu (mx/scalar 1))))
        mu))))

(def bb-inference-model
  (dyn/auto-key
    (gen [observations]
      (let [p (trace :p (dist/beta-dist (mx/scalar 1) (mx/scalar 1)))]
        (doseq [[i obs] (map-indexed vector observations)]
          (trace (keyword (str "x" i)) (dist/bernoulli p)))
        p))))

(def linreg-inference-model
  (dyn/auto-key
    (gen [xs]
      (let [slope (trace :slope (dist/gaussian (mx/scalar 0) (mx/scalar 10)))]
        (doseq [[i x] (map-indexed vector xs)]
          (trace (keyword (str "y" i)) (dist/gaussian (mx/multiply slope (mx/scalar x)) (mx/scalar 1))))
        slope))))

(def gp-inference-model
  (dyn/auto-key
    (gen [observations]
      (let [lambda (trace :lambda (dist/gamma-dist (mx/scalar 2) (mx/scalar 1)))]
        (doseq [[i obs] (map-indexed vector observations)]
          (trace (keyword (str "x" i)) (dist/poisson lambda)))
        lambda))))

(def dc-inference-model
  (dyn/auto-key
    (gen [observations]
      (let [p (trace :p (dist/dirichlet (mx/array #js [1 1 1])))]
        (doseq [[i obs] (map-indexed vector observations)]
          (trace (keyword (str "x" i)) (dist/categorical (mx/log p))))
        p))))

(def inference-model-lookup
  {"normal_normal"           nn-inference-model
   "beta_bernoulli_iid"      bb-inference-model
   "normal_linreg"           linreg-inference-model
   "gamma_poisson"           gp-inference-model
   "dirichlet_categorical"   dc-inference-model})

(defn make-inference-observations
  "Build observation choicemap from spec data and model type."
  [model-name data]
  (case model-name
    "normal_normal"
    (let [obs (get data "observations")]
      (reduce (fn [cm [i v]]
                (cm/set-value cm (keyword (str "x" i)) (mx/scalar v)))
              (cm/choicemap) (map-indexed vector obs)))

    "beta_bernoulli_iid"
    (let [obs (get data "observations")]
      (reduce (fn [cm [i v]]
                (cm/set-value cm (keyword (str "x" i)) (mx/scalar v)))
              (cm/choicemap) (map-indexed vector obs)))

    "normal_linreg"
    (let [ys (get data "ys")]
      (reduce (fn [cm [i v]]
                (cm/set-value cm (keyword (str "y" i)) (mx/scalar v)))
              (cm/choicemap) (map-indexed vector ys)))

    "gamma_poisson"
    (let [obs (get data "observations")]
      (reduce (fn [cm [i v]]
                (cm/set-value cm (keyword (str "x" i)) (mx/scalar (int v) mx/int32)))
              (cm/choicemap) (map-indexed vector obs)))

    "dirichlet_categorical"
    (let [obs (get data "observations")]
      (reduce (fn [cm [i v]]
                (cm/set-value cm (keyword (str "x" i)) (mx/scalar (int v) mx/int32)))
              (cm/choicemap) (map-indexed vector obs)))))

(defn make-inference-args
  "Build args vector for inference model."
  [model-name data]
  (case model-name
    "normal_normal"          [(get data "observations")]
    "beta_bernoulli_iid"     [(get data "observations")]
    "normal_linreg"          [(get data "xs")]
    "gamma_poisson"          [(get data "observations")]
    "dirichlet_categorical"  [(get data "observations")]))

(defn extract-trace-value
  "Extract a scalar value from a trace at target-addr.
   For vector-valued traces (e.g. Dirichlet), extracts target-component."
  [t target-addr target-component]
  (let [v (cm/get-value (cm/get-submap (:choices t) target-addr))]
    (mx/eval! v)
    (if target-component
      ;; Vector-valued: extract component
      (let [arr (.tolist v)]
        (aget arr target-component))
      (mx/item v))))

(defn eval-inference [spec]
  (let [model-name       (get spec "model")
        model            (get inference-model-lookup model-name)
        algorithm        (get spec "algorithm")
        algo-params      (get spec "algorithm_params")
        data             (get spec "data")
        target-addr      (keyword (get spec "target_addr"))
        target-component (get spec "target_component")
        comparison       (get spec "comparison")
        args             (make-inference-args model-name data)
        obs-cm           (make-inference-observations model-name data)]
    (try
      (let [result
            (case algorithm
              "importance_sampling"
              (let [n-particles (get algo-params "n_particles" 1000)
                    {:keys [traces log-weights log-ml-estimate]}
                    (is/importance-sampling {:samples n-particles} model args obs-cm)
                    ;; Compute weighted mean
                    raw-weights (mapv (fn [w] (mx/eval! w) (mx/item w)) log-weights)
                    max-w (apply max raw-weights)
                    exp-weights (mapv #(js/Math.exp (- % max-w)) raw-weights)
                    sum-w (reduce + exp-weights)
                    norm-weights (mapv #(/ % sum-w) exp-weights)
                    vals (mapv (fn [t] (extract-trace-value t target-addr target-component))
                               traces)]
                {:mean (reduce + (map * vals norm-weights))
                 :accept-rate nil
                 :log-ml (mx/item log-ml-estimate)})

              "mh"
              (let [n-steps (get algo-params "n_steps" 2000)
                    burn    (get algo-params "burn" 500)
                    traces  (mcmc/mh {:samples (- n-steps burn) :burn burn
                                      :selection (sel/select target-addr)}
                                     model args obs-cm)
                    vals    (mapv (fn [t] (extract-trace-value t target-addr target-component))
                                  traces)]
                {:mean (/ (reduce + vals) (count vals))
                 :accept-rate (:acceptance-rate (meta traces))})

              "hmc"
              (let [n-steps  (get algo-params "n_steps" 500)
                    burn     (get algo-params "burn" 200)
                    L        (get algo-params "leapfrog_steps" 10)
                    eps      (get algo-params "step_size" 0.01)
                    samples  (mcmc/hmc {:samples n-steps :burn burn
                                        :leapfrog-steps L :step-size eps
                                        :addresses [target-addr]
                                        :compile? false}
                                       model args obs-cm)
                    ;; HMC returns vectors — extract first element for single param
                    vals     (mapv (fn [s] (if (vector? s) (first s) s)) samples)]
                {:mean (/ (reduce + vals) (count vals))
                 :accept-rate (:acceptance-rate (meta samples))})

              "mala"
              (let [n-steps  (get algo-params "n_steps" 500)
                    burn     (get algo-params "burn" 200)
                    eps      (get algo-params "step_size" 0.01)
                    samples  (mcmc/mala {:samples n-steps :burn burn
                                         :step-size eps
                                         :addresses [target-addr]
                                         :compile? false}
                                        model args obs-cm)
                    ;; MALA returns vectors — extract first element for single param
                    vals     (mapv (fn [s] (if (vector? s) (first s) s)) samples)]
                {:mean (/ (reduce + vals) (count vals))
                 :accept-rate (:acceptance-rate (meta samples))}))]
        (cond-> #js {"id" (get spec "id")
                     "posterior_mean" (:mean result)
                     "acceptance_rate" (:accept-rate result)}
          ;; Include log-ML for log_ml and log_ml_analytical comparison tests
          (and (contains? #{"log_ml" "log_ml_analytical"} comparison)
               (:log-ml result))
          (doto (aset "log_ml" (:log-ml result)))))
      (catch :default e
        #js {"id" (get spec "id") "error" (str e)}))))

;; --- SSM model for SMC tests ---

(def ssm-kernel
  (dyn/auto-key
    (gen [t state]
      (let [x (trace :x (dist/gaussian (mx/scalar state) (mx/scalar 1)))]
        (trace :y (dist/gaussian x (mx/scalar 0.5)))
        x))))

(defn run-smc-ssm
  "Run SMC on linear-Gaussian SSM using smc-unfold."
  [observations n-particles]
  (let [obs-seq (mapv (fn [y]
                        (cm/set-value (cm/choicemap) :y (mx/scalar y)))
                      observations)
        result (smc/smc-unfold {:particles n-particles} ssm-kernel 0.0 obs-seq)
        log-ml (mx/item (:log-ml result))
        final-ess (:final-ess result)]
    {:log-ml log-ml :ess final-ess}))

(defn run-smc-single
  "Single-step SMC (equivalent to IS) for normal-normal model."
  [model args obs-cm n-particles]
  (let [{:keys [traces log-weights]}
        (is/importance-sampling {:samples n-particles} model args obs-cm)
        raw-weights (mapv (fn [w] (mx/eval! w) (mx/item w)) log-weights)
        max-w (apply max raw-weights)
        log-ml (+ max-w (js/Math.log (/ (reduce + (mapv #(js/Math.exp (- % max-w)) raw-weights))
                                         n-particles)))]
    {:log-ml log-ml}))

(defn eval-inference-smc [spec]
  (let [algorithm   (get spec "algorithm")
        algo-params (get spec "algorithm_params")
        data        (get spec "data")
        comparison  (get spec "comparison")]
    (try
      (case algorithm
        "smc"
        (let [obs (get data "observations")
              n-particles (get algo-params "n_particles" 500)
              {:keys [log-ml ess]} (run-smc-ssm obs n-particles)]
          (case comparison
            "log_ml"
            #js {"id" (get spec "id") "log_ml" log-ml}

            "ess"
            #js {"id" (get spec "id") "ess" ess "log_ml" log-ml}

            "analytical_log_ml"
            #js {"id" (get spec "id") "log_ml" log-ml}))

        "smc_single"
        (let [model-name (get spec "model")
              model      (get inference-model-lookup model-name)
              args       (make-inference-args model-name data)
              obs-cm     (make-inference-observations model-name data)
              n-particles (get algo-params "n_particles" 500)
              {:keys [log-ml]} (run-smc-single model args obs-cm n-particles)]
          #js {"id" (get spec "id") "log_ml" log-ml}))
      (catch :default e
        #js {"id" (get spec "id") "error" (str e)}))))

;; --- VI eval ---

(defn eval-inference-vi [spec]
  (let [model-name  (get spec "model")
        model       (get inference-model-lookup model-name)
        algo-params (get spec "algorithm_params")
        data        (get spec "data")
        target-addr (keyword (get spec "target_addr"))
        args        (make-inference-args model-name data)
        obs-cm      (make-inference-observations model-name data)
        iterations  (get algo-params "iterations" 500)
        lr          (get algo-params "learning_rate" 0.01)]
    (try
      (let [result (vi/vi-from-model {:iterations iterations :learning-rate lr}
                                      model args obs-cm [target-addr])
            mu-val (mx/item (:mu result))]
        #js {"id" (get spec "id") "posterior_mean" mu-val})
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
                "project"              (mapv eval-project  (get input-data "project_tests"))
                "regenerate"           (mapv eval-regenerate (get input-data "regenerate_tests"))
                "combinator"           (mapv eval-combinator (get input-data "combinator_tests"))
                "mlx_ops"              (mapv (comp sanitize-js-obj eval-mlx-op) (get input-data "tests"))
                "gradient"             (mapv eval-gradient (get input-data "gradient_tests"))
                "inference_quality"    (mapv (fn [spec]
                                               (let [algo (get spec "algorithm")]
                                                 (cond
                                                   (contains? #{"smc" "smc_single"} algo)
                                                   (eval-inference-smc spec)

                                                   (= algo "vi")
                                                   (eval-inference-vi spec)

                                                   :else
                                                   (eval-inference spec))))
                                             (get input-data "tests"))
                (do (.error js/console (str "Unknown test type: " test-type))
                    (js/process.exit 1)))
      output  #js {"system"    "genmlx"
                   "test_type" test-type
                   "results"   (clj->js results)}]
  (println (js/JSON.stringify output nil 2)))
