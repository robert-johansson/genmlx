(ns examples.cljs-expr-demo
  "Level 1: Structural distribution over arithmetic s-expressions."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.combinators :as comb]
            [genmlx.choicemap :as cm]
            [genmlx.dynamic :as dyn]
            [genmlx.trace :as tr]
            [genmlx.selection :as sel]
            [genmlx.mlx.random :as rng]
            [sci.core :as sci])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ops '[+ - *])
(def max-depth 2)

(def arith-expr
  (comb/recurse
    (fn [self]
      (dyn/auto-key
        (gen [depth]
          (let [at-max? (>= depth max-depth)
                p-leaf (if at-max? 1.0 (+ 0.5 (* 0.15 depth)))
                is-leaf (trace :leaf (dist/bernoulli p-leaf))
                leaf? (pos? (mx/item is-leaf))]
            (if leaf?
              (let [v (trace :value (dist/categorical
                         (mx/array [0 0 0 0 0 0 0 0 0])))]
                (inc (long (mx/item v))))
              (let [op-idx (trace :op (dist/categorical
                              (mx/array [0 0 0])))
                    left (splice :left self (inc depth))
                    right (splice :right self (inc depth))]
                (list (nth ops (long (mx/item op-idx)))
                      left right)))))))))

(defn assert-true [msg pred]
  (println (if pred "  PASS:" "  FAIL:") msg))

(defn valid-expr? [e]
  (or (integer? e)
      (and (list? e)
           (= 3 (count e))
           (symbol? (first e))
           (contains? #{'+ '- '*} (first e))
           (valid-expr? (second e))
           (valid-expr? (nth e 2)))))

(defn expr-depth [e]
  (if (integer? e)
    0
    (inc (max (expr-depth (second e))
              (expr-depth (nth e 2))))))

(defn cleanup! []
  (mx/force-gc!)
  (mx/clear-cache!))

(println "\n=== P2-19 Level 1: Structural distribution over s-expressions ===\n")

;; Test 1: simulate
(println "-- simulate --")
(dotimes [i 5]
  (let [t (p/simulate arith-expr [0])
        expr (:retval t)
        score (mx/item (:score t))]
    (mx/materialize! (:score t))
    (println "  expr:" (pr-str expr) " depth:" (expr-depth expr) " score:" score)
    (assert-true (str "valid #" i) (valid-expr? expr)))
  (cleanup!))

;; Test 2: assess = score
(println "\n-- assess --")
(let [t (p/simulate arith-expr [0])
      choices (:choices t)
      score (mx/item (:score t))
      _ (cleanup!)
      {:keys [weight]} (p/assess arith-expr [0] choices)
      w (mx/item weight)]
  (println "  expr:" (pr-str (:retval t)))
  (println "  trace score:" score "  assess weight:" w)
  (assert-true "assess = score" (< (abs (- w score)) 0.001))
  (cleanup!))

;; Test 3: generate constrained to leaf
(println "\n-- generate (leaf) --")
(let [constraints (cm/choicemap :leaf (mx/scalar 1.0))
      {:keys [trace weight]} (p/generate arith-expr [0] constraints)
      expr (:retval trace)]
  (println "  expr:" (pr-str expr) " weight:" (mx/item weight))
  (assert-true "got a number" (integer? expr))
  (cleanup!))

;; Test 4: generate constrained to + op
(println "\n-- generate (+ op) --")
(let [constraints (cm/choicemap :leaf (mx/scalar 0.0)
                                :op (mx/scalar 0 mx/int32))
      {:keys [trace weight]} (p/generate arith-expr [0] constraints)
      expr (:retval trace)]
  (println "  expr:" (pr-str expr) " weight:" (mx/item weight))
  (assert-true "got + expr" (and (list? expr) (= '+ (first expr))))
  (cleanup!))

;; Test 5: update
(println "\n-- update --")
(let [t1 (p/simulate arith-expr [0])
      before (:retval t1)
      _ (cleanup!)
      new-constraints (cm/choicemap :leaf (mx/scalar 1.0)
                                    :value (mx/scalar 5 mx/int32))
      {:keys [trace weight]} (p/update arith-expr t1 new-constraints)
      after (:retval trace)]
  (println "  before:" (pr-str before) "  after:" (pr-str after)
           " weight:" (mx/item weight))
  (assert-true "updated to 6" (= 6 after))
  (cleanup!))

;; Test 6: regenerate
(println "\n-- regenerate --")
(let [t (p/simulate arith-expr [0])
      before (:retval t)
      _ (cleanup!)
      sel sel/all
      {:keys [trace weight]} (p/regenerate arith-expr t sel)
      after (:retval trace)]
  (println "  before:" (pr-str before) "  after:" (pr-str after)
           " weight:" (mx/item weight))
  (cleanup!))

;; Test 7: distribution check
(println "\n-- distribution (20 samples) --")
(let [samples (vec (for [_ (range 20)]
                     (let [expr (:retval (p/simulate arith-expr [0]))]
                       (cleanup!)
                       expr)))
      leaves (count (filter integer? samples))
      exprs (count (filter list? samples))]
  (println "  leaves:" leaves "  exprs:" exprs)
  (assert-true "mix of leaves and expressions" (and (pos? leaves) (pos? exprs))))

(println "\n=== Level 1 complete ===")

;; ============================================================
;; Level 2: SCI as likelihood — find expressions that evaluate to target
;; ============================================================

(println "\n=== P2-19 Level 2: SCI as likelihood ===\n")

(defn eval-expr [e]
  (try (sci/eval-string (pr-str e))
       (catch :default _ nil)))

;; Wrap arith-expr in a model that checks evaluation result
(def arith-task
  (dyn/auto-key
    (gen [target]
      (let [expr (splice :program arith-expr 0)
            result (eval-expr expr)
            correct (and (number? result) (== result target))]
        (trace :correct (dist/bernoulli (if correct 0.999 0.001)))
        expr))))

;; Test: simulate produces expressions (no constraint)
(println "-- unconstrained simulate --")
(dotimes [_ 3]
  (let [t (p/simulate arith-task [42])
        expr (:retval t)
        result (eval-expr expr)]
    (println "  expr:" (pr-str expr) "= " result)
    (cleanup!)))

;; Test: generate with :correct constrained to 1 (true)
(println "\n-- constrained search (target=10, 200 attempts) --")
(let [target 10
      found (atom [])
      attempts 200]
  (dotimes [_ attempts]
    (try
      (let [constraints (cm/choicemap :correct (mx/scalar 1.0))
            {:keys [trace weight]} (p/generate arith-task [target] constraints)
            expr (:retval trace)
            result (eval-expr expr)]
        (when (and (number? result) (== result target))
          (swap! found conj {:expr expr :weight (mx/item weight)})))
      (catch :default _))
    (cleanup!))
  (println "  found" (count @found) "correct expressions out of" attempts "attempts")
  (doseq [{:keys [expr weight]} (take 5 @found)]
    (println "    " (pr-str expr) "=" (eval-expr expr) " weight:" weight))
  (assert-true "found at least one" (pos? (count @found))))

;; Test: importance sampling — collect weighted samples
(println "\n-- importance sampling (target=6, N=100) --")
(let [target 6
      n 100
      particles (vec (for [_ (range n)]
                       (let [constraints (cm/choicemap :correct (mx/scalar 1.0))
                             r (try (p/generate arith-task [target] constraints)
                                    (catch :default _ nil))
                             expr (when r (:retval (:trace r)))
                             result (when expr (eval-expr expr))
                             correct? (and (number? result) (== result target))]
                         (cleanup!)
                         {:expr expr :result result :correct? correct?
                          :weight (when r (mx/item (:weight r)))})))
      correct (filter :correct? particles)]
  (println "  particles:" n "  correct:" (count correct))
  (doseq [{:keys [expr weight]} (take 5 correct)]
    (println "    " (pr-str expr) " weight:" weight))
  (assert-true "found correct programs" (pos? (count correct))))

(println "\n=== Level 2 complete ===")

;; ============================================================
;; Level 3: Expanded grammar — let bindings, variables, if/cond
;; ============================================================

(println "\n=== P2-19 Level 3: Full ClojureScript grammar ===\n")

(def cljs-max-depth 3)
(def cmp-ops '[> < =])

(def cljs-expr
  (comb/recurse
    (fn [self]
      (dyn/auto-key
        (gen [depth ctx]
          (let [has-vars? (pos? (count ctx))
                at-max? (>= depth cljs-max-depth)
                ;; 5 form types: number, var, binop, let, if
                form-logits
                (cond
                  at-max?
                  (if has-vars? (mx/array [0 0 -100 -100 -100])
                                (mx/array [0 -100 -100 -100 -100]))
                  has-vars?
                  (mx/array [1 2 2 1 0.5])
                  :else
                  (mx/array [2 -100 3 2 0.5]))
                form-idx (long (mx/item (trace :form (dist/categorical form-logits))))]
            (case form-idx
              ;; number
              0 (let [v (trace :value (dist/categorical (mx/array [0 0 0 0 0 0 0 0 0])))]
                  (inc (long (mx/item v))))
              ;; var reference
              1 (let [idx (trace :var (dist/categorical (mx/array (vec (repeat (count ctx) 0)))))]
                  (nth ctx (long (mx/item idx))))
              ;; binary op
              2 (let [op-idx (long (mx/item (trace :op (dist/categorical (mx/array [0 0 0])))))
                      left (splice :left self (inc depth) ctx)
                      right (splice :right self (inc depth) ctx)]
                  (list (nth ops op-idx) left right))
              ;; let binding
              3 (let [var-name (symbol (str "v" (count ctx)))
                      binding (splice :binding self (inc depth) ctx)
                      body (splice :body self (inc depth) (conj ctx var-name))]
                  (list 'let [var-name binding] body))
              ;; if expression
              4 (let [cmp-idx (long (mx/item (trace :cmp (dist/categorical (mx/array [0 0 0])))))
                      pred-l (splice :pred-l self (inc depth) ctx)
                      pred-r (splice :pred-r self (inc depth) ctx)
                      then-br (splice :then self (inc depth) ctx)
                      else-br (splice :else self (inc depth) ctx)]
                  (list 'if (list (nth cmp-ops cmp-idx) pred-l pred-r)
                        then-br else-br)))))))))

;; Test: simulate cljs-expr
(println "-- simulate cljs-expr --")
(dotimes [_ 8]
  (let [t (p/simulate cljs-expr [0 []])
        expr (:retval t)
        result (eval-expr expr)]
    (println "  " (pr-str expr) " => " result)
    (cleanup!)))

;; Test: constrained search for expressions evaluating to target
(println "\n-- search for expressions = 42 (500 attempts) --")
(def cljs-task
  (dyn/auto-key
    (gen [target]
      (let [expr (splice :program cljs-expr 0 [])
            result (eval-expr expr)
            correct (and (number? result) (== result target))]
        (trace :correct (dist/bernoulli (if correct 0.999 0.001)))
        expr))))

(let [target 42
      found (atom [])
      attempts 500]
  (dotimes [_ attempts]
    (try
      (let [constraints (cm/choicemap :correct (mx/scalar 1.0))
            {:keys [trace weight]} (p/generate cljs-task [target] constraints)
            expr (:retval trace)
            result (eval-expr expr)]
        (when (and (number? result) (== result target))
          (swap! found conj {:expr expr :weight (mx/item weight)})))
      (catch :default _))
    (cleanup!))
  (println "  found" (count @found) "expressions = 42 out of" attempts)
  (doseq [{:keys [expr]} (take 8 @found)]
    (println "    " (pr-str expr) " => " (eval-expr expr)))
  (assert-true "found at least one" (pos? (count @found))))

(println "\n=== Level 3 complete ===")
