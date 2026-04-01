(ns genmlx.mlx-node-validation-test
  "Validate that our mlx-node fork can replace node-mlx for GenMLX.
   Exercises every category of ops against known numerical results."
  (:require ["../../src/genmlx/llm/bridge.js" :as b]
            ["@mlx-node/core" :as c]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(def M (.-MxArray c))

(defn ok [label v]
  (if v (do (swap! pass inc) (println (str "  PASS: " label)))
        (do (swap! fail inc) (println (str "  FAIL: " label)))))

(defn close [label expected actual tol]
  (let [d (js/Math.abs (- expected actual))]
    (if (<= d tol) (do (swap! pass inc) (println (str "  PASS: " label)))
                   (do (swap! fail inc) (println (str "  FAIL: " label " exp=" expected " got=" actual))))))

(defn f32 [arr] (vec (.toFloat32 b arr)))
(defn i32 [arr] (vec (.toInt32 b arr)))
(defn sh [arr] (.shape b arr))
(defn item [arr] (first (f32 arr)))

;; ============================================================
(println "\n== 1. Array creation ==")
(let [a (.fromFloat32 b #js [1 2 3] #js [3])]
  (ok "fromFloat32" (= (f32 a) [1 2 3]))
  (ok "shape" (= (vec (sh a)) [3])))
(ok "zeros" (= (f32 (.zeros b #js [3])) [0 0 0]))
(ok "ones" (= (f32 (.ones b #js [2 2])) [1 1 1 1]))
(ok "scalarFloat" (= (f32 (.scalarFloat M 3.14)) [3.140000104904175]))
(ok "scalarInt" (= (i32 (.scalarInt M 42)) [42]))
(ok "arange" (= (f32 (.arange M 0 5 1 nil)) [0 1 2 3 4]))
(ok "eye" (= (f32 (.eye M 3 nil nil nil)) [1 0 0 0 1 0 0 0 1]))

;; ============================================================
(println "\n== 2. Arithmetic ==")
(let [a (.fromFloat32 b #js [1 2 3] #js [3])
      b2 (.fromFloat32 b #js [4 5 6] #js [3])]
  (ok "add" (= (f32 (.add a b2)) [5 7 9]))
  (ok "sub" (= (f32 (.sub a b2)) [-3 -3 -3]))
  (ok "mul" (= (f32 (.mul a b2)) [4 10 18]))
  (ok "div" (= (f32 (.div a b2)) [0.25 0.4000000059604645 0.5]))
  (ok "square" (= (f32 (.square a)) [1 4 9]))
  (ok "negative" (= (f32 (.negative a)) [-1 -2 -3]))
  (ok "abs" (= (f32 (.abs (.negative a))) [1 2 3]))
  (ok "maximum" (= (f32 (.maximum a b2)) [4 5 6]))
  (ok "minimum" (= (f32 (.minimum a b2)) [1 2 3])))

;; ============================================================
(println "\n== 3. Math functions ==")
(let [a (.fromFloat32 b #js [1.0] #js [])]
  (close "exp(1)" 2.71828 (item (.exp a)) 0.001)
  (close "log(e)" 1.0 (item (.log (.exp a))) 0.001)
  (close "sqrt(1)" 1.0 (item (.sqrt a)) 0.001)
  (close "sigmoid(0)" 0.5 (item (.sigmoid (.scalarFloat M 0))) 0.001)
  (close "erf(0)" 0.0 (item (.erf (.scalarFloat M 0))) 0.001)
  (close "lgamma(1)" 0.0 (item (.lgamma a)) 0.001)
  (close "expm1(0)" 0.0 (item (.expm1 (.scalarFloat M 0))) 0.001))
(let [a (.fromFloat32 b #js [-1000.0] #js [1])
      b2 (.fromFloat32 b #js [-1000.5] #js [1])]
  (ok "logaddexp stable" (not (js/isNaN (item (.logaddexp a b2))))))

;; ============================================================
(println "\n== 4. Reductions ==")
(let [a (.fromFloat32 b #js [1 2 3 4 5 6] #js [2 3])]
  (close "sum" 21.0 (item (.sum a nil nil)) 0.001)
  (close "mean" 3.5 (item (.mean a nil nil)) 0.001)
  (ok "argmax" (= (i32 (.argmax a -1 nil)) [2 2]))
  (ok "all" (= (f32 (.all a nil nil)) [1]))
  (ok "any" (= (f32 (.any a nil nil)) [1]))
  (ok "topk" (= (f32 (.topk a 2 nil)) [2 3 5 6])))

;; ============================================================
(println "\n== 5. Shape ops ==")
(let [a (.fromFloat32 b #js [1 2 3 4 5 6] #js [2 3])]
  (ok "reshape" (= (vec (sh (.reshape b a #js [3 2]))) [3 2]))
  (ok "transpose" (= (f32 (.transpose a nil)) [1 4 2 5 3 6]))
  (ok "flatten" (= (f32 (.flatten a)) [1 2 3 4 5 6])))
(let [a (.fromFloat32 b #js [1 2] #js [2])
      b2 (.fromFloat32 b #js [3 4] #js [2])]
  (ok "inner" (= (f32 (.inner a b2)) [11])))

;; ============================================================
(println "\n== 6. Comparison ==")
(let [a (.fromFloat32 b #js [1 2 3] #js [3])
      b2 (.fromFloat32 b #js [2 2 2] #js [3])]
  (ok "greater" (= (f32 (.greater a b2)) [0 0 1]))
  (ok "equal" (= (f32 (.equal a b2)) [0 1 0]))
  (ok "where" (= (f32 (.where (.greater a b2) a b2)) [2 2 3])))

;; ============================================================
(println "\n== 7. Linear algebra ==")
(let [a (.fromFloat32 b #js [4 2 2 3] #js [2 2])
      b2 (.fromFloat32 b #js [1 2] #js [2 1])]
  (let [L (.cholesky a nil)]
    (ok "cholesky L@Lt=A" (= (f32 (.matmul L (.transpose L nil))) [4 2 2 3])))
  (close "solve" -0.125 (first (f32 (.linalgSolve a b2))) 0.001)
  (let [inv (.linalgInv a)]
    (close "inv [0,0]" 0.375 (first (f32 inv)) 0.001))
  (let [qr (.qr a)]
    (ok "qr returns 2" (= (count qr) 2)))
  (let [svd (.svd a)]
    (ok "svd returns 3" (= (count svd) 3)))
  (let [eigh (.eigh a nil)]
    (close "eigh eigenval" 1.438 (first (f32 (aget eigh 0))) 0.01)))

;; ============================================================
(println "\n== 8. Key-based PRNG ==")
(let [key (.randomKey b 42)
      [k1 k2] (.randomSplit b key)]
  (ok "key created" (some? key))
  (ok "split returns 2" (some? k2))
  (let [s1 (f32 (.keyNormal b k1 #js [3] nil))
        s2 (f32 (.keyNormal b k1 #js [3] nil))]
    (ok "reproducible" (= s1 s2)))
  (ok "uniform in [0,1)" (every? #(and (>= % 0) (< % 1))
                                  (f32 (.keyUniform b k2 #js [100] nil nil nil))))
  (ok "categorical" (some? (.keyCategorical b k1
                              (.fromFloat32 b #js [-1 0 1 2] #js [4]) nil))))

;; ============================================================
(println "\n== 9. Autograd ==")
(let [x (.fromFloat32 b #js [3.0] #js [])
      [loss grad] (.valueAndGrad M (fn [x] (.sum (.square x) nil nil)) #js [x])]
  (close "x² loss" 9.0 (item loss) 0.001)
  (close "x² grad" 6.0 (item grad) 0.001))
(let [a (.fromFloat32 b #js [2.0] #js [])
      b2 (.fromFloat32 b #js [3.0] #js [])
      [loss ga gb] (.valueAndGrad M (fn [a b] (.sum (.mul a b) nil nil)) #js [a b2])]
  (close "a*b da" 3.0 (item ga) 0.001)
  (close "a*b db" 2.0 (item gb) 0.001))

;; ============================================================
(println "\n== 10. Vmap ==")
(let [x (.fromFloat32 b #js [1 2 3 4 5 6] #js [3 2])
      result (.vmap M (fn [row] (.square row)) #js [x] #js [0] #js [0])]
  (ok "vmap square" (= (f32 (aget result 0)) [1 4 9 16 25 36])))

;; ============================================================
(println "\n== 11. Compile ==")
(let [x (.fromFloat32 b #js [1 2 3] #js [3])
      result (.compileFn M (fn [x] (.sum (.square x) nil nil)) #js [x] nil)]
  (close "compile sum(x²)" 14.0 (item (aget result 0)) 0.001))

;; ============================================================
(println "\n== 12. Memory management ==")
(ok "metalIsAvailable" (.metalIsAvailable c))
(ok "getActiveMemory" (number? (.getActiveMemory c)))
(ok "getPeakMemory" (number? (.getPeakMemory c)))
(.resetPeakMemory c)
(ok "resetPeakMemory" (= 0 (.getPeakMemory c)))
(.clearCache c)
(ok "clearCache" true)
(.synchronize c)
(ok "synchronize" true)

;; ============================================================
(println (str "\n== TOTAL: " @pass " passed, " @fail " failed =="))
