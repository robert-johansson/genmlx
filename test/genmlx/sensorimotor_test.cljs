;; Unit tests for genmlx.sensorimotor.
;;
;; Tests the symbolic-layer (Beta-Bernoulli posterior arithmetic, projection,
;; induction, revision, decision logic) and the kernel-layer (trace structure,
;; particle weights via p/generate constraints).
;;
;; Ground-truth values verified by math-verifier; tolerances per
;; PLAN_SENSORIMOTOR_NARS_GENMLX.md §14.
;;
;; Run: bun run --bun nbb test/genmlx/sensorimotor_test.cljs

(ns genmlx.sensorimotor-test
  (:require [genmlx.sensorimotor :as sm]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.protocols :as p]
            [genmlx.mlx.random :as rng]
            [clojure.string :as str]
            [genmlx.gen :refer-macros [gen]]))

;; -----------------------------------------------------------------------------
;; Test infrastructure
;; -----------------------------------------------------------------------------

(def passed (volatile! 0))
(def failed (volatile! 0))

(defn assert-true [msg cond]
  (if cond
    (do (vswap! passed inc) (println " PASS" msg))
    (do (vswap! failed inc) (println " FAIL" msg))))

(defn assert-close [msg expected actual tol]
  (cond
    (or (and (number? actual)   (js/Number.isNaN actual))
        (and (number? expected) (js/Number.isNaN expected)))
    (do (vswap! failed inc)
        (println " FAIL" msg "  NaN encountered  expected:" expected "actual:" actual))
    :else
    (let [diff (Math/abs (- expected actual))]
      (if (<= diff tol)
        (do (vswap! passed inc) (println " PASS" msg "  =" actual))
        (do (vswap! failed inc)
            (println " FAIL" msg "  expected:" expected "  got:" actual "  diff:" diff))))))

(defn assert-equal [msg expected actual]
  (if (= expected actual)
    (do (vswap! passed inc) (println " PASS" msg "  =" actual))
    (do (vswap! failed inc)
        (println " FAIL" msg "  expected:" expected "  got:" actual))))

(defn fixture-impl
  "Test fixture: a fully-specified Implication via keyword args. Survives
   field-order changes in the Implication record."
  [pkey op-key K alpha beta t]
  (sm/map->Implication
    {:precondition pkey :operation op-key :consequent K
     :alpha alpha :beta beta :last-time t :update-count 0 :priority 0.0}))

;; -----------------------------------------------------------------------------
;; Section 1: Projection algebra
;; -----------------------------------------------------------------------------

(println "\n== Section 1: Temporal projection ==")

(let [impl (fixture-impl :A1 :none :B 3.0 2.0 5)
      out  (sm/project-implication impl 5 0.9)]
  (println "-- project at Δt=0 is identity")
  (assert-close "alpha unchanged" 3.0 (:alpha out) 1e-12)
  (assert-close "beta unchanged"  2.0 (:beta out) 1e-12)
  (assert-equal "last-time updated" 5 (:last-time out))
  (assert-true "result is still an Implication record"
               (instance? sm/Implication out)))

(let [impl  (fixture-impl :A1 :none :B 3.0 2.0 0)
      out   (sm/project-implication impl 10 0.9)
      decay (Math/pow 0.9 10)]
  (println "-- project decays pseudocounts at Δt=10, β_proj=0.9")
  (assert-close "alpha = 1 + 2·0.9^10" (+ 1.0 (* 2.0 decay)) (:alpha out) 1e-9)
  (assert-close "beta  = 1 + 1·0.9^10" (+ 1.0 (* 1.0 decay)) (:beta out)  1e-9)
  (assert-equal "last-time = 10" 10 (:last-time out)))

(let [impl (fixture-impl :A1 :none :B 1.0 1.0 0)
      out  (sm/project-implication impl 1000 0.9)]
  (println "-- prior preservation: α=β=1 stays at 1 forever")
  (assert-close "alpha stays at 1" 1.0 (:alpha out) 0)
  (assert-close "beta stays at 1"  1.0 (:beta out)  0))

;; -----------------------------------------------------------------------------
;; Section 2: Induction & revision
;; -----------------------------------------------------------------------------

(println "\n== Section 2: Induction and revision ==")

(let [m (sm/induce-from-trial sm/empty-memory :A1 :none :B1 0)
      i (sm/lookup-implication m :A1 :none)]
  (println "-- induce creates with α=2, β=1 and updates indices")
  (assert-close "alpha = 2" 2.0 (:alpha i) 0)
  (assert-close "beta = 1"  1.0 (:beta i)  0)
  (assert-equal "consequent = :B1" :B1 (:consequent i))
  (assert-equal "last-time = 0" 0 (:last-time i))
  (assert-close "Beta-mean = 2/3" (/ 2.0 3.0) (sm/beta-mean i) 1e-12)
  (assert-true "by-precondition index updated"
               (contains? (get-in m [:by-precondition :A1]) [:A1 :none]))
  (assert-true "by-consequent index updated"
               (contains? (get-in m [:by-consequent :B1]) [:A1 :none])))

(let [m1 (sm/induce-from-trial sm/empty-memory :A1 :none :B1 0)
      m2 (sm/induce-from-trial m1 :A1 :none :B1 5)]
  (println "-- induce is idempotent for existing key")
  (assert-true "memory unchanged"
               (= (sm/lookup-implication m1 :A1 :none)
                  (sm/lookup-implication m2 :A1 :none))))

(let [m0 (sm/induce-from-trial sm/empty-memory :A1 :none :B1 0)
      m1 (sm/revise-after-observation m0 :A1 :none :B1 5 0.5 0.9)
      m2 (sm/revise-after-observation m1 :A1 :none :B-other 10 0.5 0.9)
      i  (sm/lookup-implication m2 :A1 :none)]
  (println "-- full sequence: induce, project+positive, project+negative")
  (assert-close "alpha = 1.93916844" 1.93916844 (:alpha i) 1e-7)
  (assert-close "beta = 1.5" 1.5 (:beta i) 1e-12)
  (assert-close "Beta-mean = 0.56384806" 0.56384806 (sm/beta-mean i) 1e-7))

;; -----------------------------------------------------------------------------
;; Section 3: Memory queries
;; -----------------------------------------------------------------------------

(println "\n== Section 3: Memory queries ==")

(defn- build-test-memory
  "Helper: build a ConceptMemory from [pkey op-key K α β t] tuples.
   add-implication itself throws on duplicate keys, so silent fixture bugs
   produce a clear error rather than a quietly-dropped entry."
  [tuples]
  (reduce (fn [m [pkey op-key K alpha beta t]]
            (sm/add-implication m (fixture-impl pkey op-key K alpha beta t)))
          sm/empty-memory
          tuples))

(defn- indices-consistent?
  "Three-way invariant check on a ConceptMemory:
   1. every primary entry is forward-indexed under its precondition AND consequent
   2. no reverse-index entry points to a missing primary key
   3. no empty sets in the reverse indices"
  [memory]
  (let [{:keys [by-key by-precondition by-consequent]} memory
        all-primary (set (keys by-key))
        forward-ok? (every? (fn [[k impl]]
                              (and (contains? (get by-precondition (:precondition impl) #{}) k)
                                   (contains? (get by-consequent  (:consequent  impl) #{}) k)))
                            by-key)
        reverse-pre-ok? (every? (fn [[_ k-set]] (every? all-primary k-set)) by-precondition)
        reverse-con-ok? (every? (fn [[_ k-set]] (every? all-primary k-set)) by-consequent)
        no-empty-sets?  (and (every? seq (vals by-precondition))
                             (every? seq (vals by-consequent)))]
    (and forward-ok? reverse-pre-ok? reverse-con-ok? no-empty-sets?)))

(let [m (build-test-memory
          [[:A1 :none  :B1         5 1 0]
           [:A1 :left  :reinforced 3 2 0]
           [:A2 :left  :reinforced 1 4 0]])]
  (println "-- lookup-implication via by-key")
  (assert-true "found"     (some? (sm/lookup-implication m :A1 :none)))
  (assert-true "not-found" (nil?  (sm/lookup-implication m :A99 :none))))

(let [m (build-test-memory
          [[:A1 :none  :B1         5 1 0]
           [:A1 :left  :reinforced 3 2 0]
           [:A2 :left  :reinforced 1 4 0]])
      reinforced (sm/implications-with-consequent m :reinforced)
      reinforced-keys (set (map (juxt :precondition :operation) reinforced))]
  (println "-- implications-with-consequent via by-consequent index")
  (assert-true "two reinforced"     (= 2 (count reinforced)))
  (assert-true "contains [A1 left]" (contains? reinforced-keys [:A1 :left]))
  (assert-true "contains [A2 left]" (contains? reinforced-keys [:A2 :left]))
  (assert-true "excludes [A1 none]" (not (contains? reinforced-keys [:A1 :none]))))

(let [m (build-test-memory
          [[:A1 :none  :B1         5 1 0]
           [:A1 :left  :reinforced 3 2 0]
           [:A2 :left  :reinforced 1 4 0]])
      a1-impls    (sm/implications-matching-antecedent m :A1)
      a1-keys     (set (map (juxt :precondition :operation) a1-impls))]
  (println "-- implications-matching-antecedent via by-precondition index")
  (assert-true "two for :A1"        (= 2 (count a1-impls)))
  (assert-true "contains [A1 none]" (contains? a1-keys [:A1 :none]))
  (assert-true "contains [A1 left]" (contains? a1-keys [:A1 :left]))
  (assert-true "excludes [A2 left]" (not (contains? a1-keys [:A2 :left]))))

;; -----------------------------------------------------------------------------
;; Section 4: Decision logic — particle-rate and decision-logits
;; -----------------------------------------------------------------------------

(println "\n== Section 4: Decision logic ==")

(let [memories
      [(build-test-memory                                       ;; mean 0.9 / 0.1
         [[:A1 :left  :reinforced 9 1 0]
          [:A1 :right :reinforced 1 9 0]])
       (build-test-memory                                       ;; mean 0.5 / 0.5
         [[:A1 :left  :reinforced 1 1 0]
          [:A1 :right :reinforced 1 1 0]])
       sm/empty-memory]                                          ;; 0.5 default
      rates (sm/particle-rates memories :A1 :reinforced [:left :right])]
  (println "-- particle-rates: per-particle, per-op success-rate tensor")
  (assert-true "shape [3, 2]" (= [3 2] (mx/shape rates)))
  (let [vs (mx/->clj rates)]
    (assert-close "p0 left"  0.9 (get-in vs [0 0]) 1e-6)
    (assert-close "p0 right" 0.1 (get-in vs [0 1]) 1e-6)
    (assert-close "p1 left"  0.5 (get-in vs [1 0]) 1e-6)
    (assert-close "p1 right" 0.5 (get-in vs [1 1]) 1e-6)
    (assert-close "p2 left"  0.5 (get-in vs [2 0]) 1e-6)
    (assert-close "p2 right" 0.5 (get-in vs [2 1]) 1e-6)))

(let [;; rates: high-confidence p0, threshold p1, below-threshold p2
      rates  (mx/array [[0.9 0.5]
                        [0.51 0.5]
                        [0.4 0.3]])
      logits (sm/particle-decision-logits rates {:decision-threshold 0.51
                                                  :min-temperature 0.1})
      vs (mx/->clj logits)]
  (println "-- particle-decision-logits: threshold-gated softmax temperature")
  ;; p0: max=0.9 ≥ 0.51, temp = max(0.1, 1-0.9) = 0.1; logits = [9.0, 5.0]
  (assert-close "p0 left logit"  9.0 (get-in vs [0 0]) 1e-5)
  (assert-close "p0 right logit" 5.0 (get-in vs [0 1]) 1e-5)
  ;; p1: max=0.51 ≥ 0.51, temp = max(0.1, 1-0.51) = 0.49; logits = [1.0408, 1.0204]
  (assert-close "p1 left logit"  (/ 0.51 0.49) (get-in vs [1 0]) 1e-5)
  (assert-close "p1 right logit" (/ 0.5 0.49)  (get-in vs [1 1]) 1e-5)
  ;; p2: max=0.4 < 0.51, uniform (zeros)
  (assert-close "p2 left logit"  0.0 (get-in vs [2 0]) 1e-7)
  (assert-close "p2 right logit" 0.0 (get-in vs [2 1]) 1e-7))

;; -----------------------------------------------------------------------------
;; Section 5: Kernel — trace structure and particle weights
;; -----------------------------------------------------------------------------

(println "\n== Section 5: Kernel trace structure & per-particle weights ==")

;; Minimal percept-suite for testing — just returns a fixed stimulus
(def trivial-percept-suite
  (dyn/auto-key
    (gen [retina]
      [(:stim retina) (:stim retina)])))

;; Two-step pattern: action-kernel → gather chosen rates → consequent-kernel
(let [n 4
      rates           (mx/array [[0.9 0.1]
                                  [0.1 0.9]
                                  [0.51 0.5]
                                  [0.5 0.5]])
      decision-logits (sm/particle-decision-logits rates
                                                    {:decision-threshold 0.51
                                                     :min-temperature 0.1})
      [k1 k2] (rng/split-n (rng/fresh-key 42) 2)
      ;; Step 1: action-kernel
      action-vt (dyn/vgenerate sm/action-kernel
                                [0 {:percept-suite trivial-percept-suite
                                    :retina {:stim :A1}
                                    :decision-logits decision-logits}]
                                cm/EMPTY n k1)
      op-idxs-tensor (:retval action-vt)
      op-idxs (mx/->clj op-idxs-tensor)
      ;; Step 2: gather chosen rates and run consequent-kernel
      chosen-rates (sm/gather-chosen-rates rates op-idxs-tensor)
      ;; Per-particle constraint: particle 0 sees obs=1, 1 sees obs=0, etc.
      consequent-vt (dyn/vgenerate sm/consequent-kernel
                                    [0 {:percept-suite trivial-percept-suite
                                        :retina {:stim :A1}
                                        :rates chosen-rates}]
                                    (cm/from-map {:expected-consequent
                                                  (mx/array [1.0 0.0 1.0 0.0])})
                                    n k2)
      weights-arr (mx/->clj (:weight consequent-vt))]
  (println "-- two-step: action-kernel then consequent-kernel with constraint")
  (println "    chosen ops:" op-idxs)
  (println "    weights:   " weights-arr)
  (assert-true "consequent weight shape [N]" (= [4] (mx/shape (:weight consequent-vt))))
  (assert-true "action retval shape [N]"     (= [4] (mx/shape (:retval action-vt))))
  (assert-true "ops are 0 or 1"              (every? #(or (= 0 %) (= 1 %)) op-idxs))
  (let [vs (mx/->clj rates)]
    (doseq [i (range n)]
      (let [op   (nth op-idxs i)
            rate (get-in vs [i op])
            obs  (nth [1.0 0.0 1.0 0.0] i)
            expected-w (if (= obs 1.0)
                         (Math/log rate)
                         (Math/log (- 1.0 rate)))]
        (assert-close (str "p" i " weight = log P(obs|rate)")
                      expected-w (nth weights-arr i) 5e-5)))))

;; -----------------------------------------------------------------------------
;; Section 6: Per-op evidence-confinement invariants
;; -----------------------------------------------------------------------------

(println "\n== Section 6: Revision invariants ==")

(let [m  (build-test-memory
           [[:A1 :left  :reinforced 5 1 0]
            [:A1 :right :reinforced 5 1 0]
            [:A2 :left  :reinforced 5 1 0]])
      m' (sm/revise-after-observation m :A1 :left :reinforced 10 0.5 0.9)]
  (println "-- revise touches only the (pkey, op-key) entry, not siblings")
  (assert-true "[A1 :left] revised"
               (not= (sm/lookup-implication m  :A1 :left)
                     (sm/lookup-implication m' :A1 :left)))
  (assert-true "[A1 :right] untouched"
               (= (sm/lookup-implication m  :A1 :right)
                  (sm/lookup-implication m' :A1 :right)))
  (assert-true "[A2 :left] untouched"
               (= (sm/lookup-implication m  :A2 :left)
                  (sm/lookup-implication m' :A2 :left))))

(let [m  (build-test-memory [[:A1 :left :R 5 1 0]])
      m' (sm/revise-after-observation m :NEW :op :R 5 0.5 0.9)]
  (println "-- revise is a no-op when (pkey, op-key) absent (no silent insertion)")
  (assert-true "memory unchanged" (= m m')))

;; -----------------------------------------------------------------------------
;; Section 7: nil goal in particle-rates
;; -----------------------------------------------------------------------------

(println "\n== Section 7: nil-goal exploration path ==")

(let [memories [(build-test-memory
                  [[:A1 :left  :R 9 1 0]
                   [:A1 :right :R 1 9 0]])]
      rates (sm/particle-rates memories :A1 nil [:left :right])
      vs    (mx/->clj rates)]
  (println "-- nil goal yields uniform 0.5 regardless of memory")
  (assert-close "p0 left = 0.5"  0.5 (get-in vs [0 0]) 1e-12)
  (assert-close "p0 right = 0.5" 0.5 (get-in vs [0 1]) 1e-12))

;; -----------------------------------------------------------------------------
;; Section 8: Numerical-edge particle weights
;; -----------------------------------------------------------------------------

(println "\n== Section 8: Numerical-edge weight invariants ==")

;; rate=1.0, obs=1 → weight = log(1) = 0 exactly
(let [n 4
      rates       (mx/broadcast-to (mx/scalar 1.0) [n])
      args        {:percept-suite trivial-percept-suite
                   :retina        {:stim :A1}
                   :rates         rates}
      constraints (cm/from-map {:expected-consequent (mx/broadcast-to (mx/scalar 1.0) [n])})
      vt (dyn/vgenerate sm/consequent-kernel
                         [0 args] constraints n (rng/fresh-key 99))
      weights (mx/->clj (:weight vt))]
  (println "-- rate=1, obs=1 → weight = log(1) = 0 exactly")
  (doseq [w weights]
    (assert-close "weight = 0 (within float precision)" 0.0 w 1e-5)))

;; -----------------------------------------------------------------------------
;; Section 8b: Index-invariant preservation across all mutators
;; -----------------------------------------------------------------------------

(println "\n== Section 8b: Index-invariant preservation ==")

(let [m (-> sm/empty-memory
            (sm/induce-from-trial :A1 :left  :R 0)
            (sm/induce-from-trial :A1 :right :R 0)
            (sm/induce-from-trial :A2 :left  :R 0))]
  (println "-- empty-memory satisfies the invariant")
  (assert-true "empty-memory consistent"     (indices-consistent? sm/empty-memory))
  (println "-- consistent after induce chain")
  (assert-true "after induce x3"             (indices-consistent? m))
  (println "-- consistent after positive revise")
  (assert-true "after revise (+α)"
               (indices-consistent? (sm/revise-after-observation m :A1 :left :R 10 0.5 0.9)))
  (println "-- consistent after negative revise (mismatched consequent)")
  (let [m' (sm/revise-after-observation m :A1 :left :OTHER 10 0.5 0.9)]
    (assert-true "after revise (+β, no consequent change)" (indices-consistent? m'))
    (assert-equal "consequent stays :R" :R
                  (:consequent (sm/lookup-implication m' :A1 :left)))
    (assert-true "by-consequent[:R] still contains [A1 :left]"
                 (contains? (get (:by-consequent m') :R) [:A1 :left]))
    (assert-true "by-consequent[:OTHER] absent"
                 (nil? (get (:by-consequent m') :OTHER))))
  (println "-- consistent after update-implication")
  (let [impl  (sm/lookup-implication m :A1 :left)
        impl' (assoc impl :alpha 99.0 :update-count 7)
        m'    (sm/update-implication m impl')]
    (assert-true "after update-implication"      (indices-consistent? m'))
    (assert-close "alpha replaced" 99.0 (:alpha (sm/lookup-implication m' :A1 :left)) 0)
    (assert-true "by-precondition unchanged"     (= (:by-precondition m) (:by-precondition m')))
    (assert-true "by-consequent unchanged"       (= (:by-consequent m) (:by-consequent m'))))
  (println "-- project-all preserves indices")
  (let [m' (sm/project-all m 50 0.9)]
    (assert-true "after project-all"             (indices-consistent? m'))
    (assert-true "by-precondition byte-equal"    (= (:by-precondition m) (:by-precondition m')))
    (assert-true "by-consequent byte-equal"      (= (:by-consequent m) (:by-consequent m')))))

;; add-implication throws on duplicate primary key
(let [m (sm/induce-from-trial sm/empty-memory :A1 :left :K1 0)]
  (println "-- add-implication throws on duplicate primary key (no silent drop)")
  (assert-true "duplicate add throws"
               (try
                 (sm/add-implication m (fixture-impl :A1 :left :K2 9 9 0))
                 false
                 (catch :default _ true)))
  (println "-- induce-from-trial is the no-op-on-duplicate caller")
  (let [m2 (sm/induce-from-trial m :A1 :left :K2 5)]
    (assert-equal "consequent unchanged" :K1
                  (:consequent (sm/lookup-implication m2 :A1 :left)))
    (assert-true "indices still consistent" (indices-consistent? m2))))

;; update-implication asserts consequent immutability
(let [m (sm/induce-from-trial sm/empty-memory :A1 :left :K1 0)
      i (sm/lookup-implication m :A1 :left)]
  (println "-- update-implication throws when consequent would change")
  (assert-true "consequent-change throws"
               (try
                 (sm/update-implication m (assoc i :consequent :K2))
                 false
                 (catch :default _ true))))

;; lookup-implication trusts :by-key; reverse queries trust the indices
(let [m (build-test-memory [[:A1 :left :R 5 1 0]])
      desynced (-> m (assoc :by-precondition {}) (assoc :by-consequent {}))]
  (println "-- lookup vs reverse-index contract under desync")
  (assert-true "lookup still finds (uses :by-key)"
               (some? (sm/lookup-implication desynced :A1 :left)))
  (assert-true "implications-with-consequent empty (uses :by-consequent)"
               (empty? (sm/implications-with-consequent desynced :R)))
  (assert-true "implications-matching-antecedent empty (uses :by-precondition)"
               (empty? (sm/implications-matching-antecedent desynced :A1)))
  (assert-true "indices-consistent? detects the desync"
               (not (indices-consistent? desynced))))

;; antecedent + consequent edge cases
(println "-- empty memory: every query is empty")
(assert-true "lookup nil"
             (nil? (sm/lookup-implication sm/empty-memory :A :a)))
(assert-true "antecedent match returns empty"
             (empty? (sm/implications-matching-antecedent sm/empty-memory :A)))
(assert-true "consequent match returns empty"
             (empty? (sm/implications-with-consequent sm/empty-memory :K)))

(let [m (build-test-memory [[:A1 :left  :K1 2 1 0]
                             [:A1 :right :K2 2 1 0]
                             [:A1 :up    :K1 2 1 0]])]
  (println "-- antecedent-matching crosses consequents")
  (let [hits (sm/implications-matching-antecedent m :A1)]
    (assert-equal "three matches under one antecedent" 3 (count hits))
    (assert-equal "consequents span both K1 and K2" #{:K1 :K2}
                  (set (map :consequent hits))))
  (assert-true "no match for unknown antecedent"
               (empty? (sm/implications-matching-antecedent m :A99))))

;; -----------------------------------------------------------------------------
;; Section 9: Architectural enforcement — load-bearing claims as tests
;; -----------------------------------------------------------------------------

(println "\n== Section 9: Architectural enforcement ==")

(let [fs       (js/require "fs")
      operant  (.readFileSync fs "examples/sensorimotor_operant.cljs" "utf-8")
      ;; Strip comments line-prefixed with ;; before checking
      code-only (->> (str/split-lines operant)
                     (remove #(re-find #"^\s*;" %))
                     (str/join "\n"))]
  (println "-- operant file uses no mx/where in code (only in comments, if any)")
  (assert-true "no mx/where outside comments" (not (re-find #"mx/where" code-only)))
  (println "-- operant file uses vgenerate-based weighting")
  (assert-true "vgenerate present" (re-find #"vgenerate" code-only)))

(let [fs        (js/require "fs")
      classical (.readFileSync fs "examples/sensorimotor_classical.cljs" "utf-8")
      operant   (.readFileSync fs "examples/sensorimotor_operant.cljs" "utf-8")
      kernel-src (.readFileSync fs "src/genmlx/sensorimotor.cljs" "utf-8")]
  (println "-- both phases use sm/consequent-kernel identically")
  (assert-true "classical uses sm/consequent-kernel"
               (re-find #"sm/consequent-kernel" classical))
  (assert-true "operant uses sm/consequent-kernel"
               (re-find #"sm/consequent-kernel" operant))
  (println "-- only operant uses sm/action-kernel")
  (assert-true "operant uses sm/action-kernel"
               (re-find #"sm/action-kernel" operant))
  (assert-true "classical does NOT use sm/action-kernel"
               (not (re-find #"sm/action-kernel" classical)))
  (println "-- kernel source has no task-type branching")
  (assert-true "no :classical literal in kernel" (not (re-find #":classical" kernel-src)))
  (assert-true "no :operant literal in kernel"   (not (re-find #":operant"   kernel-src)))
  (assert-true "no task-type symbol in kernel"   (not (re-find #"task-type"  kernel-src))))

;; -----------------------------------------------------------------------------
;; Summary — exit non-zero on any failure
;; -----------------------------------------------------------------------------

(println "\n=================================")
(println "PASSED:" @passed "FAILED:" @failed)
(when (pos? @failed)
  (println "FAIL — some tests did not pass")
  (js/process.exit 1))
