(ns genmlx.dep-graph-test
  "Tests for dependency graph construction and conditional independence.
   Covers: build-dep-graph, d-separated?, find-markov-blanket,
   find-independent-blocks, find-gibbs-blocks.
   Gate 5: correctness on 5 canonical DAG structures."
  (:require [genmlx.dep-graph :as dg]
            [genmlx.schema :as schema]
            [clojure.set :as set]))

;; =========================================================================
;; Test helpers
;; =========================================================================

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn assert-true [desc pred]
  (if pred
    (do (swap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " desc)))))

(defn assert-equal [desc expected actual]
  (if (= expected actual)
    (do (swap! pass-count inc)
        (println (str "  PASS: " desc)))
    (do (swap! fail-count inc)
        (println (str "  FAIL: " desc " — expected " (pr-str expected) " got " (pr-str actual))))))

;; =========================================================================
;; Test schemas for canonical DAG structures
;; =========================================================================

;; 1. Chain: a -> b -> c
(def chain-schema
  (schema/extract-schema '([x]
    (let [a (trace :a (dist/gaussian 0 1))
          b (trace :b (dist/gaussian a 1))
          c (trace :c (dist/gaussian b 1))]
      c))))

;; 2. Fork: b -> a, b -> c  (b is common parent)
(def fork-schema
  (schema/extract-schema '([x]
    (let [b (trace :b (dist/gaussian 0 1))
          a (trace :a (dist/gaussian b 1))
          c (trace :c (dist/gaussian b 1))]
      c))))

;; 3. Collider: a -> c <- b
(def collider-schema
  (schema/extract-schema '([x]
    (let [a (trace :a (dist/gaussian 0 1))
          b (trace :b (dist/gaussian 0 1))
          c (trace :c (dist/gaussian a b))]
      c))))

;; 4. Diamond: a -> b, a -> c, b -> d <- c
(def diamond-schema
  (schema/extract-schema '([x]
    (let [a (trace :a (dist/gaussian 0 1))
          b (trace :b (dist/gaussian a 1))
          c (trace :c (dist/gaussian a 1))
          d (trace :d (dist/gaussian b c))]
      d))))

;; 5. Two independent components: (a -> b) and (c -> d)
(def indep-schema
  (schema/extract-schema '([x]
    (let [a (trace :a (dist/gaussian 0 1))
          b (trace :b (dist/gaussian a 1))
          c (trace :c (dist/gaussian 0 1))
          d (trace :d (dist/gaussian c 1))]
      d))))

;; 6. Multi-obs: mu -> y1, mu -> y2, mu -> y3 (shared prior)
(def multi-obs-schema
  (schema/extract-schema '([x]
    (let [mu (trace :mu (dist/gaussian 0 10))]
      (trace :y1 (dist/gaussian mu 1))
      (trace :y2 (dist/gaussian mu 1))
      (trace :y3 (dist/gaussian mu 1))
      mu))))

;; 7. Five-site model: tau -> mu, sigma (root), mu+sigma -> y1, mu+sigma -> y2
(def five-site-schema
  (schema/extract-schema '([xs]
    (let [tau (trace :tau (dist/gamma-dist 1 1))
          mu (trace :mu (dist/gaussian 0 tau))
          sigma (trace :sigma (dist/gamma-dist 2 1))]
      (trace :y1 (dist/gaussian mu sigma))
      (trace :y2 (dist/gaussian mu sigma))
      mu))))

;; =========================================================================
;; Tests: build-dep-graph
;; =========================================================================

(println "\n== build-dep-graph ==")

(let [g (dg/build-dep-graph chain-schema)]
  (assert-equal "chain: nodes" #{:a :b :c} (:nodes g))
  (assert-equal "chain: edges" #{[:a :b] [:b :c]} (:edges g))
  (assert-equal "chain: parents of :a" #{} (get (:parents g) :a))
  (assert-equal "chain: parents of :b" #{:a} (get (:parents g) :b))
  (assert-equal "chain: parents of :c" #{:b} (get (:parents g) :c))
  (assert-equal "chain: children of :a" #{:b} (get (:children g) :a))
  (assert-equal "chain: children of :b" #{:c} (get (:children g) :b))
  (assert-equal "chain: children of :c" #{} (get (:children g) :c)))

(let [g (dg/build-dep-graph fork-schema)]
  (assert-equal "fork: nodes" #{:a :b :c} (:nodes g))
  (assert-equal "fork: edges" #{[:b :a] [:b :c]} (:edges g))
  (assert-equal "fork: parents of :a" #{:b} (get (:parents g) :a))
  (assert-equal "fork: children of :b" #{:a :c} (get (:children g) :b)))

(let [g (dg/build-dep-graph collider-schema)]
  (assert-equal "collider: nodes" #{:a :b :c} (:nodes g))
  (assert-equal "collider: edges" #{[:a :c] [:b :c]} (:edges g))
  (assert-equal "collider: parents of :c" #{:a :b} (get (:parents g) :c))
  (assert-equal "collider: children of :a" #{:c} (get (:children g) :a))
  (assert-equal "collider: children of :b" #{:c} (get (:children g) :b)))

(let [g (dg/build-dep-graph diamond-schema)]
  (assert-equal "diamond: nodes" #{:a :b :c :d} (:nodes g))
  (assert-equal "diamond: edges" #{[:a :b] [:a :c] [:b :d] [:c :d]} (:edges g))
  (assert-equal "diamond: parents of :d" #{:b :c} (get (:parents g) :d))
  (assert-equal "diamond: children of :a" #{:b :c} (get (:children g) :a)))

(let [g (dg/build-dep-graph indep-schema)]
  (assert-equal "indep: nodes" #{:a :b :c :d} (:nodes g))
  (assert-equal "indep: edges" #{[:a :b] [:c :d]} (:edges g))
  (assert-equal "indep: parents of :b" #{:a} (get (:parents g) :b))
  (assert-equal "indep: parents of :d" #{:c} (get (:parents g) :d))
  (assert-equal "indep: children of :a" #{:b} (get (:children g) :a))
  (assert-equal "indep: children of :c" #{:d} (get (:children g) :c)))

(let [g (dg/build-dep-graph five-site-schema)]
  (assert-equal "5-site: nodes" #{:tau :mu :sigma :y1 :y2} (:nodes g))
  (assert-equal "5-site: direct parents of :y1" #{:mu :sigma} (get (:parents g) :y1))
  (assert-equal "5-site: direct parents of :mu" #{:tau} (get (:parents g) :mu))
  (assert-equal "5-site: :tau is root" #{} (get (:parents g) :tau))
  (assert-equal "5-site: :sigma is root" #{} (get (:parents g) :sigma)))

;; Edge cases
(println "\n== build-dep-graph edge cases ==")

;; Single-node model
(let [s (schema/extract-schema '([x] (trace :a (dist/gaussian 0 1))))
      g (dg/build-dep-graph s)]
  (assert-equal "single node: nodes" #{:a} (:nodes g))
  (assert-equal "single node: edges" #{} (:edges g)))

;; Multi-obs shared prior
(let [g (dg/build-dep-graph multi-obs-schema)]
  (assert-equal "multi-obs: children of :mu" #{:y1 :y2 :y3} (get (:children g) :mu))
  (assert-equal "multi-obs: parents of :y1" #{:mu} (get (:parents g) :y1)))

;; =========================================================================
;; Tests: d-separated?
;; =========================================================================

(println "\n== d-separated? ==")

;; Chain: a -> b -> c
(let [g (dg/build-dep-graph chain-schema)]
  (assert-true "chain: a ⊥ c | b (blocked by conditioning)"
    (dg/d-separated? g :a :c #{:b}))
  (assert-true "chain: a NOT ⊥ c (marginally dependent)"
    (not (dg/d-separated? g :a :c #{})))
  (assert-true "chain: a NOT ⊥ b (direct edge)"
    (not (dg/d-separated? g :a :b #{}))))

;; Fork: a <- b -> c
(let [g (dg/build-dep-graph fork-schema)]
  (assert-true "fork: a ⊥ c | b (common cause blocked)"
    (dg/d-separated? g :a :c #{:b}))
  (assert-true "fork: a NOT ⊥ c (marginally dependent via b)"
    (not (dg/d-separated? g :a :c #{}))))

;; Collider: a -> c <- b
(let [g (dg/build-dep-graph collider-schema)]
  (assert-true "collider: a ⊥ b (marginally independent)"
    (dg/d-separated? g :a :b #{}))
  (assert-true "collider: a NOT ⊥ b | c (explaining away)"
    (not (dg/d-separated? g :a :b #{:c}))))

;; Diamond: a -> b, a -> c, b -> d <- c
(let [g (dg/build-dep-graph diamond-schema)]
  (assert-true "diamond: b ⊥ c | a (common cause blocked)"
    (dg/d-separated? g :b :c #{:a}))
  (assert-true "diamond: b NOT ⊥ c | a,d (collider d activated)"
    (not (dg/d-separated? g :b :c #{:a :d})))
  (assert-true "diamond: b NOT ⊥ c | d (collider d activates, a open)"
    (not (dg/d-separated? g :b :c #{:d})))
  (assert-true "diamond: a ⊥ d | b,c (blocked by both paths)"
    (dg/d-separated? g :a :d #{:b :c})))

;; Two independent components
(let [g (dg/build-dep-graph indep-schema)]
  (assert-true "indep: a ⊥ c (different components)"
    (dg/d-separated? g :a :c #{}))
  (assert-true "indep: a ⊥ d (different components)"
    (dg/d-separated? g :a :d #{}))
  (assert-true "indep: b ⊥ d (different components)"
    (dg/d-separated? g :b :d #{}))
  (assert-true "indep: a NOT ⊥ b (same component)"
    (not (dg/d-separated? g :a :b #{}))))

;; Self d-separation
(let [g (dg/build-dep-graph chain-schema)]
  (assert-true "self: a NOT ⊥ a (trivially not separated)"
    (not (dg/d-separated? g :a :a #{}))))

;; =========================================================================
;; Tests: find-markov-blanket
;; =========================================================================

(println "\n== find-markov-blanket ==")

;; Chain: a -> b -> c
(let [g (dg/build-dep-graph chain-schema)]
  (assert-equal "chain: MB(a)" #{:b} (dg/find-markov-blanket g :a))
  (assert-equal "chain: MB(b)" #{:a :c} (dg/find-markov-blanket g :b))
  (assert-equal "chain: MB(c)" #{:b} (dg/find-markov-blanket g :c)))

;; Collider: a -> c <- b
(let [g (dg/build-dep-graph collider-schema)]
  (assert-equal "collider: MB(a)" #{:b :c} (dg/find-markov-blanket g :a))
  (assert-equal "collider: MB(b)" #{:a :c} (dg/find-markov-blanket g :b))
  (assert-equal "collider: MB(c)" #{:a :b} (dg/find-markov-blanket g :c)))

;; Diamond: a -> b, a -> c, b -> d <- c
(let [g (dg/build-dep-graph diamond-schema)]
  (assert-equal "diamond: MB(a)" #{:b :c} (dg/find-markov-blanket g :a))
  (assert-equal "diamond: MB(b)" #{:a :c :d} (dg/find-markov-blanket g :b))
  (assert-equal "diamond: MB(d)" #{:b :c} (dg/find-markov-blanket g :d)))

;; Multi-obs
(let [g (dg/build-dep-graph multi-obs-schema)]
  (assert-equal "multi-obs: MB(mu)" #{:y1 :y2 :y3}
    (dg/find-markov-blanket g :mu))
  (assert-equal "multi-obs: MB(y1)" #{:mu}
    (dg/find-markov-blanket g :y1)))

;; =========================================================================
;; Tests: find-independent-blocks
;; =========================================================================

(println "\n== find-independent-blocks ==")

;; Two independent components, nothing observed
(let [g (dg/build-dep-graph indep-schema)
      blocks (dg/find-independent-blocks g #{})]
  (assert-equal "indep unobs: 2 blocks" 2 (count blocks))
  (assert-true "indep unobs: contains {a,b}"
    (some #(= % #{:a :b}) blocks))
  (assert-true "indep unobs: contains {c,d}"
    (some #(= % #{:c :d}) blocks)))

;; Chain with middle observed -> splits into two blocks
(let [g (dg/build-dep-graph chain-schema)
      blocks (dg/find-independent-blocks g #{:b})]
  (assert-equal "chain obs-b: 2 blocks" 2 (count blocks))
  (assert-true "chain obs-b: contains {a}"
    (some #(= % #{:a}) blocks))
  (assert-true "chain obs-b: contains {c}"
    (some #(= % #{:c}) blocks)))

;; Chain with nothing observed -> one block
(let [g (dg/build-dep-graph chain-schema)
      blocks (dg/find-independent-blocks g #{})]
  (assert-equal "chain unobs: 1 block" 1 (count blocks))
  (assert-equal "chain unobs: block is {a,b,c}" #{:a :b :c} (first blocks)))

;; Diamond with all leaves observed
(let [g (dg/build-dep-graph diamond-schema)
      blocks (dg/find-independent-blocks g #{:d})]
  ;; With d observed (collider), b and c become dependent
  ;; So a, b, c should be in one block
  (assert-equal "diamond obs-d: 1 block" 1 (count blocks))
  (assert-equal "diamond obs-d: block is {a,b,c}" #{:a :b :c} (first blocks)))

;; All nodes observed -> 0 blocks (nothing to sample)
(let [g (dg/build-dep-graph chain-schema)
      blocks (dg/find-independent-blocks g #{:a :b :c})]
  (assert-equal "all observed: 0 blocks" 0 (count blocks)))

;; Five-site with y1,y2 observed
(let [g (dg/build-dep-graph five-site-schema)
      blocks (dg/find-independent-blocks g #{:y1 :y2})]
  ;; tau -> mu, mu+sigma -> y1,y2 (observed)
  ;; Unobserved: tau, mu, sigma
  ;; mu depends on tau, and mu+sigma are co-parents of observed y1,y2
  ;; So tau-mu-sigma are all connected via moralization
  (assert-equal "5-site obs-y1y2: 1 block" 1 (count blocks))
  (assert-equal "5-site obs-y1y2: block is {tau,mu,sigma}" #{:tau :mu :sigma} (first blocks)))

;; =========================================================================
;; Tests: find-gibbs-blocks
;; =========================================================================

(println "\n== find-gibbs-blocks ==")

;; Independent components
(let [g (dg/build-dep-graph indep-schema)
      blocks (dg/find-gibbs-blocks g #{})]
  (assert-equal "gibbs indep: 2 blocks" 2 (count blocks))
  (assert-true "gibbs indep: each block has :addresses"
    (every? :addresses blocks))
  (assert-true "gibbs indep: each block has :markov-blanket"
    (every? #(contains? % :markov-blanket) blocks)))

;; Chain with middle observed
(let [g (dg/build-dep-graph chain-schema)
      blocks (dg/find-gibbs-blocks g #{:b})]
  (assert-equal "gibbs chain obs-b: 2 blocks" 2 (count blocks)))

;; =========================================================================
;; Tests: utility functions
;; =========================================================================

(println "\n== utility functions ==")

;; roots (no parents)
(let [g (dg/build-dep-graph diamond-schema)]
  (assert-equal "diamond: roots" #{:a} (dg/find-roots g)))

(let [g (dg/build-dep-graph indep-schema)]
  (assert-equal "indep: roots" #{:a :c} (dg/find-roots g)))

;; leaves (no children)
(let [g (dg/build-dep-graph diamond-schema)]
  (assert-equal "diamond: leaves" #{:d} (dg/find-leaves g)))

(let [g (dg/build-dep-graph fork-schema)]
  (assert-equal "fork: leaves" #{:a :c} (dg/find-leaves g)))

;; ancestors
(let [g (dg/build-dep-graph diamond-schema)]
  (assert-equal "diamond: ancestors of :d" #{:a :b :c} (dg/find-ancestors g :d))
  (assert-equal "diamond: ancestors of :b" #{:a} (dg/find-ancestors g :b))
  (assert-equal "diamond: ancestors of :a" #{} (dg/find-ancestors g :a)))

;; descendants
(let [g (dg/build-dep-graph diamond-schema)]
  (assert-equal "diamond: descendants of :a" #{:b :c :d} (dg/find-descendants g :a))
  (assert-equal "diamond: descendants of :b" #{:d} (dg/find-descendants g :b))
  (assert-equal "diamond: descendants of :d" #{} (dg/find-descendants g :d)))

;; =========================================================================
;; Summary
;; =========================================================================

(println (str "\n== RESULTS: " @pass-count " passed, " @fail-count " failed =="))
