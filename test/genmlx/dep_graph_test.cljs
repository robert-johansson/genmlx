(ns genmlx.dep-graph-test
  "Tests for dependency graph construction and conditional independence.
   Covers: build-dep-graph, d-separated?, find-markov-blanket,
   find-independent-blocks, find-gibbs-blocks.
   Correctness on 5 canonical DAG structures."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.dep-graph :as dg]
            [genmlx.schema :as schema]
            [clojure.set :as set]))

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

(deftest build-dep-graph-chain
  (testing "Chain: a -> b -> c"
    (let [g (dg/build-dep-graph chain-schema)]
      (is (= #{:a :b :c} (:nodes g)) "chain: nodes")
      (is (= #{[:a :b] [:b :c]} (:edges g)) "chain: edges")
      (is (= #{} (get (:parents g) :a)) "chain: parents of :a")
      (is (= #{:a} (get (:parents g) :b)) "chain: parents of :b")
      (is (= #{:b} (get (:parents g) :c)) "chain: parents of :c")
      (is (= #{:b} (get (:children g) :a)) "chain: children of :a")
      (is (= #{:c} (get (:children g) :b)) "chain: children of :b")
      (is (= #{} (get (:children g) :c)) "chain: children of :c"))))

(deftest build-dep-graph-fork
  (testing "Fork: a <- b -> c"
    (let [g (dg/build-dep-graph fork-schema)]
      (is (= #{:a :b :c} (:nodes g)) "fork: nodes")
      (is (= #{[:b :a] [:b :c]} (:edges g)) "fork: edges")
      (is (= #{:b} (get (:parents g) :a)) "fork: parents of :a")
      (is (= #{:a :c} (get (:children g) :b)) "fork: children of :b"))))

(deftest build-dep-graph-collider
  (testing "Collider: a -> c <- b"
    (let [g (dg/build-dep-graph collider-schema)]
      (is (= #{:a :b :c} (:nodes g)) "collider: nodes")
      (is (= #{[:a :c] [:b :c]} (:edges g)) "collider: edges")
      (is (= #{:a :b} (get (:parents g) :c)) "collider: parents of :c")
      (is (= #{:c} (get (:children g) :a)) "collider: children of :a")
      (is (= #{:c} (get (:children g) :b)) "collider: children of :b"))))

(deftest build-dep-graph-diamond
  (testing "Diamond: a -> b, a -> c, b -> d <- c"
    (let [g (dg/build-dep-graph diamond-schema)]
      (is (= #{:a :b :c :d} (:nodes g)) "diamond: nodes")
      (is (= #{[:a :b] [:a :c] [:b :d] [:c :d]} (:edges g)) "diamond: edges")
      (is (= #{:b :c} (get (:parents g) :d)) "diamond: parents of :d")
      (is (= #{:b :c} (get (:children g) :a)) "diamond: children of :a"))))

(deftest build-dep-graph-indep
  (testing "Two independent components"
    (let [g (dg/build-dep-graph indep-schema)]
      (is (= #{:a :b :c :d} (:nodes g)) "indep: nodes")
      (is (= #{[:a :b] [:c :d]} (:edges g)) "indep: edges")
      (is (= #{:a} (get (:parents g) :b)) "indep: parents of :b")
      (is (= #{:c} (get (:parents g) :d)) "indep: parents of :d")
      (is (= #{:b} (get (:children g) :a)) "indep: children of :a")
      (is (= #{:d} (get (:children g) :c)) "indep: children of :c"))))

(deftest build-dep-graph-five-site
  (testing "Five-site model"
    (let [g (dg/build-dep-graph five-site-schema)]
      (is (= #{:tau :mu :sigma :y1 :y2} (:nodes g)) "5-site: nodes")
      (is (= #{:mu :sigma} (get (:parents g) :y1)) "5-site: direct parents of :y1")
      (is (= #{:tau} (get (:parents g) :mu)) "5-site: direct parents of :mu")
      (is (= #{} (get (:parents g) :tau)) "5-site: :tau is root")
      (is (= #{} (get (:parents g) :sigma)) "5-site: :sigma is root"))))

(deftest build-dep-graph-edge-cases
  (testing "Single-node model"
    (let [s (schema/extract-schema '([x] (trace :a (dist/gaussian 0 1))))
          g (dg/build-dep-graph s)]
      (is (= #{:a} (:nodes g)) "single node: nodes")
      (is (= #{} (:edges g)) "single node: edges")))

  (testing "Multi-obs shared prior"
    (let [g (dg/build-dep-graph multi-obs-schema)]
      (is (= #{:y1 :y2 :y3} (get (:children g) :mu)) "multi-obs: children of :mu")
      (is (= #{:mu} (get (:parents g) :y1)) "multi-obs: parents of :y1"))))

;; =========================================================================
;; Tests: d-separated?
;; =========================================================================

(deftest d-separated-chain
  (testing "Chain: a -> b -> c"
    (let [g (dg/build-dep-graph chain-schema)]
      (is (dg/d-separated? g :a :c #{:b}) "chain: a _|_ c | b (blocked by conditioning)")
      (is (not (dg/d-separated? g :a :c #{})) "chain: a NOT _|_ c (marginally dependent)")
      (is (not (dg/d-separated? g :a :b #{})) "chain: a NOT _|_ b (direct edge)"))))

(deftest d-separated-fork
  (testing "Fork: a <- b -> c"
    (let [g (dg/build-dep-graph fork-schema)]
      (is (dg/d-separated? g :a :c #{:b}) "fork: a _|_ c | b (common cause blocked)")
      (is (not (dg/d-separated? g :a :c #{})) "fork: a NOT _|_ c (marginally dependent via b)"))))

(deftest d-separated-collider
  (testing "Collider: a -> c <- b"
    (let [g (dg/build-dep-graph collider-schema)]
      (is (dg/d-separated? g :a :b #{}) "collider: a _|_ b (marginally independent)")
      (is (not (dg/d-separated? g :a :b #{:c})) "collider: a NOT _|_ b | c (explaining away)"))))

(deftest d-separated-diamond
  (testing "Diamond: a -> b, a -> c, b -> d <- c"
    (let [g (dg/build-dep-graph diamond-schema)]
      (is (dg/d-separated? g :b :c #{:a}) "diamond: b _|_ c | a (common cause blocked)")
      (is (not (dg/d-separated? g :b :c #{:a :d})) "diamond: b NOT _|_ c | a,d (collider d activated)")
      (is (not (dg/d-separated? g :b :c #{:d})) "diamond: b NOT _|_ c | d (collider d activates, a open)")
      (is (dg/d-separated? g :a :d #{:b :c}) "diamond: a _|_ d | b,c (blocked by both paths)"))))

(deftest d-separated-indep
  (testing "Two independent components"
    (let [g (dg/build-dep-graph indep-schema)]
      (is (dg/d-separated? g :a :c #{}) "indep: a _|_ c (different components)")
      (is (dg/d-separated? g :a :d #{}) "indep: a _|_ d (different components)")
      (is (dg/d-separated? g :b :d #{}) "indep: b _|_ d (different components)")
      (is (not (dg/d-separated? g :a :b #{})) "indep: a NOT _|_ b (same component)"))))

(deftest d-separated-self
  (testing "Self d-separation"
    (let [g (dg/build-dep-graph chain-schema)]
      (is (not (dg/d-separated? g :a :a #{})) "self: a NOT _|_ a (trivially not separated)"))))

;; =========================================================================
;; Tests: find-markov-blanket
;; =========================================================================

(deftest markov-blanket-chain
  (testing "Chain: a -> b -> c"
    (let [g (dg/build-dep-graph chain-schema)]
      (is (= #{:b} (dg/find-markov-blanket g :a)) "chain: MB(a)")
      (is (= #{:a :c} (dg/find-markov-blanket g :b)) "chain: MB(b)")
      (is (= #{:b} (dg/find-markov-blanket g :c)) "chain: MB(c)"))))

(deftest markov-blanket-collider
  (testing "Collider: a -> c <- b"
    (let [g (dg/build-dep-graph collider-schema)]
      (is (= #{:b :c} (dg/find-markov-blanket g :a)) "collider: MB(a)")
      (is (= #{:a :c} (dg/find-markov-blanket g :b)) "collider: MB(b)")
      (is (= #{:a :b} (dg/find-markov-blanket g :c)) "collider: MB(c)"))))

(deftest markov-blanket-diamond
  (testing "Diamond"
    (let [g (dg/build-dep-graph diamond-schema)]
      (is (= #{:b :c} (dg/find-markov-blanket g :a)) "diamond: MB(a)")
      (is (= #{:a :c :d} (dg/find-markov-blanket g :b)) "diamond: MB(b)")
      (is (= #{:b :c} (dg/find-markov-blanket g :d)) "diamond: MB(d)"))))

(deftest markov-blanket-multi-obs
  (testing "Multi-obs"
    (let [g (dg/build-dep-graph multi-obs-schema)]
      (is (= #{:y1 :y2 :y3} (dg/find-markov-blanket g :mu)) "multi-obs: MB(mu)")
      (is (= #{:mu} (dg/find-markov-blanket g :y1)) "multi-obs: MB(y1)"))))

;; =========================================================================
;; Tests: find-independent-blocks
;; =========================================================================

(deftest independent-blocks-indep
  (testing "Two independent components, nothing observed"
    (let [g (dg/build-dep-graph indep-schema)
          blocks (dg/find-independent-blocks g #{})]
      (is (= 2 (count blocks)) "indep unobs: 2 blocks")
      (is (some #(= % #{:a :b}) blocks) "indep unobs: contains {a,b}")
      (is (some #(= % #{:c :d}) blocks) "indep unobs: contains {c,d}"))))

(deftest independent-blocks-chain-split
  (testing "Chain with middle observed -> splits into two blocks"
    (let [g (dg/build-dep-graph chain-schema)
          blocks (dg/find-independent-blocks g #{:b})]
      (is (= 2 (count blocks)) "chain obs-b: 2 blocks")
      (is (some #(= % #{:a}) blocks) "chain obs-b: contains {a}")
      (is (some #(= % #{:c}) blocks) "chain obs-b: contains {c}"))))

(deftest independent-blocks-chain-unobs
  (testing "Chain with nothing observed -> one block"
    (let [g (dg/build-dep-graph chain-schema)
          blocks (dg/find-independent-blocks g #{})]
      (is (= 1 (count blocks)) "chain unobs: 1 block")
      (is (= #{:a :b :c} (first blocks)) "chain unobs: block is {a,b,c}"))))

(deftest independent-blocks-diamond
  (testing "Diamond with d observed"
    (let [g (dg/build-dep-graph diamond-schema)
          blocks (dg/find-independent-blocks g #{:d})]
      ;; With d observed (collider), b and c become dependent
      (is (= 1 (count blocks)) "diamond obs-d: 1 block")
      (is (= #{:a :b :c} (first blocks)) "diamond obs-d: block is {a,b,c}"))))

(deftest independent-blocks-all-observed
  (testing "All nodes observed -> 0 blocks"
    (let [g (dg/build-dep-graph chain-schema)
          blocks (dg/find-independent-blocks g #{:a :b :c})]
      (is (= 0 (count blocks)) "all observed: 0 blocks"))))

(deftest independent-blocks-five-site
  (testing "Five-site with y1,y2 observed"
    (let [g (dg/build-dep-graph five-site-schema)
          blocks (dg/find-independent-blocks g #{:y1 :y2})]
      ;; tau -> mu, mu+sigma -> y1,y2 (observed)
      ;; Unobserved: tau, mu, sigma -- all connected via moralization
      (is (= 1 (count blocks)) "5-site obs-y1y2: 1 block")
      (is (= #{:tau :mu :sigma} (first blocks)) "5-site obs-y1y2: block is {tau,mu,sigma}"))))

;; =========================================================================
;; Tests: find-gibbs-blocks
;; =========================================================================

(deftest gibbs-blocks-indep
  (testing "Independent components"
    (let [g (dg/build-dep-graph indep-schema)
          blocks (dg/find-gibbs-blocks g #{})]
      (is (= 2 (count blocks)) "gibbs indep: 2 blocks")
      (is (every? :addresses blocks) "gibbs indep: each block has :addresses")
      (is (every? #(contains? % :markov-blanket) blocks) "gibbs indep: each block has :markov-blanket"))))

(deftest gibbs-blocks-chain
  (testing "Chain with middle observed"
    (let [g (dg/build-dep-graph chain-schema)
          blocks (dg/find-gibbs-blocks g #{:b})]
      (is (= 2 (count blocks)) "gibbs chain obs-b: 2 blocks"))))

;; =========================================================================
;; Tests: utility functions
;; =========================================================================

(deftest utility-roots
  (testing "roots (no parents)"
    (let [g (dg/build-dep-graph diamond-schema)]
      (is (= #{:a} (dg/find-roots g)) "diamond: roots"))
    (let [g (dg/build-dep-graph indep-schema)]
      (is (= #{:a :c} (dg/find-roots g)) "indep: roots"))))

(deftest utility-leaves
  (testing "leaves (no children)"
    (let [g (dg/build-dep-graph diamond-schema)]
      (is (= #{:d} (dg/find-leaves g)) "diamond: leaves"))
    (let [g (dg/build-dep-graph fork-schema)]
      (is (= #{:a :c} (dg/find-leaves g)) "fork: leaves"))))

(deftest utility-ancestors
  (testing "ancestors"
    (let [g (dg/build-dep-graph diamond-schema)]
      (is (= #{:a :b :c} (dg/find-ancestors g :d)) "diamond: ancestors of :d")
      (is (= #{:a} (dg/find-ancestors g :b)) "diamond: ancestors of :b")
      (is (= #{} (dg/find-ancestors g :a)) "diamond: ancestors of :a"))))

(deftest utility-descendants
  (testing "descendants"
    (let [g (dg/build-dep-graph diamond-schema)]
      (is (= #{:b :c :d} (dg/find-descendants g :a)) "diamond: descendants of :a")
      (is (= #{:d} (dg/find-descendants g :b)) "diamond: descendants of :b")
      (is (= #{} (dg/find-descendants g :d)) "diamond: descendants of :d"))))

(cljs.test/run-tests)
