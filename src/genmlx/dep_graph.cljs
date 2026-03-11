(ns genmlx.dep-graph
  "Dependency graph construction and conditional independence analysis.
   Builds directed acyclic graphs from schema trace sites and provides
   d-separation testing, Markov blanket computation, and independent
   block detection for parallel Gibbs sampling.

   Level 3 — WP-4: Dependency Graph & Conditional Independence."
  (:require [clojure.set :as set]))

;; =========================================================================
;; DepGraph record
;; =========================================================================

(defrecord DepGraph [nodes edges parents children])

(defn- compute-direct-parents
  "Compute direct parent addresses for each trace site.
   Schema :deps are transitive — we need only direct parents.
   direct-deps(B) = deps(B) - union(deps(A) for A in deps(B) where A is a trace addr)
   NOTE: Drops skip edges (a→c when a→b→c also exists). This is a known
   limitation — fix requires schema to track direct vs transitive deps separately."
  [trace-sites addr-set]
  (let [site-map (into {} (map (juxt :addr identity))
                        (filter #(contains? addr-set (:addr %)) trace-sites))]
    (into {}
      (map (fn [site]
             (let [transitive (set/intersection (:deps site) addr-set)
                   indirect (reduce (fn [acc dep-addr]
                                      (if-let [dep-site (get site-map dep-addr)]
                                        (into acc (set/intersection (:deps dep-site) addr-set))
                                        acc))
                                    #{}
                                    transitive)]
               [(:addr site) (set/difference transitive indirect)]))
           (filter #(contains? addr-set (:addr %)) trace-sites)))))

(defn build-dep-graph
  "Build a directed acyclic graph from schema trace sites.
   Nodes: static trace addresses.
   Edges: A -> B if B directly depends on A's value.

   Returns DepGraph with:
   - :nodes — set of all trace addresses
   - :edges — set of [from to] pairs
   - :parents — {node -> #{parent nodes}}
   - :children — {node -> #{child nodes}}"
  [schema]
  (let [static-sites (filter :static? (:trace-sites schema))
        addr-set (set (map :addr static-sites))
        direct-parents (compute-direct-parents static-sites addr-set)
        edges (set (for [[child parents] direct-parents
                         parent parents]
                     [parent child]))
        children (reduce (fn [m [from to]] (update m from (fnil conj #{}) to))
                         (into {} (map (fn [a] [a #{}]) addr-set))
                         edges)]
    (->DepGraph addr-set edges direct-parents children)))

;; =========================================================================
;; Graph queries
;; =========================================================================

(defn find-roots
  "Find root nodes (no parents)."
  [graph]
  (set (filter #(empty? (get (:parents graph) %)) (:nodes graph))))

(defn find-leaves
  "Find leaf nodes (no children)."
  [graph]
  (set (filter #(empty? (get (:children graph) %)) (:nodes graph))))

(defn find-ancestors
  "Find all ancestors of a node (transitive parents)."
  [graph node]
  (loop [frontier (get (:parents graph) node #{})
         visited #{}]
    (if (empty? frontier)
      visited
      (let [next-node (first frontier)
            frontier' (disj frontier next-node)]
        (if (contains? visited next-node)
          (recur frontier' visited)
          (recur (into frontier' (get (:parents graph) next-node #{}))
                 (conj visited next-node)))))))

(defn find-descendants
  "Find all descendants of a node (transitive children)."
  [graph node]
  (loop [frontier (get (:children graph) node #{})
         visited #{}]
    (if (empty? frontier)
      visited
      (let [next-node (first frontier)
            frontier' (disj frontier next-node)]
        (if (contains? visited next-node)
          (recur frontier' visited)
          (recur (into frontier' (get (:children graph) next-node #{}))
                 (conj visited next-node)))))))

;; =========================================================================
;; D-separation (Bayes-Ball algorithm)
;; =========================================================================

(defn d-separated?
  "Test if x and y are d-separated given z-set in the DAG.
   Uses the Bayes-Ball algorithm.

   x ⊥ y | Z iff no active path from x to y given Z.

   The algorithm tracks reachability with direction:
   - Visit (node, :up) = arrived from a child
   - Visit (node, :down) = arrived from a parent

   Active path rules:
   - Non-evidence node visited from child (:up): can go to parents (:up) and children (:down)
   - Non-evidence node visited from parent (:down): can go to children (:down)
   - Evidence node visited from child (:up): blocked (can't pass through)
   - Evidence node visited from parent (:down): can go to parents (:up) — explaining away"
  [graph x y z-set]
  (if (= x y)
    false
    (let [;; BFS with (node, direction) pairs
          ;; direction: :up (arrived from child) or :down (arrived from parent)
          reachable (loop [queue (into cljs.core/PersistentQueue.EMPTY
                                       [[:up x] [:down x]])  ;; start both directions from x
                          visited #{}
                          reached #{}]
                      (if (empty? queue)
                        reached
                        (let [[dir node] (peek queue)
                              queue' (pop queue)]
                          (if (contains? visited [dir node])
                            (recur queue' visited reached)
                            (let [visited' (conj visited [dir node])
                                  reached' (if (not (contains? z-set node))
                                             (conj reached node)
                                             reached)
                                  in-z? (contains? z-set node)
                                  ;; Determine next visits based on direction and evidence
                                  next-visits
                                  (cond
                                    ;; Arrived from child (:up), node NOT in Z:
                                    ;; Can pass to parents (up) and children (down)
                                    (and (= dir :up) (not in-z?))
                                    (concat
                                      (map (fn [p] [:up p]) (get (:parents graph) node #{}))
                                      (map (fn [c] [:down c]) (get (:children graph) node #{})))

                                    ;; Arrived from child (:up), node IN Z:
                                    ;; Blocked — explaining away only works from parent direction
                                    (and (= dir :up) in-z?)
                                    []

                                    ;; Arrived from parent (:down), node NOT in Z:
                                    ;; Can pass to children (down)
                                    (and (= dir :down) (not in-z?))
                                    (map (fn [c] [:down c]) (get (:children graph) node #{}))

                                    ;; Arrived from parent (:down), node IN Z:
                                    ;; Collider is activated! Can go to parents (up) — explaining away
                                    (and (= dir :down) in-z?)
                                    (map (fn [p] [:up p]) (get (:parents graph) node #{})))]
                              (recur (reduce conj queue' next-visits)
                                     visited'
                                     reached'))))))]
      (not (contains? reachable y)))))

;; =========================================================================
;; Markov blanket
;; =========================================================================

(defn find-markov-blanket
  "Find the Markov blanket of a node: parents + children + co-parents.
   The minimal set that makes the node conditionally independent of all others."
  [graph node]
  (let [parents (get (:parents graph) node #{})
        children (get (:children graph) node #{})
        co-parents (into #{}
                     (mapcat (fn [child] (get (:parents graph) child #{})))
                     children)]
    (disj (set/union parents children co-parents) node)))

;; =========================================================================
;; Independent blocks
;; =========================================================================

(defn find-independent-blocks
  "Find maximal sets of unobserved nodes that are mutually independent
   given observed nodes. Uses moralized graph on unobserved nodes.

   Moralization: for each node, add undirected edges between all co-parents.
   Then remove observed nodes and find connected components.

   observed: set of addresses that are constrained.
   Returns: vector of #{addr} sets, each an independent block."
  [graph observed]
  (let [unobserved (set/difference (:nodes graph) observed)]
    (if (empty? unobserved)
      []
      (let [;; Build undirected adjacency on unobserved nodes
            ;; Include: parent-child edges + co-parent edges (moralization)
            adj (reduce
                  (fn [adj node]
                    (let [parents (set/difference (get (:parents graph) node #{}) observed)
                          children (set/difference (get (:children graph) node #{}) observed)
                          ;; Co-parents: for each child (including observed children),
                          ;; find other parents that are unobserved
                          co-parents-via-all-children
                          (reduce
                            (fn [acc child]
                              (let [child-parents (get (:parents graph) child #{})]
                                (into acc (disj (set/intersection child-parents unobserved) node))))
                            #{}
                            ;; Include observed children too — co-parents of observed
                            ;; children become dependent (explaining away)
                            (get (:children graph) node #{}))
                          neighbors (set/union parents children co-parents-via-all-children)]
                      (reduce (fn [a nbr] (-> a
                                              (update node (fnil conj #{}) nbr)
                                              (update nbr (fnil conj #{}) node)))
                              adj
                              neighbors)))
                  (into {} (map (fn [n] [n #{}]) unobserved))
                  unobserved)
            ;; Find connected components via BFS
            components (loop [remaining unobserved
                              result []]
                         (if (empty? remaining)
                           result
                           (let [start (first remaining)
                                 component (loop [frontier #{start}
                                                  visited #{}]
                                             (if (empty? frontier)
                                               visited
                                               (let [node (first frontier)
                                                     frontier' (disj frontier node)]
                                                 (if (contains? visited node)
                                                   (recur frontier' visited)
                                                   (recur (into frontier'
                                                               (set/intersection
                                                                 (get adj node #{})
                                                                 (set/difference remaining visited #{node})))
                                                          (conj visited node))))))]
                             (recur (set/difference remaining component)
                                    (conj result component)))))]
        components))))

(defn find-gibbs-blocks
  "Find blocks for block Gibbs sampling.
   Each block can be updated independently of others.

   Returns vector of maps:
   - :addresses — set of trace addresses in this block
   - :markov-blanket — union of Markov blankets for all addresses in block"
  [graph observed]
  (let [blocks (find-independent-blocks graph observed)]
    (mapv (fn [block]
            {:addresses block
             :markov-blanket (reduce set/union #{}
                               (map #(find-markov-blanket graph %) block))})
          blocks)))
