;; @tier medium
(ns genmlx.tabular-test
  "Tests for genmlx.tabular: frontier-mask BFS over transition tables
   (genmlx-9ufc) and batched exact-match scoring (genmlx-8aqn)."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.tabular :as tab]))

;; -------------------------------------------------------------------------
;; Host reference BFS (independent oracle)
;; -------------------------------------------------------------------------

(defn- host-bfs
  "Plain ClojureScript BFS over nested-vector table [S A]. Returns dist vec."
  [table sources]
  (let [dist (atom (vec (repeat (count table) -1)))]
    (doseq [s sources] (swap! dist assoc s 0))
    (loop [frontier (set sources), d 1]
      (when (seq frontier)
        (let [nxt (->> frontier
                       (mapcat #(nth table %))
                       (filter #(neg? (nth @dist %)))
                       set)]
          (doseq [s nxt] (swap! dist assoc s d))
          (recur nxt (inc d)))))
    @dist))

(defn- park-miller
  "Deterministic pseudo-random ints in [0, modulus). 16807*x mod (2^31-1)
   stays under 2^53, so it is exact in JS doubles."
  [seed n modulus]
  (loop [x (inc (mod seed 2147483646)), acc []]
    (if (= (count acc) n)
      acc
      (let [x' (mod (* 16807 x) 2147483647)]
        (recur x' (conj acc (mod x' modulus)))))))

(defn- random-table [seed S A]
  (mapv vec (partition A (park-miller seed (* S A) S))))

(defn- parent-consistent?
  "Every reached non-source state s must satisfy T[parent[s], action[s]] = s
   and dist[parent[s]] = dist[s] - 1."
  [table {:keys [dist parent action]}]
  (every? (fn [s]
            (let [d (nth dist s)]
              (or (<= d 0)
                  (let [p (nth parent s)
                        a (nth action s)]
                    (and (= s (nth (nth table p) a))
                         (= (dec d) (nth dist p)))))))
          (range (count dist))))

;; -------------------------------------------------------------------------
;; BFS: chains, cycles, unreachable, shortest paths
;; -------------------------------------------------------------------------

(deftest bfs-chain-test
  (testing "chain 0->1->2->3 (terminal self-loop)"
    (let [table [[1] [2] [3] [3]]
          r     (tab/bfs->host (tab/bfs table 0))]
      (is (= [0 1 2 3] (:dist r)) "distances along the chain")
      (is (= [-1 0 1 2] (:parent r)) "parents")
      (is (= [-1 0 0 0] (:action r)) "actions (single action = 0)")
      (is (= [1 1 1 1] (:visited r)) "all reached")
      (is (:complete? r) "frontier exhausted")
      (is (= {:states [0 1 2 3] :actions [0 0 0]}
             (tab/extract-path r 3)) "path extraction")
      (is (= {:states [0] :actions []}
             (tab/extract-path r 0)) "source path is trivial"))))

(deftest bfs-unreachable-test
  (testing "disconnected component stays -1 / nil path"
    (let [table [[0] [2] [2]]
          r     (tab/bfs->host (tab/bfs table 0))]
      (is (= [0 -1 -1] (:dist r)) "only the source reached")
      (is (= [1 0 0] (:visited r)) "visited mask")
      (is (nil? (tab/extract-path r 2)) "no path to unreachable"))))

(deftest bfs-cycle-test
  (testing "4-ring terminates and measures ring distance"
    (let [table [[1] [2] [3] [0]]
          r     (tab/bfs->host (tab/bfs table 0))]
      (is (= [0 1 2 3] (:dist r)) "ring distances")
      (is (:complete? r) "cycle terminates"))))

(deftest bfs-shortest-path-test
  (testing "two routes: BFS takes the shorter"
    ;; 0 -a0-> 1 -a0-> 3 (len 2)  vs  0 -a1-> 2 -a0-> 4 -a1-> 3 (len 3)
    (let [table [[1 2] [3 1] [4 2] [3 3] [4 3]]
          r     (tab/bfs->host (tab/bfs table 0))
          path  (tab/extract-path r 3)]
      (is (= 2 (nth (:dist r) 3)) "shortest distance wins")
      (is (= [0 1 3] (:states path)) "shortest route extracted")
      (is (= [0 0] (:actions path)) "actions along the route"))))

(deftest bfs-gridworld-test
  (testing "3x3 gridworld, 4 clamped moves: manhattan distances from corner"
    (let [move  (fn [s a] ;; 0=up 1=down 2=left 3=right on a 3x3 grid
                  (let [r (quot s 3) c (mod s 3)
                        [r c] (case a
                                0 [(max 0 (dec r)) c]
                                1 [(min 2 (inc r)) c]
                                2 [r (max 0 (dec c))]
                                3 [r (min 2 (inc c))])]
                    (+ (* 3 r) c)))
          table (mapv (fn [s] (mapv #(move s %) (range 4))) (range 9))
          r     (tab/bfs->host (tab/bfs table 0))]
      (is (= [0 1 2 1 2 3 2 3 4] (:dist r)) "manhattan distances")
      (is (parent-consistent? table r) "parent/action arrays consistent")
      (is (= 4 (count (:actions (tab/extract-path r 8))))
          "corner-to-corner path length"))))

;; -------------------------------------------------------------------------
;; BFS: multi-source, batched, max-depth, random cross-check
;; -------------------------------------------------------------------------

(deftest bfs-multi-source-test
  (testing "flat id seq = one multi-source run (elementwise min distance)"
    (let [table [[1] [2] [3] [4] [4]]
          from0 (:dist (tab/bfs->host (tab/bfs table 0)))
          from3 (:dist (tab/bfs->host (tab/bfs table 3)))
          multi (:dist (tab/bfs->host (tab/bfs table [0 3])))]
      (is (= (mapv (fn [a b] (if (neg? a) b (if (neg? b) a (min a b))))
                   from0 from3)
             multi)
          "multi-source distance = min over sources"))))

(deftest bfs-batched-test
  (testing "seq of id-seqs = B independent runs, one fused pass"
    (let [table [[1] [2] [3] [3]]
          r     (tab/bfs->host (tab/bfs table [[0] [2]]))
          d     (:dist r)]
      (is (= [0 1 2 3] (nth d 0)) "row 0: from state 0")
      (is (= [-1 -1 0 1] (nth d 1)) "row 1: from state 2")
      (is (= {:states [2 3] :actions [0]}
             (tab/extract-path r 1 3)) "row-indexed path extraction")
      (is (= [[-1 0 1 2] [-1 -1 -1 2]] (:parent r))
          "parents localized per batch row"))))

(deftest bfs-max-depth-test
  (testing "max-depth truncates and reports incompleteness"
    (let [table [[1] [2] [3] [3]]
          r     (tab/bfs->host (tab/bfs table 0 {:max-depth 1}))]
      (is (= [0 1 -1 -1] (:dist r)) "one level only")
      (is (not (:complete? r)) "reported as cut off"))))

(deftest bfs-random-cross-check-test
  (testing "random tables match the host-reference BFS exactly"
    (doseq [[seed S A] [[7 30 3] [11 50 2] [13 40 5]]]
      (let [table (random-table seed S A)
            r     (tab/bfs->host (tab/bfs table 0))]
        (is (= (host-bfs table [0]) (:dist r))
            (str "dist matches oracle (seed " seed ")"))
        (is (parent-consistent? table r)
            (str "parents consistent (seed " seed ")"))))
    (testing "batched rows equal their individual runs"
      (let [table (random-table 17 40 3)
            rb    (tab/bfs->host (tab/bfs table [[0] [7] [23]]))]
        (doseq [[row src] (map-indexed vector [0 7 23])]
          (is (= (host-bfs table [src]) (nth (:dist rb) row))
              (str "batched row " row " matches oracle")))))))

;; -------------------------------------------------------------------------
;; Exact-match scoring
;; -------------------------------------------------------------------------

(deftest match-scoring-test
  (testing "match-mask / match-count / match-frac on a small batch"
    (let [target [[1 2] [3 4]]
          batch  [[[1 2] [3 4]]   ;; exact
                  [[1 2] [3 5]]   ;; 3/4 cells
                  [[9 9] [9 9]]]] ;; 0/4 cells
      (is (= [1 0 0] (mx/->clj (tab/match-mask batch target))) "exact mask")
      (is (= [4 3 0] (mx/->clj (tab/match-count batch target))) "counts")
      (is (h/all-close? [1.0 0.75 0.0]
                        (mx/->clj (tab/match-frac batch target)))
          "fractions")
      (is (= [0] (tab/matching-indices batch target)) "indices"))))

(deftest match-ignore-mask-test
  (testing "don't-care cells are excluded from comparison"
    (let [target [[1 2] [3 4]]
          batch  [[[1 2] [3 5]]   ;; mismatch only at the ignored cell
                  [[1 9] [3 4]]]  ;; mismatch at a compared cell
          dc     [[1 1] [1 0]]]   ;; ignore bottom-right
      (is (= [1 0] (mx/->clj (tab/match-mask batch target dc)))
          "near-miss at ignored cell becomes a match")
      (is (= [3 2] (mx/->clj (tab/match-count batch target dc)))
          "counts over compared cells only")
      (is (h/all-close? [1.0 (/ 2.0 3.0)]
                        (mx/->clj (tab/match-frac batch target dc)))
          "fractions over compared cells")
      (is (= [0] (tab/matching-indices batch target dc)) "indices"))))

(deftest match-batched-target-test
  (testing "per-row targets ([N & dims] target broadcast)"
    (let [batch  [[[1 2] [3 4]] [[5 6] [7 8]]]
          target [[[1 2] [3 4]] [[5 6] [7 0]]]]
      (is (= [1 0] (mx/->clj (tab/match-mask batch target)))
          "row-wise comparison"))))

(deftest match-edge-cases-test
  (testing "N=1 batch"
    (is (= [1] (mx/->clj (tab/match-mask [[[7]]] [[7]]))) "single row"))
  (testing "MLX-array inputs pass through"
    (let [t (mx/array [[1 2] [3 4]] mx/int32)
          b (mx/stack [t (mx/add t (mx/scalar 1 mx/int32))])]
      (is (= [1 0] (mx/->clj (tab/match-mask b t))) "arrays in, no host data"))))

(deftest match-big-batch-smoke-test
  (testing "GPU-built [2N,8,8] batch: planted matches found, timing sane"
    (let [target (mx/reshape (mx/astype (mx/arange 64) mx/int32) [8 8])
          n      16384
          hits   (mx/tile (mx/reshape target [1 8 8]) [n 1 1])
          misses (mx/add (mx/tile (mx/reshape target [1 8 8]) [n 1 1])
                         (mx/scalar 1 mx/int32))
          batch  (mx/concatenate [hits misses] 0)
          _      (mx/materialize! batch target)     ;; time scoring, not setup
          _      (h/realize (mx/sum (tab/match-mask batch target))) ;; warmup
          t0     (js/Date.now)
          total  (h/realize (mx/sum (tab/match-mask batch target)))
          ms     (- (js/Date.now) t0)]
      (is (= n total) "exactly the planted matches")
      (println (str "  [bench] match-mask over " (* 2 n) " 8x8 grids: " ms
                    " ms (" (.toFixed (/ (* 1000.0 ms) (* 2 n)) 2)
                    " us/hypothesis)")))))

;; -------------------------------------------------------------------------
;; BFS scale smoke (host work is per-level, not per-state)
;; -------------------------------------------------------------------------

(deftest bfs-scale-smoke-test
  (testing "S=2000, A=4 random table: full reachability vs oracle"
    (let [S      2000
          table  (random-table 23 S 4)
          t0     (js/Date.now)
          r      (tab/bfs->host (tab/bfs table 0))
          ms     (- (js/Date.now) t0)
          oracle (host-bfs table [0])]
      (is (= oracle (:dist r)) "dist matches oracle at scale")
      (is (parent-consistent? table r) "parents consistent at scale")
      (println (str "  [bench] bfs S=" S " A=4: " (:levels r) " levels in "
                    ms " ms")))))

(cljs.test/run-tests)
