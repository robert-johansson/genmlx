;; @tier fast
(ns genmlx.choicemap-property-test
  "Property-based choicemap algebra tests using test.check.
   Verifies set/get round-trips, to-map/from-map isomorphism,
   addresses enumeration, merge identity and override semantics."
  (:require [cljs.test :as t]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.choicemap :as cm])
  (:require-macros [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Generators
;; ---------------------------------------------------------------------------

(def addr-pool
  [:a :b :c :d :e :f :g :h :i :j
   :k :l :m :n :o :p :q :r :s :t])

(def gen-addr
  "Generator for a random keyword address."
  (gen/elements addr-pool))

(def gen-path
  "Generator for a random path of 1-3 keywords."
  (gen/not-empty (gen/vector gen-addr 1 3)))

(def gen-value
  "Generator for a random leaf value (plain number)."
  (gen/double* {:min -100.0 :max 100.0 :NaN? false :infinite? false}))

(def gen-path-value
  "Generator for a [path value] pair."
  (gen/tuple gen-path gen-value))

(defn gen-choicemap-from-paths
  "Generator for a choicemap built from 1-5 random path-value pairs."
  []
  (gen/fmap
    (fn [pvs]
      (reduce (fn [cm [path val]]
                (cm/set-choice cm path val))
              cm/EMPTY pvs))
    (gen/not-empty (gen/vector gen-path-value 1 5))))

(def gen-flat-map
  "Generator for a flat {:keyword value} map (1-5 entries)."
  (gen/fmap
    (fn [kvs] (into {} kvs))
    (gen/not-empty (gen/vector (gen/tuple gen-addr gen-value) 1 5))))

;; ---------------------------------------------------------------------------
;; Properties
;; ---------------------------------------------------------------------------

(defspec set-choice-get-choice-round-trips 100
  (prop/for-all [path gen-path
                 val gen-value]
    (let [cm (cm/set-choice cm/EMPTY path val)]
      (= val (cm/get-choice cm path)))))

(defspec to-map-from-map-to-map-round-trips 100
  (prop/for-all [cm (gen-choicemap-from-paths)]
    (let [m1 (cm/to-map cm)
          cm2 (cm/from-map m1)
          m2 (cm/to-map cm2)]
      (= m1 m2))))

(defspec from-map-to-map-round-trips 100
  (prop/for-all [m gen-flat-map]
    (= m (cm/to-map (cm/from-map m)))))

(defspec addresses-returns-exactly-the-set-paths 100
  (prop/for-all [path gen-path
                 val gen-value]
    (let [cm (cm/set-choice cm/EMPTY path val)
          addrs (cm/addresses cm)]
      (= [path] addrs))))

(defspec has-value-true-on-leaves-false-on-nodes 50
  (prop/for-all [path gen-path
                 val gen-value]
    (let [cm (cm/set-choice cm/EMPTY path val)
          leaf (reduce cm/get-submap cm path)
          root-is-node (not (cm/has-value? cm))]
      (and root-is-node (cm/has-value? leaf)))))

(defspec get-submap-of-missing-addr-returns-EMPTY 50
  (prop/for-all [cm (gen-choicemap-from-paths)
                 addr gen-addr]
    (let [missing-addr (keyword (str "__missing__" (name addr)))
          sub (cm/get-submap cm missing-addr)]
      (= sub cm/EMPTY))))

(defspec EMPTY-has-no-addresses 1
  (prop/for-all [_ (gen/return nil)]
    (empty? (cm/addresses cm/EMPTY))))

(defspec merge-cm-a-EMPTY-equals-a 100
  (prop/for-all [cm (gen-choicemap-from-paths)]
    (= (cm/to-map (cm/merge-cm cm cm/EMPTY))
       (cm/to-map cm))))

(defspec merge-cm-EMPTY-b-equals-b 100
  (prop/for-all [cm (gen-choicemap-from-paths)]
    (= (cm/to-map (cm/merge-cm cm/EMPTY cm))
       (cm/to-map cm))))

(defspec merge-b-overrides-a-at-shared-paths 100
  (prop/for-all [path gen-path
                 val-a gen-value
                 val-b gen-value]
    (let [a (cm/set-choice cm/EMPTY path val-a)
          b (cm/set-choice cm/EMPTY path val-b)
          merged (cm/merge-cm a b)]
      (= val-b (cm/get-choice merged path)))))

(defspec merge-preserves-values-at-disjoint-paths 100
  (prop/for-all [val-a gen-value
                 val-b gen-value]
    (let [path-a [:__left :val]
          path-b [:__right :val]
          a (cm/set-choice cm/EMPTY path-a val-a)
          b (cm/set-choice cm/EMPTY path-b val-b)
          merged (cm/merge-cm a b)]
      (and (= val-a (cm/get-choice merged path-a))
           (= val-b (cm/get-choice merged path-b))))))

;; ---------------------------------------------------------------------------
;; Merge associativity (genmlx-ota8)
;;
;; merge-cm is a RIGHT-biased deep merge: on a leaf-vs-node conflict, b replaces
;; a's entire subtree (documented). That makes it associative ONLY over
;; PREFIX-FREE address sets (no path is a proper prefix of another) — which is
;; exactly the invariant real GFI choicemaps satisfy (a model's trace addresses
;; are prefix-free). Counterexample WITH a prefix conflict, to show the
;; constraint is load-bearing, not incidental:
;;   a={:x {:y 1}}, b={:x 2}, c={:x {:z 3}}
;;   (merge (merge a b) c) = {:x {:z 3}}      (b's leaf wiped a's :y subtree)
;;   (merge a (merge b c)) = {:x {:y 1 :z 3}} (a's :y survived)
;; So the generators below are deliberately prefix-free: flat (depth 1) and
;; fixed-depth-2 (all leaves at depth 2, all nodes at depth 1 — never a
;; leaf-vs-node clash). Over those, last-write-wins is associative.

(def gen-path2
  "Generator for a path of EXACTLY 2 keywords (prefix-free w.r.t. other depth-2
   paths: no depth-2 path is a prefix of another)."
  (gen/vector gen-addr 2 2))

(def gen-path2-value
  (gen/tuple gen-path2 gen-value))

(defn gen-choicemap2
  "Generator for a prefix-free nested choicemap from 1-5 depth-2 path-value pairs."
  []
  (gen/fmap
    (fn [pvs]
      (reduce (fn [cm [path val]] (cm/set-choice cm path val))
              cm/EMPTY pvs))
    (gen/not-empty (gen/vector gen-path2-value 1 5))))

(defspec merge-cm-associative-flat 100
  (prop/for-all [a gen-flat-map
                 b gen-flat-map
                 c gen-flat-map]
    (let [ca (cm/from-map a) cb (cm/from-map b) cc (cm/from-map c)]
      (= (cm/to-map (cm/merge-cm (cm/merge-cm ca cb) cc))
         (cm/to-map (cm/merge-cm ca (cm/merge-cm cb cc)))))))

(defspec merge-cm-associative-nested-prefix-free 100
  (prop/for-all [a (gen-choicemap2)
                 b (gen-choicemap2)
                 c (gen-choicemap2)]
    (= (cm/to-map (cm/merge-cm (cm/merge-cm a b) c))
       (cm/to-map (cm/merge-cm a (cm/merge-cm b c))))))

(t/run-tests)
