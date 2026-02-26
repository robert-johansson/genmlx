(ns genmlx.choicemap-property-test
  "Property-based choicemap algebra tests using test.check.
   Verifies set/get round-trips, to-map/from-map isomorphism,
   addresses enumeration, merge identity and override semantics."
  (:require [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.choicemap :as cm]))

;; ---------------------------------------------------------------------------
;; Test infrastructure
;; ---------------------------------------------------------------------------

(def ^:private pass-count (volatile! 0))
(def ^:private fail-count (volatile! 0))

(defn- report-result [name result]
  (if (:pass? result)
    (do (vswap! pass-count inc)
        (println "  PASS:" name (str "(" (:num-tests result) " trials)")))
    (do (vswap! fail-count inc)
        (println "  FAIL:" name)
        (println "    seed:" (:seed result))
        (when-let [s (get-in result [:shrunk :smallest])]
          (println "    shrunk:" s)))))

(defn- check [name prop & {:keys [num-tests] :or {num-tests 100}}]
  (let [result (tc/quick-check num-tests prop)]
    (report-result name result)))

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

(println "\n=== ChoiceMap Property-Based Tests ===\n")

;; ---------------------------------------------------------------------------
;; Property 1: set-choice then get-choice round-trips
;; ---------------------------------------------------------------------------

(println "-- set/get round-trips --")

(check "set-choice → get-choice round-trips"
  (prop/for-all [path gen-path
                 val gen-value]
    (let [cm (cm/set-choice cm/EMPTY path val)]
      (= val (cm/get-choice cm path)))))

;; ---------------------------------------------------------------------------
;; Property 2: to-map → from-map → to-map round-trips
;; ---------------------------------------------------------------------------

(println "\n-- to-map/from-map isomorphism --")

(check "to-map → from-map → to-map round-trips"
  (prop/for-all [cm (gen-choicemap-from-paths)]
    (let [m1 (cm/to-map cm)
          cm2 (cm/from-map m1)
          m2 (cm/to-map cm2)]
      (= m1 m2))))

;; ---------------------------------------------------------------------------
;; Property 3: from-map → to-map round-trips
;; ---------------------------------------------------------------------------

(check "from-map → to-map round-trips"
  (prop/for-all [m gen-flat-map]
    (= m (cm/to-map (cm/from-map m)))))

;; ---------------------------------------------------------------------------
;; Property 4: addresses returns exactly the set paths
;; ---------------------------------------------------------------------------

(println "\n-- addresses enumeration --")

(check "addresses returns exactly the set paths"
  (prop/for-all [path gen-path
                 val gen-value]
    (let [cm (cm/set-choice cm/EMPTY path val)
          addrs (cm/addresses cm)]
      (= [path] addrs))))

;; ---------------------------------------------------------------------------
;; Property 5: has-value? true on leaves, false on nodes
;; ---------------------------------------------------------------------------

(check "has-value? true on leaves, false on nodes"
  (prop/for-all [path gen-path
                 val gen-value]
    (let [cm (cm/set-choice cm/EMPTY path val)
          ;; Navigate to the leaf
          leaf (reduce cm/get-submap cm path)
          ;; The root is a node (unless path is length 1 and only has that key)
          root-is-node (not (cm/has-value? cm))]
      (and root-is-node (cm/has-value? leaf))))
  :num-tests 50)

;; ---------------------------------------------------------------------------
;; Property 6: get-submap of missing addr returns EMPTY
;; ---------------------------------------------------------------------------

(check "get-submap of missing addr returns EMPTY"
  (prop/for-all [cm (gen-choicemap-from-paths)
                 addr gen-addr]
    ;; Use an address unlikely to collide: prefix with :__missing__
    (let [missing-addr (keyword (str "__missing__" (name addr)))
          sub (cm/get-submap cm missing-addr)]
      (= sub cm/EMPTY)))
  :num-tests 50)

;; ---------------------------------------------------------------------------
;; Property 7: EMPTY has no addresses
;; ---------------------------------------------------------------------------

(check "EMPTY has no addresses"
  (prop/for-all [_ (gen/return nil)]
    (empty? (cm/addresses cm/EMPTY)))
  :num-tests 1)

;; ---------------------------------------------------------------------------
;; Property 8: merge-cm(a, EMPTY) = a
;; ---------------------------------------------------------------------------

(println "\n-- merge properties --")

(check "merge-cm(a, EMPTY) = a"
  (prop/for-all [cm (gen-choicemap-from-paths)]
    (= (cm/to-map (cm/merge-cm cm cm/EMPTY))
       (cm/to-map cm))))

;; ---------------------------------------------------------------------------
;; Property 9: merge-cm(EMPTY, b) = b
;; ---------------------------------------------------------------------------

(check "merge-cm(EMPTY, b) = b"
  (prop/for-all [cm (gen-choicemap-from-paths)]
    (= (cm/to-map (cm/merge-cm cm/EMPTY cm))
       (cm/to-map cm))))

;; ---------------------------------------------------------------------------
;; Property 10: merge: b overrides a at shared paths
;; ---------------------------------------------------------------------------

(check "merge: b overrides a at shared paths"
  (prop/for-all [path gen-path
                 val-a gen-value
                 val-b gen-value]
    (let [a (cm/set-choice cm/EMPTY path val-a)
          b (cm/set-choice cm/EMPTY path val-b)
          merged (cm/merge-cm a b)]
      (= val-b (cm/get-choice merged path)))))

;; ---------------------------------------------------------------------------
;; Property 11: merge: preserves a's values at disjoint paths
;; ---------------------------------------------------------------------------

(check "merge: preserves a's values at disjoint paths"
  (prop/for-all [val-a gen-value
                 val-b gen-value]
    (let [;; Use fixed disjoint paths to ensure no overlap
          path-a [:__left :val]
          path-b [:__right :val]
          a (cm/set-choice cm/EMPTY path-a val-a)
          b (cm/set-choice cm/EMPTY path-b val-b)
          merged (cm/merge-cm a b)]
      (and (= val-a (cm/get-choice merged path-a))
           (= val-b (cm/get-choice merged path-b))))))

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== ChoiceMap Property Tests Complete: "
              @pass-count " passed, " @fail-count " failed ==="))
