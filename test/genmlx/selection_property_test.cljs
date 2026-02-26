(ns genmlx.selection-property-test
  "Property-based selection algebra tests using test.check.
   Verifies all/none semantics, from-set membership, complement involution,
   hierarchical subselection, and interaction with choicemaps."
  (:require [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.selection :as sel]
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

(def addr-pool [:a :b :c :d :e :f :g :h :i :j])

(def gen-addr (gen/elements addr-pool))

(def gen-addr-set
  "Generator for a non-empty set of addresses."
  (gen/fmap set (gen/not-empty (gen/vector gen-addr 1 5))))

(def gen-flat-selection
  "Generator for a flat selection from random addresses."
  (gen/fmap #(apply sel/select %) (gen/not-empty (gen/vector gen-addr 1 5))))

(println "\n=== Selection Property-Based Tests ===\n")

;; ---------------------------------------------------------------------------
;; Property 1: all selects any address
;; ---------------------------------------------------------------------------

(println "-- all/none semantics --")

(check "all selects any address"
  (prop/for-all [addr gen-addr]
    (sel/selected? sel/all addr))
  :num-tests 50)

;; ---------------------------------------------------------------------------
;; Property 2: none selects no address
;; ---------------------------------------------------------------------------

(check "none selects no address"
  (prop/for-all [addr gen-addr]
    (not (sel/selected? sel/none addr)))
  :num-tests 50)

;; ---------------------------------------------------------------------------
;; Property 3: from-set(S) selects k iff k ∈ S
;; ---------------------------------------------------------------------------

(println "\n-- from-set properties --")

(check "from-set(S) selects k iff k ∈ S"
  (prop/for-all [s gen-addr-set
                 k gen-addr]
    (let [sel (sel/from-set s)]
      (= (contains? s k) (sel/selected? sel k)))))

;; ---------------------------------------------------------------------------
;; Property 4: complement(complement(s)) = s (behaviorally)
;; ---------------------------------------------------------------------------

(println "\n-- complement properties --")

(check "complement(complement(s)) = s"
  (prop/for-all [s gen-addr-set
                 k gen-addr]
    (let [sel (sel/from-set s)
          cc (sel/complement-sel (sel/complement-sel sel))]
      (= (sel/selected? sel k) (sel/selected? cc k)))))

;; ---------------------------------------------------------------------------
;; Property 5: complement(all) = none
;; ---------------------------------------------------------------------------

(check "complement(all) behaves as none"
  (prop/for-all [addr gen-addr]
    (not (sel/selected? (sel/complement-sel sel/all) addr)))
  :num-tests 50)

;; ---------------------------------------------------------------------------
;; Property 6: complement(none) = all
;; ---------------------------------------------------------------------------

(check "complement(none) behaves as all"
  (prop/for-all [addr gen-addr]
    (sel/selected? (sel/complement-sel sel/none) addr))
  :num-tests 50)

;; ---------------------------------------------------------------------------
;; Property 7: hierarchical selects its keys
;; ---------------------------------------------------------------------------

(println "\n-- hierarchical properties --")

(check "hierarchical selects its keys"
  (prop/for-all [addrs gen-addr-set]
    (let [;; Build hierarchical with each addr → all
          h (apply sel/hierarchical (mapcat (fn [a] [a sel/all]) addrs))]
      (every? #(sel/selected? h %) addrs)))
  :num-tests 50)

;; ---------------------------------------------------------------------------
;; Property 8: subselection returns nested selection
;; ---------------------------------------------------------------------------

(check "subselection returns nested selection"
  (prop/for-all [outer-addr gen-addr
                 inner-addr gen-addr]
    (let [inner-sel (sel/select inner-addr)
          h (sel/hierarchical outer-addr inner-sel)
          sub (sel/get-subselection h outer-addr)]
      (sel/selected? sub inner-addr))))

;; ---------------------------------------------------------------------------
;; Property 9: subselection of missing addr = none
;; ---------------------------------------------------------------------------

(check "subselection of missing addr = none"
  (prop/for-all [present-addr gen-addr]
    (let [h (sel/hierarchical present-addr sel/all)
          missing (keyword (str "__missing__" (name present-addr)))
          sub (sel/get-subselection h missing)]
      (not (sel/selected? sub :anything))))
  :num-tests 50)

;; ---------------------------------------------------------------------------
;; Property 10: all addresses of a choicemap selected by all
;; ---------------------------------------------------------------------------

(println "\n-- selection × choicemap --")

(check "all addresses of a choicemap selected by all"
  (prop/for-all [addrs (gen/not-empty (gen/vector gen-addr 1 4))]
    (let [cm (reduce (fn [c a] (cm/set-choice c [a] 1.0)) cm/EMPTY addrs)
          leaf-addrs (cm/addresses cm)]
      (every? (fn [[a]] (sel/selected? sel/all a)) leaf-addrs)))
  :num-tests 50)

;; ---------------------------------------------------------------------------
;; Property 11: no addresses of a choicemap selected by none
;; ---------------------------------------------------------------------------

(check "no addresses of a choicemap selected by none"
  (prop/for-all [addrs (gen/not-empty (gen/vector gen-addr 1 4))]
    (let [cm (reduce (fn [c a] (cm/set-choice c [a] 1.0)) cm/EMPTY addrs)
          leaf-addrs (cm/addresses cm)]
      (not-any? (fn [[a]] (sel/selected? sel/none a)) leaf-addrs)))
  :num-tests 50)

;; ---------------------------------------------------------------------------
;; Summary
;; ---------------------------------------------------------------------------

(println (str "\n=== Selection Property Tests Complete: "
              @pass-count " passed, " @fail-count " failed ==="))
