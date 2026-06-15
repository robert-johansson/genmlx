;; @tier fast
(ns genmlx.world-memory-test
  "Tests for genmlx.memory — the PERSISTENCE face of the Bun world membrane
   (bun:sqlite). Oracles are INDEPENDENT of the serializer wherever possible:
   - round-trip scores are reproduced by RE-GENERATION (p/generate on the
     loaded choices), not by comparing the serializer against itself;
   - the content hash is pinned to a golden hex computed by an independent
     shell sha256 over the hand-written canonical string;
   - the session checkpoint is proven by an actual close+reopen (the
     across-process-death case)."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm]
            [genmlx.learning :as learn]
            [genmlx.sensorimotor :as sm]
            [genmlx.serialize :as ser]
            [genmlx.memory :as mem])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ^:private os (js/require "os"))
(def ^:private path-mod (js/require "path"))
(def ^:private fs (js/require "fs"))

(def ^:private temp-counter (atom 0))

(defn- temp-path []
  (.join path-mod (.tmpdir os)
         (str "genmlx-mem-test-" (.-pid js/process) "-" (swap! temp-counter inc) ".db")))

(defn- rm-store! [p]
  (doseq [suf ["" "-wal" "-shm"]]
    (try (.unlinkSync fs (str p suf)) (catch :default _ nil))))

(defn- with-temp-db
  "Run (f path) against a unique on-disk store path, deleting all sqlite files
   (db, -wal, -shm) afterwards."
  [f]
  (let [p (temp-path)]
    (try (f p) (finally (rm-store! p)))))

(defn- leaf [choices addr]
  (cm/get-value (cm/get-submap choices addr)))

;; ===========================================================================
;; Availability + WAL (the sync backend; gotcha: WAL is on-disk only)
;; ===========================================================================

(deftest availability
  (testing "bun:sqlite + node:crypto backends present under bun"
    (is (true? (mem/available?)) "memory face available under bun")))

(deftest wal-engages-on-disk-only
  (testing "an on-disk store reports journal_mode = wal"
    (with-temp-db
      (fn [p]
        (let [store (mem/open-store p)]
          (is (= "wal" (mem/journal-mode store)) "on-disk store is WAL")
          (is (= mem/schema-version-v (mem/schema-version store)) "schema version stamped")
          (mem/close-store! store)))))
  (testing "an in-memory store reports 'memory' (NOT wal) — the documented gotcha"
    (let [store (mem/open-store ":memory:")]
      (is (= "memory" (mem/journal-mode store)))
      (mem/close-store! store))))

;; ===========================================================================
;; Round-trip via re-generation (the independent score oracle)
;; ===========================================================================

(deftest choices-round-trip-regeneration-oracle
  (testing "save->restore choices; p/generate on them reproduces the score"
    (let [model (dyn/auto-key
                  (gen []
                    (let [s (trace :slope (dist/gaussian 0 2))]
                      (trace :y (dist/gaussian s 1))
                      s)))
          tr1   (p/simulate model [])
          orig  (h/realize (:score tr1))
          store (mem/open-store ":memory:")]
      (mem/save-choices! store "c1" tr1)
      (let [choices (mem/restore-choices store "c1")
            {:keys [trace]} (p/generate model [] choices)
            regen   (h/realize (:score trace))]
        (is (h/close? orig regen 1e-5)
            "re-generated score reproduces the saved trace's score"))
      (mem/close-store! store))))

(deftest full-trace-round-trip
  (testing "save-trace!/restore-trace re-generates an equivalent trace"
    (let [model (dyn/auto-key
                  (gen [m]
                    (let [s (trace :slope (dist/gaussian m 1))]
                      (trace :y (dist/gaussian s 0.5))
                      s)))
          tr1   (p/simulate model [3.0])
          orig  (h/realize (:score tr1))
          store (mem/open-store ":memory:")]
      (mem/save-trace! store "t1" tr1)
      (is (= "trace" (:kind (mem/fetch store "t1"))))
      (is (= :joint (:score-type (mem/fetch store "t1"))) "score-type recorded")
      (let [rt (mem/restore-trace store "t1" model)]
        (is (h/close? orig (h/realize (:score rt)) 1e-5) "trace score reproduced")
        (is (h/close? (h/realize (leaf (:choices tr1) :slope))
                      (h/realize (leaf (:choices rt) :slope)) 1e-6)
            "slope choice preserved"))
      (mem/close-store! store))))

(deftest collapsed-trace-refused
  (testing ":collapsed traces (empty choicemap) cannot be persisted"
    (let [model (dyn/auto-key (gen [] (trace :x (dist/gaussian 0 1))))
          tr1   (tr/with-score-type (p/simulate model []) :collapsed)
          store (mem/open-store ":memory:")]
      (is (thrown? js/Error (mem/save-trace! store "bad" tr1))
          "save-trace! throws on :collapsed")
      (mem/close-store! store))))

;; ===========================================================================
;; Content addressing — THE load-bearing canonicalization law
;; ===========================================================================

(deftest content-address-canonical
  (testing "insertion order independent (the load-bearing law)"
    (let [m1 (array-map :a 1 :b 2 :c 3)
          m2 (array-map :c 3 :a 1 :b 2)]
      (is (= (mem/content-hash m1) (mem/content-hash m2))
          "structurally-equal value, different key order -> same hash")))
  (testing "JSON payload key order is canonicalized away"
    (is (= (mem/content-hash-json "{\"b\":1,\"a\":2}")
           (mem/content-hash-json "{\"a\":2,\"b\":1}"))
        "reordered JSON object keys -> same content hash"))
  (testing "nested + set order independence"
    (is (= (mem/content-hash {:s #{3 1 2} :m {:y 1 :x 2}})
           (mem/content-hash {:m {:x 2 :y 1} :s #{2 3 1}}))))
  (testing "collision sensitive (different content -> different hash)"
    (is (not= (mem/content-hash {:a 1}) (mem/content-hash {:a 2})))
    (is (not= (mem/content-hash {:a 1}) (mem/content-hash {:b 1}))))
  (testing "container- and scalar-TYPE collisions are avoided (injective across CLJS types)"
    (is (not= (mem/content-hash #{1 2 3}) (mem/content-hash [1 2 3]))
        "a set must not collide with a vector of the same elements")
    (is (not= (mem/content-hash {:a #{1}}) (mem/content-hash {:a [1]}))
        "nested set vs vector")
    (is (not= (mem/content-hash {:tag :red}) (mem/content-hash {:tag "red"}))
        "a keyword value must not collide with the same-named string value")
    (is (not= (mem/content-hash {:x js/Infinity}) (mem/content-hash {:x (- js/Infinity)}))
        "+Inf must not collide with -Inf")
    (is (not= (mem/content-hash {:x js/NaN}) (mem/content-hash {:x nil}))
        "NaN must not collide with nil"))
  (testing "map KEYS deliberately collapse keyword<->string (cross-codec JSON/EDN equivalence)"
    (is (= (mem/content-hash {:a 1}) (mem/content-hash {"a" 1}))
        "a JSON string key and the EDN keyword it encodes address identically"))
  (testing "golden hex pin — cross-process reproducible (independent shell sha256)"
    (is (= "sha256:99d7061eca4a3858bdd9b2a59beb4e13951e1b638fecc2855ec6c92511ed223f"
           (mem/content-hash {:a [1 2 3] :b 2 :c {:x 8 :y 9}}))
        "matches `printf ... | shasum -a 256` over the canonical string"))
  (testing "put-content! is idempotent by value"
    (let [store (mem/open-store ":memory:")
          k1 (mem/put-content! store "trace" "{\"b\":1,\"a\":2}")
          k2 (mem/put-content! store "trace" "{\"a\":2,\"b\":1}")]
      (is (= k1 k2) "same logical payload -> same content key")
      (is (= 1 (mem/count-objects store)) "stored once, not twice")
      (mem/close-store! store))))

(deftest content-address-real-artifact-path
  (testing "real serializer output: choicemap key-insertion order does NOT change the content key"
    (let [v1 (mx/scalar 1.5)
          v2 (mx/scalar -0.3)
          ;; the SAME logical choicemap, addresses inserted in opposite order
          n1 (cm/->Node (array-map :slope (cm/->Value v1) :y (cm/->Value v2)))
          n2 (cm/->Node (array-map :y (cm/->Value v2) :slope (cm/->Value v1)))
          j1 (ser/save-choices {:choices n1})
          j2 (ser/save-choices {:choices n2})]
      (is (not= j1 j2) "precondition: the two serializations differ in key order")
      (is (= (mem/content-hash-json j1) (mem/content-hash-json j2))
          "structurally-equal choicemaps, different insertion order -> SAME content key")))
  (testing "save-trace-content! addresses by value (idempotent) and refuses :collapsed"
    (let [model (dyn/auto-key
                  (gen []
                    (let [s (trace :slope (dist/gaussian 0 2))]
                      (trace :y (dist/gaussian s 1))
                      s)))
          tr1   (p/simulate model [])
          store (mem/open-store ":memory:")
          k1    (mem/save-trace-content! store tr1)
          k2    (mem/save-trace-content! store tr1)]
      (is (= k1 k2) "identical trace -> identical content key")
      (is (= 1 (mem/count-objects store)) "stored once (idempotent by value)")
      (is (thrown? js/Error
            (mem/save-trace-content! store (tr/with-score-type tr1 :collapsed)))
          "save-trace-content! refuses :collapsed traces")
      (mem/close-store! store))))

;; ===========================================================================
;; Membrane-boundary invariants (done-means item 5 — enforced, not just prose)
;; ===========================================================================

(deftest membrane-invariants
  (testing "memory.cljs source obeys the persistence-membrane contract"
    (let [src (.readFileSync fs
                             (.join path-mod (.cwd js/process) "src/genmlx/memory.cljs")
                             "utf8")]
      (is (not (re-find #"\[promesa" src))
          "ZERO promesa require — persistence is the SYNC half of the Bun membrane")
      (is (not (re-find #"genmlx\.control" src)) "no dependency on genmlx.control")
      (is (not (re-find #"genmlx\.agents" src)) "no dependency on genmlx.agents")
      (is (not (re-find #"defrecord|reify|make-gen-fn|extend-protocol|extend-type" src))
          "memory exports plain fns and reifies no protocol — it is never a GF")))
  (testing "the public store value is plain data, not a reified protocol object"
    (let [store (mem/open-store ":memory:")]
      (is (map? store) "a store is a plain map {:db .. :path ..}")
      (is (not (tr/trace? store)) "a store is not a trace / GFI value")
      (mem/close-store! store))))

;; ===========================================================================
;; Particle set (VectorizedTrace) round-trip
;; ===========================================================================

(deftest particle-set-round-trip
  (testing "N=64 particle set round-trips elementwise to 1e-6"
    (let [model (dyn/auto-key (gen [] (trace :z (dist/gaussian 0 1))))
          vt0   (dyn/vsimulate model [] 64 (h/deterministic-key 7))
          ;; exercise the weight codec with a non-trivial [N] array
          vt    (assoc vt0 :weight (mx/array (mapv double (range 64))))
          z0    (h/realize-vec (leaf (:choices vt) :z))
          s0    (h/realize-vec (:score vt))
          w0    (h/realize-vec (:weight vt))
          store (mem/open-store ":memory:")]
      (mem/save-particles! store "p1" vt)
      (let [rvt (mem/restore-particles store "p1" model)]
        (is (= 64 (:n-particles rvt)) "n-particles preserved")
        (is (h/all-close? z0 (h/realize-vec (leaf (:choices rvt) :z)) 1e-6)
            "[N] choices round-trip")
        (is (h/all-close? s0 (h/realize-vec (:score rvt)) 1e-6)
            "[N] score round-trip")
        (is (h/all-close? w0 (h/realize-vec (:weight rvt)) 1e-6)
            "[N] weight round-trip"))
      (mem/close-store! store))))

;; ===========================================================================
;; Parameter store — named-key OVERWRITE semantics
;; ===========================================================================

(deftest param-store-overwrite
  (testing "re-saving under the same key overwrites (mutable threaded slot)"
    (let [store (mem/open-store ":memory:")
          ps0   (learn/make-param-store {:w 1.5 :b -2.0})
          ps1   (learn/set-param ps0 :w 9.0)]
      (mem/save-param-store! store "model/params" ps0)
      (is (= 1 (mem/count-objects store)))
      (let [r0 (mem/restore-param-store store "model/params")]
        (is (h/close? 1.5 (h/realize (learn/get-param r0 :w)) 1e-6) "w restored")
        (is (h/close? -2.0 (h/realize (learn/get-param r0 :b)) 1e-6) "b restored")
        (is (= 0 (:version r0)) "version preserved"))
      ;; overwrite
      (mem/save-param-store! store "model/params" ps1)
      (is (= 1 (mem/count-objects store)) "still one row — overwrite, not append")
      (let [r1 (mem/restore-param-store store "model/params")]
        (is (h/close? 9.0 (h/realize (learn/get-param r1 :w)) 1e-6) "w overwritten")
        (is (= 1 (:version r1)) "version bumped"))
      (mem/close-store! store))))

;; ===========================================================================
;; ConceptMemory — field round-trip (pure EDN; records -> plain maps)
;; ===========================================================================

(deftest concept-memory-round-trip
  (testing "ConceptMemory fields survive a durable round-trip"
    (let [cmem  (-> sm/empty-memory
                    (sm/add-implication (sm/fresh-implication :p1 :op1 :c1 0))
                    (sm/add-implication (sm/fresh-implication :p2 :op2 :c2 0)))
          store (mem/open-store ":memory:")]
      (mem/save-concept-memory! store "world/model" cmem)
      (let [r (mem/restore-concept-memory store "world/model")]
        (is (= 2 (count (:by-key r))) "both implications retained")
        (is (= (into {} (sm/lookup-implication cmem :p1 :op1))
               (get-in r [:by-key [:p1 :op1]]))
            "every Implication field round-trips")
        (is (h/close? 2.0 (get-in r [:by-key [:p1 :op1] :alpha]) 1e-9) "alpha")
        (is (= (:by-precondition cmem) (:by-precondition r)) "antecedent index")
        (is (= (:by-consequent cmem) (:by-consequent r)) "consequent index"))
      (mem/close-store! store))))

;; ===========================================================================
;; Recall surface — sqlite-indexed list/recent/has?/forget!/count
;; ===========================================================================

(deftest recall-indices
  (testing "list-keys / recent / has? / forget! / count-objects"
    (let [store (mem/open-store ":memory:")]
      (mem/put! store "a" "choices" "{}" nil 100)
      (mem/put! store "b" "choices" "{}" nil 200)
      (mem/put! store "c" "trace" "{}" :joint 300)
      (is (= ["a" "b" "c"] (mem/list-keys store)) "all keys ascending")
      (is (= ["a" "b"] (mem/list-keys store "choices")) "keys filtered by kind")
      (is (= ["b" "a"] (mem/recent store "choices" 2)) "recent newest-first by created_at")
      (is (mem/has? store "a"))
      (is (not (mem/has? store "zzz")))
      (is (= 3 (mem/count-objects store)) "total count")
      (is (= 2 (mem/count-objects store "choices")) "count by kind")
      (let [row (mem/fetch store "c")]
        (is (= "trace" (:kind row)))
        (is (= :joint (:score-type row)))
        (is (= 300 (:created-at row))))
      (is (true? (mem/forget! store "a")) "forget returns true when present")
      (is (false? (mem/forget! store "a")) "forget returns false when already gone")
      (is (not (mem/has? store "a")))
      (is (= 2 (mem/count-objects store)))
      (mem/close-store! store))))

;; ===========================================================================
;; EDN experience log (generic save-edn!)
;; ===========================================================================

(deftest edn-experience-log-round-trip
  (testing "an experience log (vector of plain maps) round-trips via EDN"
    (let [store (mem/open-store ":memory:")
          log   [{:t 0 :obs :red :reward 1.0}
                 {:t 1 :obs :blue :reward -0.5}]]
      (mem/save-edn! store "exp/log" "experience" log)
      (is (= log (mem/restore-edn store "exp/log")) "experience log identical")
      (mem/close-store! store))))

;; ===========================================================================
;; Session lifecycle — atomic checkpoint surviving close+reopen
;; ===========================================================================

(deftest session-checkpoint-survives-reopen
  (testing "snapshot! then close+reopen recovers the carried accumulator"
    (with-temp-db
      (fn [p]
        ;; --- session 1: open, atomic snapshot, persist a param store, CLOSE ---
        (let [store (mem/open-session p "sess-A")
              ps    (learn/make-param-store {:theta 0.75})]
          (is (= 2 (mem/snapshot! store
                                  [{:key "k1" :kind "edn" :payload (pr-str {:n 7})}
                                   {:key "k2" :kind "edn" :payload (pr-str {:n 8})}]))
              "snapshot! writes the whole atomic batch")
          (mem/save-param-store! store "acc/params" ps)
          (mem/close-store! store))
        ;; --- session 2: simulate process death — fresh open of the SAME path ---
        (let [store    (mem/open-store p)
              manifest (mem/restore-session store)
              acc      (mem/restore-param-store store "acc/params")]
          (is (= "sess-A" (:session-id manifest)) "session id survived restart")
          (is (= mem/schema-version-v (:schema-version manifest)))
          (is (= 3 (:count manifest)) "all three objects present after reopen")
          (is (h/close? 0.75 (h/realize (learn/get-param acc :theta)) 1e-6)
              "carried accumulator restored across process death")
          (is (= {:n 7} (mem/restore-edn store "k1")) "snapshot entry restored")
          (mem/close-store! store))))))

(deftest snapshot-is-atomic
  (testing "a throw mid-snapshot rolls back the whole batch"
    (let [store (mem/open-store ":memory:")]
      (is (thrown? js/Error
            (mem/snapshot! store
                           [{:key "ok" :kind "edn" :payload "{}"}
                            ;; nil payload violates NOT NULL -> the transaction aborts
                            {:key "bad" :kind "edn" :payload nil}]))
          "invalid write aborts the transaction")
      (is (= 0 (mem/count-objects store)) "nothing committed (atomic rollback)")
      (mem/close-store! store))))

;; ===========================================================================
;; Resource boundary — with-store closes on throw
;; ===========================================================================

(deftest with-store-closes-on-throw
  (testing "with-store runs its finally (close) even when f throws"
    (let [captured (atom nil)]
      (is (thrown? js/Error
            (mem/with-store ":memory:"
              (fn [store] (reset! captured store) (throw (ex-info "boom" {})))))
          "the throw propagates")
      (is (thrown? js/Error (mem/has? @captured "x"))
          "the db is closed after the scope (use-after-close throws)"))))

(deftest with-store-returns-value-sync
  (testing "with-store is synchronous and returns (f store)"
    (with-temp-db
      (fn [p]
        (let [n (mem/with-store p
                  (fn [store]
                    (mem/put! store "x" "edn" "{}")
                    (mem/count-objects store)))]
          (is (= 1 n) "returns the value of f directly (no promise)")
          (is (number? n) "result is a plain value, not a promise"))))))

(cljs.test/run-tests)
