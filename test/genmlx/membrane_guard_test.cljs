;; @tier fast
(ns genmlx.membrane-guard-test
  "Tests for the two-tier install/startup capability guard (genmlx-91b3):
   - Tier-A: the native-core probe in genmlx.mlx (require-time fail-fast).
   - Tier-B: the LLM-forward probe in genmlx.llm.backend (at load-model).
   Negative cases use FAKE core/model objects + with-redefs, so no GPU eval and
   no model load are needed; the happy path confirms the LIVE core is capable
   (the require-time guard having already passed to load this ns). Assertions are
   on ex-data (capability/version), never on message strings — and the guards
   assert CAPABILITY, not version."
  (:require [cljs.test :refer [deftest is testing]]
            [clojure.string :as str]
            [genmlx.mlx :as mx]
            [genmlx.llm.backend :as backend]
            [genmlx.llm.forward :as fwd]))

(def ^:private fs (js/require "fs"))
(def ^:private path-mod (js/require "path"))

;; ===========================================================================
;; Tier-A — native-core probe (genmlx.mlx)
;; ===========================================================================

(deftest tier-a-live-core-capable
  (testing "the live @mlx-node/core is capable (require-time guard already passed)"
    (let [r (mx/native-core-report)]
      (is (:ok? r) "live core ok")
      (is (empty? (:missing r)) "no missing capabilities")
      (is (string? (:version r)) "@mlx-node/core version resolved"))))

(deftest tier-a-fully-broken-core-named
  (testing "an empty core reports every capability missing and asserts loudly"
    (let [r (mx/native-report #js {} "0.0.0")]
      (is (false? (:ok? r)))
      (is (= #{"MxArray" "add" "item" "evalArrays"} (set (:missing r)))
          "all required capabilities flagged")
      (is (thrown? js/Error (mx/assert-core-capable! r)))
      (try
        (mx/assert-core-capable! r)
        (is false "assert-core-capable! should have thrown")
        (catch :default e
          (let [d (ex-data e)]
            (is (= :native-core-incompatible (:genmlx/error d)))
            (is (= "0.0.0" (:mlx-node-core-version d)) "version carried in ex-data")
            (is (contains? (set (:missing d)) "MxArray")
                "ex-data names the missing capability")))))))

(deftest tier-a-partial-core-rejected
  (testing "a partial core (MxArray+add present, item+evalArrays missing) is rejected"
    (let [fake #js {:MxArray (fn []) :add (fn [])}
          r    (mx/native-report fake "0.0.7")]
      (is (false? (:ok? r)))
      (is (= #{"item" "evalArrays"} (set (:missing r))))
      (is (thrown? js/Error (mx/assert-core-capable! r))))))

(deftest tier-a-capable-fake-passes
  (testing "a fake core with all required exports passes (assert returns nil)"
    (let [fake #js {:MxArray (fn []) :add (fn []) :item (fn []) :evalArrays (fn [])}
          r    (mx/native-report fake "9.9.9")]
      (is (:ok? r))
      (is (empty? (:missing r)))
      (is (nil? (mx/assert-core-capable! r)) "capable core -> no throw, returns nil"))))

(deftest tier-a-asserts-capability-not-version
  (testing "the guard never pins a version (capability-only)"
    ;; a wildly different version with all caps present must still pass
    (let [fake #js {:MxArray (fn []) :add (fn []) :item (fn []) :evalArrays (fn [])}]
      (is (nil? (mx/assert-core-capable! (mx/native-report fake "123.456.789")))
          "version is irrelevant when capabilities are present"))))

;; ===========================================================================
;; Tier-B — LLM forward probe (genmlx.llm.backend)
;; ===========================================================================

(deftest tier-b-upstream-forward
  (testing "a model exposing .forward + .forwardWithCache passes"
    (is (nil? (backend/assert-upstream-forward!
                #js {:forward (fn []) :forwardWithCache (fn [])}))))
  (testing "a stale model missing .forwardWithCache (the genmlx-7siy case) is rejected"
    (let [bad #js {:forward (fn [])}]
      (is (thrown? js/Error (backend/assert-upstream-forward! bad)))
      (try
        (backend/assert-upstream-forward! bad)
        (is false "should have thrown")
        (catch :default e
          (let [d (ex-data e)]
            (is (= :upstream-forward-incompatible (:genmlx/error d)))
            (is (contains? (set (:missing d)) :forwardWithCache)
                "ex-data names the missing native method"))))))
  (testing "a model missing both forward methods is rejected"
    (is (thrown? js/Error (backend/assert-upstream-forward! #js {})))))

(deftest tier-b-owned-forward
  (testing "the real GenMLX-owned forward surface is complete"
    (is (nil? (backend/assert-owned-forward!))))
  (testing "a nil fwd/step is caught and named"
    (with-redefs [fwd/step nil]
      (is (thrown? js/Error (backend/assert-owned-forward!)))
      (try
        (backend/assert-owned-forward!)
        (is false "should have thrown")
        (catch :default e
          (let [d (ex-data e)]
            (is (= :owned-forward-incomplete (:genmlx/error d)))
            (is (contains? (set (:missing d)) :step) "ex-data names :step")))))))

(deftest tier-b-wired-into-load-model
  ;; A GPU-free source-level contract guard: load-model must invoke BOTH Tier-B
  ;; guards. This catches the most likely wiring regression (someone deletes an
  ;; assert call or guards the wrong branch). The FULL success-path behavior
  ;; ("load-model unchanged on success") is covered end-to-end by the @tier slow
  ;; LLM suite, which drives load-model with a real model through the guarded
  ;; path; the JS module singletons aren't writable, so it can't be stubbed
  ;; GPU-free here.
  (testing "load-model wires assert-owned-forward! and assert-upstream-forward!"
    (let [src  (.readFileSync fs
                              (.join path-mod (.cwd js/process) "src/genmlx/llm/backend.cljs")
                              "utf8")
          body (last (str/split src #"\(defn load-model"))]
      (is (re-find #"assert-owned-forward!" body)
          "owned path guard wired into load-model")
      (is (re-find #"assert-upstream-forward!" body)
          "upstream path guard wired into load-model"))))

(cljs.test/run-tests)
