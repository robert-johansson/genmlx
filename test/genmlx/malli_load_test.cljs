;; @tier fast
(ns genmlx.malli-load-test
  "Smoke test: the malli submodule loads under the pinned nbb, and the two
   GenMLX consumers (genmlx.schemas, genmlx.dev) work end-to-end.

   No other test requires these namespaces, so without this file a malli
   that fails to load under nbb's SCI (cljs.core/-pr-writer and
   IPrintWithWriter are exposed only from nbb 1.4.207 — bean genmlx-7oxz)
   leaves the suite green while dev-mode instrumentation is broken."
  (:require [cljs.test :refer [deftest is testing]]
            [malli.core :as m]
            [malli.error :as me]
            [malli.util :as mu]
            [genmlx.schemas :as gs]
            [genmlx.dev :as dev]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.trace :as tr]
            [genmlx.choicemap :as cm])
  (:require-macros [genmlx.gen :refer [gen]]))

(def tiny-model
  (gen []
    (trace :x (dist/gaussian 0 1))))

(deftest malli-loads-and-validates-test
  (testing "malli.core basics"
    (is (m/validate [:map [:x :int]] {:x 1}) "validate accepts")
    (is (not (m/validate [:map [:x :int]] {:x "no"})) "validate rejects")
    (is (some? (me/humanize (m/explain :int "x"))) "explain + humanize"))
  (testing "malli.util"
    (is (m/validate (mu/merge [:map [:x :int]] [:map [:y :int]])
                    {:x 1 :y 2})
        "mu/merge"))
  (testing "schema printing (the nbb-compat regression surface)"
    (is (string? (pr-str (m/schema [:map [:x :int]]))) "pr-str of a schema")))

(deftest genmlx-schemas-validate-real-data-test
  (testing "gs/ChoiceMap against a real choicemap"
    (is (m/validate gs/ChoiceMap (cm/choicemap :x 1.0)) "Node validates")
    (is (m/validate gs/ChoiceMap cm/EMPTY) "EMPTY validates"))
  (testing "gs/Trace against a real trace"
    (let [t (p/simulate (dyn/with-key tiny-model (rng/fresh-key 42)) [])]
      (is (m/validate gs/Trace t) "simulate result validates")
      (is (some? (me/humanize (m/explain gs/Trace {:not :a-trace})))
          "explain humanizes a non-trace"))))

(deftest dev-instrumentation-roundtrip-test
  (testing "start! -> instrumented simulate -> stop!"
    (is (= :started (dev/start!)) "start! returns :started")
    (let [t (p/simulate (dyn/with-key tiny-model (rng/fresh-key 43)) [])]
      (is (tr/trace? t) "simulate works under validation")
      (is (number? (mx/item (:score t))) "score extracts under validation"))
    (is (= :stopped (dev/stop!)) "stop! returns :stopped")))

(cljs.test/run-tests)
