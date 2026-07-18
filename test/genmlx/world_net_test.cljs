;; @tier fast
;; genmlx-tb5f regression: with-server teardown must not double-settle.
;;
;; Under nbb/SCI promesa, a `p/finally` teardown followed by a downstream
;; `p/catch` DOUBLE-SETTLES: the catch handler runs, yet the promise stays
;; rejected — the original error leaks past the catch to an outer handler /
;; unhandledRejection. with-server therefore tears down via `p/handle` (stop on
;; BOTH arms, re-raise exactly once). This file pins that contract: a rejecting
;; body wrapped in p/catch resolves to the catch value, and the listener is torn
;; down on both arms.
;;
;; Run: bun run --bun nbb test/genmlx/world_net_test.cljs

(ns genmlx.world-net-test
  (:require [genmlx.world.net :as net]
            [promesa.core :as p]))

(def failures (atom 0))

(defn- assert-true [desc ok]
  (if ok
    (println "PASS:" desc)
    (do (swap! failures inc) (println "FAIL:" desc))))

(defn- check-catch-resolves
  "The genmlx-tb5f leak shape: a with-server body that rejects, wrapped in
   p/catch, must resolve to the catch value — the rejection must not leak."
  []
  (-> (net/with-server (fn [_ _] {:ok true})
        (fn [_url] (p/rejected (ex-info "boom" {}))))
      (p/catch (fn [_] :handled))
      (p/then (fn [v]
                (assert-true "rejecting body + p/catch resolves to the catch value"
                             (= v :handled))))
      (p/catch (fn [e]
                 (assert-true (str "rejection leaked past p/catch: " (ex-message e))
                              false)))))

(defn- check-teardown-on-failure
  "After a failing body the listener must be gone: a request to the captured url
   rejects (connection refused) instead of answering."
  []
  (let [captured (atom nil)]
    (-> (net/with-server (fn [_ _] {:ok true})
          (fn [url]
            (reset! captured url)
            (p/rejected (ex-info "boom" {}))))
        (p/catch (fn [_] :handled))
        (p/then (fn [_]
                  (-> (net/request @captured "/ping" {})
                      (p/then (fn [_] (assert-true "listener stopped after failing body" false)))
                      (p/catch (fn [_] (assert-true "listener stopped after failing body" true)))))))))

(defn- check-success-arm
  "The success arm still returns the body's result (and tears down)."
  []
  (let [captured (atom nil)]
    (-> (net/with-server (fn [route payload] {:echo (:x payload) :route route})
          (fn [url]
            (reset! captured url)
            (net/request url "/echo" {:x 42})))
        (p/then (fn [r]
                  (assert-true "success arm returns the body result" (= 42 (:echo r)))
                  (-> (net/request @captured "/echo" {:x 1})
                      (p/then (fn [_] (assert-true "listener stopped after successful body" false)))
                      (p/catch (fn [_] (assert-true "listener stopped after successful body" true))))))
        (p/catch (fn [e]
                   (assert-true (str "success arm rejected: " (ex-message e)) false))))))

(if-not (net/available?)
  (println "SKIP: Bun network membrane unavailable (not under bun) — world_net_test")
  (-> (p/do (check-catch-resolves)
            (check-teardown-on-failure)
            (check-success-arm))
      (p/then (fn [_]
                (if (zero? @failures)
                  (println "\nworld_net_test: ALL PASS")
                  (do (println (str "\nworld_net_test: " @failures " FAILURES"))
                      (set! (.-exitCode js/process) 1)))))))
