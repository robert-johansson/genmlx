(ns cone-regen-bench
  "genmlx-ltx2 productionize/shelve gate: single-site MH sweep wall-clock on a
   T-site Gaussian chain — cone-restricted vs full-compiled vs handler.
   A sweep = T single-site regenerates (one per site), realizing the weight
   and materializing the trace score per move (the mcmc.cljs discipline).
   Cone graph work is O(|direct children|) per move (chain: 1), so sweep cost
   should be ~O(T) vs the full paths' O(T^2).
   Gate (SPEC v1 §7): productionize if >=5x vs compiled at T=512 AND per-move
   time ~flat in T. Handler variant capped at T<=128 (interpreted O(T^2))."
  (:require [genmlx.protocols :as p]
            [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx.random :as rng]
            [genmlx.pef :as pef]))

(defn- chain-source [t]
  (let [syms (mapv #(symbol (str "x" %)) (range t))
        addrs (mapv #(keyword (str "x" %)) (range t))
        bindings (vec (mapcat (fn [i]
                                [(syms i)
                                 (list 'trace (addrs i)
                                       (if (zero? i)
                                         (list 'dist/gaussian 0 1)
                                         (list 'dist/gaussian (syms (dec i)) 1)))])
                              (range t)))]
    (list [] (list 'let bindings (peek syms)))))

(defn- sweep!
  "One full single-site sweep: T regenerates, weight realized + score
   materialized per move. Returns elapsed ms."
  [model t]
  (let [t0 (js/performance.now)]
    (loop [i 0
           tr (p/simulate (dyn/with-key model (rng/fresh-key 1)) [])]
      (if (= i t)
        (- (js/performance.now) t0)
        (let [addr (keyword (str "x" i))
              r (p/regenerate (dyn/with-key model (rng/fresh-key (+ 100 i)))
                              tr (sel/select addr))]
          (mx/item (:weight r))
          (mx/materialize! (:score (:trace r)))
          (recur (inc i) (:trace r)))))))

(println "== cone_regen_bench: single-site sweep, Gaussian chain ==")
(println "T\tcone-ms\tcompiled-ms\thandler-ms\tcone-us/move\tspeedup-vs-compiled")

;; warmup (JIT + kernel caches)
(let [m (pef/source->model (chain-source 16))]
  (sweep! m 16)
  (sweep! (update m :schema dissoc :cone-regenerate) 16))

(doseq [t [32 128 512 1024]]
  (let [model (pef/source->model (chain-source t))
        _ (assert (fn? (:cone-regenerate (:schema model))) "cone must attach")
        cone-ms (sweep! model t)
        compiled-ms (sweep! (update model :schema dissoc :cone-regenerate) t)
        handler-ms (when (<= t 128)
                     (sweep! (dyn/strip-alternate-paths model) t))]
    (println (str t "\t"
                  (.toFixed cone-ms 0) "\t"
                  (.toFixed compiled-ms 0) "\t"
                  (if handler-ms (.toFixed handler-ms 0) "-") "\t"
                  (.toFixed (/ (* 1000 cone-ms) t) 1) "\t"
                  (.toFixed (/ compiled-ms cone-ms) 1) "x"))))
