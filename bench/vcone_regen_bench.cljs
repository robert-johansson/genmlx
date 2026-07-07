(ns vcone-regen-bench
  "genmlx-js93 gate: vmh sweep wall-clock on a T-site Gaussian chain with N
   parallel chains — batched cone vs full-body batched handler, N and T swept.
   A sweep = T vmh-steps (one per site). The batched cone does
   O(|direct children|) graph work per move for ALL N lanes at once; the
   handler re-executes the whole body per move. Also reports the scalar-cone
   per-chain-sweep equivalent (from cone_regen_bench: N chains would cost
   N x scalar-sweep) to show the vectorization amortization."
  (:require [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.mcmc :as mcmc]
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

(defn- sweep-ms [model t n]
  (let [vt (dyn/vsimulate model [] n (rng/fresh-key 1))
        t0 (js/performance.now)]
    (mcmc/vmh model vt {:iters 1 :key (rng/fresh-key 2)})
    (- (js/performance.now) t0)))

(println "== vcone_regen_bench: one vmh sweep (T moves), Gaussian chain ==")
(println "T\tN\tvcone-ms\thandler-ms\tspeedup")

;; warmup
(let [m (pef/source->model (chain-source 16))] (sweep-ms m 16 8))

(doseq [t [64 256]
        n [16 256]]
  (let [model (pef/source->model (chain-source t))
        _ (assert (fn? (:vcone-regenerate (:schema model))) "vcone must attach")
        vcone-ms (sweep-ms model t n)
        handler-ms (sweep-ms (update model :schema dissoc :vcone-regenerate) t n)]
    (println (str t "\t" n "\t"
                  (.toFixed vcone-ms 0) "\t"
                  (.toFixed handler-ms 0) "\t"
                  (.toFixed (/ handler-ms vcone-ms) 1) "x"))))
