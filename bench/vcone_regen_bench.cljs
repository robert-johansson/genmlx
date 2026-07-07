(ns vcone-regen-bench
  "genmlx-js93/da04 paper-grade benchmark: vmh sweep wall-clock, batched cone
   vs full-body batched handler, with repeats (mean±sd), three topologies, and
   T up to 1024.

   Topologies (all flat static, gaussian noise-transform dists):
     chain  x_i ~ N(x_{i-1}, 1)                     cone = self + 1 child
     tree   x_i ~ N(x_{parent(i)}, 1), binary       cone = self + ≤2 children
     sv     h_t ~ N(0.95 h_{t-1}, 0.25),            cone = self + h_{t+1} + y_t
            y_t ~ N(0, exp(0.5 h_t))  [sweep over h's only — the latent class
            L3 analytical cannot eliminate]

   A sweep = one vmh-step per swept address. Handler reps are capped at large
   T (it is the O(T^2)-per-sweep baseline being replaced)."
  (:require [genmlx.selection :as sel]
            [genmlx.mlx :as mx]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.pef :as pef]))

;; -- model sources ----------------------------------------------------------

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

(defn- tree-source [t]
  (let [syms (mapv #(symbol (str "x" %)) (range t))
        addrs (mapv #(keyword (str "x" %)) (range t))
        bindings (vec (mapcat (fn [i]
                                [(syms i)
                                 (list 'trace (addrs i)
                                       (if (zero? i)
                                         (list 'dist/gaussian 0 1)
                                         (list 'dist/gaussian
                                               (syms (quot (dec i) 2)) 1)))])
                              (range t)))]
    (list [] (list 'let bindings (peek syms)))))

(defn- sv-source [steps]
  (let [hs (mapv #(symbol (str "h" %)) (range steps))
        ys (mapv #(symbol (str "y" %)) (range steps))
        h-addrs (mapv #(keyword (str "h" %)) (range steps))
        y-addrs (mapv #(keyword (str "y" %)) (range steps))
        bindings (vec (mapcat
                       (fn [i]
                         [(hs i)
                          (list 'trace (h-addrs i)
                                (if (zero? i)
                                  (list 'dist/gaussian 0 1)
                                  (list 'dist/gaussian
                                        (list 'mx/multiply 0.95 (hs (dec i)))
                                        0.25)))
                          (ys i)
                          (list 'trace (y-addrs i)
                                (list 'dist/gaussian 0
                                      (list 'mx/exp
                                            (list 'mx/multiply 0.5 (hs i)))))])
                       (range steps)))]
    (list [] (list 'let bindings (peek hs)))))

;; -- harness ------------------------------------------------------------------

(defn- sweep-ms [model addrs n seed]
  (let [vt (dyn/vsimulate model [] n (rng/fresh-key seed))
        t0 (js/performance.now)]
    (mcmc/vmh model vt {:iters 1 :addresses addrs :key (rng/fresh-key (inc seed))})
    (- (js/performance.now) t0)))

(defn- stats [xs]
  (let [n (count xs)
        m (/ (reduce + xs) n)
        sd (if (> n 1)
             (js/Math.sqrt (/ (reduce + (map #(let [d (- % m)] (* d d)) xs))
                              (dec n)))
             0)]
    [m sd]))

(defn- fmt [[m sd]] (str (.toFixed m 0) "±" (.toFixed sd 0)))

(println "== vcone_regen_bench v2: one vmh sweep, mean±sd ms ==")
(println "topology\tT-sites\tsweep-len\tN\tfused\tper-move\thandler\tstep-x\tfused-x")

;; warmup
(let [m (pef/source->model (chain-source 16))]
  (sweep-ms m nil 8 1))

(defn- bench-row! [nm source-fn size addrs-fn n cone-reps handler-reps]
  (let [model (pef/source->model (source-fn size))
        _ (assert (fn? (:fused-vmh (:schema model)))
                  (str nm " " size ": fused-vmh must attach"))
        addrs (addrs-fn model)
        t-sites (count (:trace-sites (:schema model)))
        fused (stats (mapv #(sweep-ms model addrs n (+ 10 %)) (range cone-reps)))
        stepped (stats (mapv #(sweep-ms (update model :schema dissoc :fused-vmh)
                                        addrs n (+ 30 %))
                             (range cone-reps)))
        handler (stats (mapv #(sweep-ms (update model :schema dissoc
                                                :fused-vmh :vcone-regenerate)
                                        addrs n (+ 50 %))
                             (range handler-reps)))]
    (println (str nm "\t" t-sites "\t" (count (or addrs (:dep-order (:schema model))))
                  "\t" n "\t" (fmt fused) "\t" (fmt stepped) "\t" (fmt handler)
                  "\t" (.toFixed (/ (first handler) (first stepped)) 1) "x"
                  "\t" (.toFixed (/ (first handler) (first fused)) 1) "x"))))

(def ^:private all-addrs (fn [m] (vec (:dep-order (:schema m)))))
(defn- h-addrs-of [steps] (fn [_] (mapv #(keyword (str "h" %)) (range steps))))

;; chain
(bench-row! "chain" chain-source 64   all-addrs 16  3 2)
(bench-row! "chain" chain-source 64   all-addrs 256 3 2)
(bench-row! "chain" chain-source 256  all-addrs 16  3 2)
(bench-row! "chain" chain-source 256  all-addrs 256 3 2)
(bench-row! "chain" chain-source 1024 all-addrs 256 3 1)

;; binary tree
(bench-row! "tree" tree-source 255  all-addrs 256 3 2)
(bench-row! "tree" tree-source 1023 all-addrs 256 3 1)

;; stochastic volatility (sweep latent h's only)
(bench-row! "sv" sv-source 128 (h-addrs-of 128) 256 3 2)
(bench-row! "sv" sv-source 512 (h-addrs-of 512) 256 3 1)
