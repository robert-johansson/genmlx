;; @tier medium
(ns genmlx.dup-fusion-test
  "Multi-consumer duplication pins (genmlx-o0ek).

   compile_fuse's all-parents-in rule strands every producer shared across
   fusion regions as its own kernel. The duplication pass (mlx fork,
   compile.cpp) clones cheap elementwise producer subtrees into consuming
   regions all-or-nothing (XLA kDuplicate), so the last consuming region
   absorbs the original outright and the standalone kernel disappears.
   CUDA-only, kill switch MLX_DISABLE_DUP_FUSION, budget MLX_DUP_FUSION_MAX.

   Pins (child processes, same fixed key, the parity linreg shape):
   1. EQUIVALENCE — fused-mala-compiled chain outputs with duplication ON
      match the kill-switch run at <=1e-5 rel (measured bit-identical on
      sm_120 2026-07-29; per-consumer rounding differences are legal, so
      the pin is tolerance, not bits).
   2. STRUCTURE — the pass FIRES: ON post-fusion tape is strictly smaller
      than OFF (measured 2026-07-29: MALA S=100 4287 vs 4820 entries), the
      ON census reports clones > 0, and the kill switch is inert (OFF
      clones = 0).

   On Metal the pass is compiled out (CUDA gate) and fused-mala-compiled
   throws by design — self-gates to a skip, like parallel_stress_test."
  (:require [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.choicemap :as cm]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.mcmc :as mcmc]
            [promesa.core :as p]
            [clojure.string :as str]))

(def pass (atom 0))
(def fail (atom 0))
(defn assert-true [desc x]
  (if x
    (do (swap! pass inc) (println "  PASS:" desc))
    (do (swap! fail inc) (println "  FAIL:" desc))))

(defn- finish! []
  (println (str "\n== dup-fusion: " @pass " pass, " @fail " fail =="))
  (js/process.exit (if (pos? @fail) 1 0)))

;; ---------------------------------------------------------------------------
;; Shared model: the parity bench's static-unrolled linreg construction
;; (literal dist params -> static schema -> tensor-native score), so the
;; pins exercise the exact tape shape the HMC-1.0x program targets.
;; ---------------------------------------------------------------------------

(def ^:private xs [0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0])
(def ^:private ys [-1.14409 0.827096 2.888684 5.701984 6.872412
                   7.502647 11.332318 12.732663 14.783041 17.115885])

(def ^:private model
  (let [src  (list '[xs]
                   (concat
                    (list 'let
                          ['slope     (list 'trace :slope
                                            (list 'dist/gaussian 0.0 10.0))
                           'intercept (list 'trace :intercept
                                            (list 'dist/gaussian 0.0 10.0))])
                    (map-indexed
                     (fn [j x]
                       (list 'trace (keyword (str "y" j))
                             (list 'dist/gaussian
                                   (list 'mx/add
                                         (list 'mx/multiply 'slope x)
                                         'intercept)
                                   1.0)))
                     xs)
                    (list 'slope)))
        body-fn
        (fn [rt _xs]
          (let [trace-op  (.-trace rt)
                slope     (trace-op :slope (dist/gaussian 0.0 10.0))
                intercept (trace-op :intercept (dist/gaussian 0.0 10.0))]
            (doseq [[j x] (map-indexed vector xs)]
              (trace-op (keyword (str "y" j))
                        (dist/gaussian (mx/add (mx/multiply slope x) intercept)
                                       1.0)))
            slope))]
    (dyn/auto-key (dyn/make-gen-fn body-fn src))))

(def ^:private obs
  (apply cm/choicemap
         (mapcat (fn [[j y]] [(keyword (str "y" j)) (mx/scalar y)])
                 (map-indexed vector ys))))

;; ---------------------------------------------------------------------------
;; Worker mode: run one fused-mala-compiled chain, print the result line.
;; ---------------------------------------------------------------------------

(def ^:private worker? (= "1" (aget (.-env js/process) "DUP_TEST_WORKER")))

(if worker?
  (let [f (mcmc/fused-mala-compiled
           {:samples 50 :burn 0 :thin 1 :step-size 0.08
            :addresses [:slope :intercept]}
           model [xs] obs)
        r ((:call f) (rng/fresh-key 4242))
        fp (mx/->clj (:final-params r))
        ss (mx/item (mx/sum (:samples r)))]
    (println (str "WORKER-RESULT "
                  (js/JSON.stringify
                   #js {:fp0 (first fp) :fp1 (second fp)
                        :ssum ss :acc (:acceptance-rate r)})))
    ((:free! f))
    (js/process.exit 0))

  ;; -------------------------------------------------------------------------
  ;; Orchestrator mode
  ;; -------------------------------------------------------------------------
  (if (mx/metal-is-available?)
    (do
      (println "Metal: duplication pass is CUDA-only — skipping (negative contract)")
      (assert-true "Metal self-gate reached" true)
      (finish!))

    (let [spawn-child
          (fn [env-overrides]
            (js/Bun.spawn
             #js {:cmd #js ["bun" "run" "--bun" "nbb"
                            "test/genmlx/dup_fusion_test.cljs"]
                  :env (js/Object.assign #js {} (.-env js/process)
                                         (clj->js (merge {:DUP_TEST_WORKER "1"
                                                          :MLX_COMPILE_DEBUG "1"}
                                                         env-overrides)))
                  :stdout "pipe" :stderr "pipe"}))
          collect
          (fn [proc]
            (p/let [code (.-exited proc)
                    out  (.text (js/Response. (.-stdout proc)))
                    err  (.text (js/Response. (.-stderr proc)))]
              {:code code :out out :err err}))
          parse-result
          (fn [{:keys [out]}]
            (when-let [line (->> (str/split-lines (or out ""))
                                 (filter #(str/starts-with? % "WORKER-RESULT "))
                                 last)]
              (js->clj (js/JSON.parse (subs line (count "WORKER-RESULT ")))
                       :keywordize-keys true)))
          parse-tape
          (fn [{:keys [err]}]
            (->> (re-seq #"post-fusion tape: (\d+) entries" (or err ""))
                 (map (comp js/parseInt second))
                 (apply max 0)))
          parse-clones
          (fn [{:keys [err]}]
            (->> (re-seq #"dup: clones=(\d+)" (or err ""))
                 (map (comp js/parseInt second))
                 (apply max 0)))
          rel-close?
          (fn [a b tol]
            (let [d (js/Math.abs (- a b))
                  m (max 1.0 (js/Math.abs a) (js/Math.abs b))]
              (<= (/ d m) tol)))]
      ;; children run serially: one chain on the GPU at a time
      (p/let [on  (collect (spawn-child {}))
              off (collect (spawn-child {:MLX_DISABLE_DUP_FUSION "1"}))]
        (let [ron  (parse-result on)
              roff (parse-result off)]
          (assert-true "dup-ON child completed" (and (zero? (:code on)) (some? ron)))
          (assert-true "dup-OFF child completed" (and (zero? (:code off)) (some? roff)))
          (when (and ron roff)
            (assert-true (str "final-params[0] equivalent at 1e-5 rel ("
                              (:fp0 ron) " vs " (:fp0 roff) ")")
                         (rel-close? (:fp0 ron) (:fp0 roff) 1e-5))
            (assert-true (str "final-params[1] equivalent at 1e-5 rel ("
                              (:fp1 ron) " vs " (:fp1 roff) ")")
                         (rel-close? (:fp1 ron) (:fp1 roff) 1e-5))
            (assert-true (str "samples-sum equivalent at 1e-4 rel ("
                              (:ssum ron) " vs " (:ssum roff) ")")
                         (rel-close? (:ssum ron) (:ssum roff) 1e-4))
            (assert-true (str "acceptance within one boundary flip ("
                              (:acc ron) " vs " (:acc roff) ")")
                         (<= (js/Math.abs (- (:acc ron) (:acc roff))) 0.05))
            (let [t-on (parse-tape on) t-off (parse-tape off)]
              (assert-true (str "duplication shrinks the post-fusion tape ("
                                t-on " < " t-off ")")
                           (and (pos? t-on) (pos? t-off) (< t-on t-off)))
              (assert-true (str "pass fired (clones=" (parse-clones on) ")")
                           (pos? (parse-clones on)))
              (assert-true "kill switch inert (OFF clones=0)"
                           (zero? (parse-clones off))))))
        (finish!)))))
