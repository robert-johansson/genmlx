;; @tier fast
(ns genmlx.program-golden-test
  "Byte-exact golden tests for genmlx.program analytical kernels.

   Guards the DRY refactor of program.cljs (bean genmlx-heal): extracting the
   shared sufficient-statistics log-ML kernel from score-all-structures and
   score-parent-sets-for-variable, reusing compute-posterior, and naming the
   runtime keyword builders. Every value below was captured from the pre-refactor
   code on FIXED, deterministic inputs (no randn / Math.random) and is asserted
   with EXACT = — the refactor is behavior-preserving only if all of these still
   hold bit-for-bit.

   Run: bun run --bun nbb test/genmlx/program_golden_test.cljs"
  (:require [genmlx.program :as prog]
            [genmlx.mlx :as mx]
            [genmlx.choicemap :as cm]))

;; ============================================================
;; Test harness
;; ============================================================

(def ^:private pass-count (atom 0))
(def ^:private fail-count (atom 0))

(defn- assert-true [msg v]
  (if v
    (do (swap! pass-count inc) (println "  PASS:" msg))
    (do (swap! fail-count inc) (println "  FAIL:" msg))))

(defn- assert-exact [msg expected actual]
  (assert-true (str msg (when-not (= expected actual)
                          (str " (expected " (pr-str expected) ", got " (pr-str actual) ")")))
               (= expected actual)))

(defn- report []
  (let [p @pass-count f @fail-count]
    (println (str "\n=== " p "/" (+ p f) " PASS ==="))
    (when (pos? f) (println (str "!!! " f " FAILURES !!!")))))

;; ============================================================
;; Fixed deterministic inputs
;; ============================================================

(def tx2
  [{:x-prev 1.0 :y-prev 2.0 :x-next 1.5 :y-next 2.5}
   {:x-prev 1.5 :y-prev 2.5 :x-next 1.8 :y-next 2.9}
   {:x-prev 1.8 :y-prev 2.9 :x-next 2.0 :y-next 3.1}
   {:x-prev 2.0 :y-prev 3.1 :x-next 2.3 :y-next 3.6}
   {:x-prev 2.3 :y-prev 3.6 :x-next 2.6 :y-next 4.0}])

(def edges-xy {["x" "y"] true ["y" "x"] false})
(def edges-none {["x" "y"] false ["y" "x"] false})

(def txk
  [{:prev {:a 1.0 :b 2.0 :c 0.5} :next {:a 1.2 :b 2.1 :c 0.7}}
   {:prev {:a 1.2 :b 2.1 :c 0.7} :next {:a 1.1 :b 2.3 :c 0.9}}
   {:prev {:a 1.1 :b 2.3 :c 0.9} :next {:a 1.4 :b 2.0 :c 1.1}}
   {:prev {:a 1.4 :b 2.0 :c 1.1} :next {:a 1.3 :b 2.4 :c 0.8}}
   {:prev {:a 1.3 :b 2.4 :c 0.8} :next {:a 1.6 :b 2.2 :c 1.0}}
   {:prev {:a 1.6 :b 2.2 :c 1.0} :next {:a 1.5 :b 2.5 :c 1.2}}])

(def kvars [:a :b :c])

(defn- cm->items [c]
  (into (sorted-map) (map (fn [[k v]] [k (mx/item v)]) (cm/to-map c))))

;; ============================================================
;; Golden values captured from pre-refactor program.cljs
;; ============================================================

(def GOLDEN
  {;; B — log-ml-variable (closed-form path, left untouched by the refactor)
   :b-xy-est   -16.879380883904552
   :b-none-est -16.63021629134939
   :b-xy-known -13.612325614243108

   ;; C — score-all-structures (shared suffstat kernel; 64 structures)
   :c-names ["independent" "a->b" "a->c" "a->b, a->c" "b->a" "a->b, b->a" "a->c, b->a" "a->b, a->c, b->a" "b->c" "a->b, b->c" "a->c, b->c" "a->b, a->c, b->c" "b->a, b->c" "a->b, b->a, b->c" "a->c, b->a, b->c" "a->b, a->c, b->a, b->c" "c->a" "a->b, c->a" "a->c, c->a" "a->b, a->c, c->a" "b->a, c->a" "a->b, b->a, c->a" "a->c, b->a, c->a" "a->b, a->c, b->a, c->a" "b->c, c->a" "a->b, b->c, c->a" "a->c, b->c, c->a" "a->b, a->c, b->c, c->a" "b->a, b->c, c->a" "a->b, b->a, b->c, c->a" "a->c, b->a, b->c, c->a" "a->b, a->c, b->a, b->c, c->a" "c->b" "a->b, c->b" "a->c, c->b" "a->b, a->c, c->b" "b->a, c->b" "a->b, b->a, c->b" "a->c, b->a, c->b" "a->b, a->c, b->a, c->b" "b->c, c->b" "a->b, b->c, c->b" "a->c, b->c, c->b" "a->b, a->c, b->c, c->b" "b->a, b->c, c->b" "a->b, b->a, b->c, c->b" "a->c, b->a, b->c, c->b" "a->b, a->c, b->a, b->c, c->b" "c->a, c->b" "a->b, c->a, c->b" "a->c, c->a, c->b" "a->b, a->c, c->a, c->b" "b->a, c->a, c->b" "a->b, b->a, c->a, c->b" "a->c, b->a, c->a, c->b" "a->b, a->c, b->a, c->a, c->b" "b->c, c->a, c->b" "a->b, b->c, c->a, c->b" "a->c, b->c, c->a, c->b" "a->b, a->c, b->c, c->a, c->b" "b->a, b->c, c->a, c->b" "a->b, b->a, b->c, c->a, c->b" "a->c, b->a, b->c, c->a, c->b" "a->b, a->c, b->a, b->c, c->a, c->b"]
   :c-log-mls [-17.89773910829456 -12.228468292776318 -14.39091671067874 -8.7216458951605 -13.024136636675115 -7.354865821156874 -9.517314239059296 -3.8480434235410548 -13.687149418818382 -8.017878603300142 -16.169044267775654 -10.499773452257411 -8.813546947198938 -3.1442761316806984 -11.295441796156208 -5.626170980637967 -15.556092463150387 -9.886821647632146 -12.049270065534566 -6.3799992500163265 -15.553515802302645 -9.884244986784402 -12.046693404686826 -6.377422589168583 -11.34550277367421 -5.67623195815597 -13.82739762263148 -8.15812680711324 -11.342926112826468 -5.673655297308228 -13.824820961783738 -8.155550146265497 -14.554443567834403 -13.96912421715451 -11.047621170218584 -10.462301819538691 -9.680841096214959 -9.095521745535066 -6.174018698599139 -5.5886993479192455 -10.343853878358226 -9.758534527678334 -12.825748727315496 -12.240429376635603 -5.470251406738782 -4.88493205605889 -7.952146255696052 -7.366826905016159 -12.21279692269023 -11.627477572010338 -8.70597452507441 -8.120655174394518 -12.210220261842487 -11.624900911162594 -8.703397864226666 -8.118078513546774 -8.002207233214055 -7.416887882534162 -10.484102082171324 -9.898782731491432 -7.999630572366311 -7.414311221686418 -10.48152542132358 -9.896206070643688]
   :c-edge-marginals {[:a :b] 0.9207716148153358, [:a :c] 0.3629954734170935, [:b :a] 0.9253129580017428, [:b :c] 0.6801564552764906, [:c :a] 0.13644497761828112, [:c :b] 0.2139318930973213}

   ;; compute-posterior (reuse target)
   :cp-posteriors [0.6652409557748217 0.09003057317038043 0.24472847105479759]

   ;; D — score-parent-sets-for-variable via discover-structure-decomposed
   :d-marginals {[:a :b] 0.9207716148153356, [:a :c] 0.3629954734170935, [:b :a] 0.9253129580017428, [:b :c] 0.6801564552764907, [:c :a] 0.13644497761828103, [:c :b] 0.21393189309732136}
   :d-per-var-logmls {:a [[[] -5.39265643018128] [["b"] -0.5190539585618352] [["c"] -3.051009785037108] [["b" "c"] -3.0484331241893643]]
                      :b [[[] -7.267972397525671] [["a"] -1.5987015820074313] [["c"] -3.9246768570655153] [["a" "c"] -3.3393575063856225]]
                      :c [[[] -5.237110280587608] [["a"] -1.7302878829717878] [["b"] -1.026520591111432] [["a" "b"] -3.508415440068701]]}

   ;; discover-structure (enum path, compute-posterior reuse target)
   :ds-best-name "a->b, b->a, b->c"
   :ds-best-logml -3.1442761316806984
   :ds-ranked-logmls [-3.1442761316806984 -3.8480434235410548 -4.88493205605889 -5.470251406738782 -5.5886993479192455 -5.626170980637967 -5.673655297308228 -5.67623195815597 -6.174018698599139 -6.377422589168583 -6.3799992500163265 -7.354865821156874 -7.366826905016159 -7.414311221686418 -7.416887882534162 -7.952146255696052 -7.999630572366311 -8.002207233214055 -8.017878603300142 -8.118078513546774 -8.120655174394518 -8.155550146265497 -8.15812680711324 -8.703397864226666 -8.70597452507441 -8.7216458951605 -8.813546947198938 -9.095521745535066 -9.517314239059296 -9.680841096214959 -9.758534527678334 -9.884244986784402 -9.886821647632146 -9.896206070643688 -9.898782731491432 -10.343853878358226 -10.462301819538691 -10.48152542132358 -10.484102082171324 -10.499773452257411 -11.047621170218584 -11.295441796156208 -11.342926112826468 -11.34550277367421 -11.624900911162594 -11.627477572010338 -12.046693404686826 -12.049270065534566 -12.210220261842487 -12.21279692269023 -12.228468292776318 -12.240429376635603 -12.825748727315496 -13.024136636675115 -13.687149418818382 -13.824820961783738 -13.82739762263148 -13.96912421715451 -14.39091671067874 -14.554443567834403 -15.553515802302645 -15.556092463150387 -16.169044267775654 -17.89773910829456]

   ;; keyword/constraint builders (obs-addr / next-key / sigma-key)
   :bc {:x0 1.5, :x1 1.7999999523162842, :x2 2, :x3 2.299999952316284, :x4 2.5999999046325684, :y0 2.5, :y1 2.9000000953674316, :y2 3.0999999046325684, :y3 3.5999999046325684, :y4 4}
   :bsc [{:x0 1.5, :y0 2.5}
         {:x0 1.5, :x1 1.7999999523162842, :y0 2.5, :y1 2.9000000953674316}
         {:x0 1.5, :x1 1.7999999523162842, :x2 2, :y0 2.5, :y1 2.9000000953674316, :y2 3.0999999046325684}
         {:x0 1.5, :x1 1.7999999523162842, :x2 2, :x3 2.299999952316284, :y0 2.5, :y1 2.9000000953674316, :y2 3.0999999046325684, :y3 3.5999999046325684}
         {:x0 1.5, :x1 1.7999999523162842, :x2 2, :x3 2.299999952316284, :x4 2.5999999046325684, :y0 2.5, :y1 2.9000000953674316, :y2 3.0999999046325684, :y3 3.5999999046325684, :y4 4}]
   :psel "#genmlx.selection.SelectAddrs{:addrs #{:ar-x :ar-y :beta-x->y :sigma-x :sigma-y}}"})

;; ============================================================
;; Assertions
;; ============================================================

(println "\n== B: score-model-analytical (closed-form, untouched) ==")
(assert-exact "log-ML xy estimated-sigma" (:b-xy-est GOLDEN)
              (prog/score-model-analytical tx2 [:x :y] edges-xy {}))
(assert-exact "log-ML none estimated-sigma" (:b-none-est GOLDEN)
              (prog/score-model-analytical tx2 [:x :y] edges-none {}))
(assert-exact "log-ML xy known-sigma" (:b-xy-known GOLDEN)
              (prog/score-model-analytical tx2 [:x :y] edges-xy {:sigma-x 0.5 :sigma-y 0.7}))

(println "\n== C: score-all-structures (shared suffstat kernel) ==")
(let [scored (prog/score-all-structures txk kvars)]
  (assert-exact "structure names" (:c-names GOLDEN) (mapv :name scored))
  (assert-exact "all 64 log-MLs" (:c-log-mls GOLDEN) (mapv :log-ml scored))
  (assert-exact "edge-marginals (compute-posterior reuse)"
                (:c-edge-marginals GOLDEN) (prog/edge-marginals kvars scored)))

(println "\n== compute-posterior ==")
(assert-exact "posteriors" (:cp-posteriors GOLDEN)
              (mapv :posterior (prog/compute-posterior
                                [{:name "a" :log-ml -5.0}
                                 {:name "b" :log-ml -7.0}
                                 {:name "c" :log-ml -6.0}])))

(println "\n== D: discover-structure-decomposed (shared suffstat kernel) ==")
(let [d (prog/discover-structure-decomposed txk kvars)]
  (assert-exact "marginals" (:d-marginals GOLDEN) (:marginals d))
  (assert-exact "per-variable parent-set log-MLs"
                (:d-per-var-logmls GOLDEN)
                (into {} (for [[v info] (:per-variable d)]
                           [v (mapv (juxt #(vec (sort (map name (:parents %)))) :log-ml)
                                    (:parent-sets info))]))))

(println "\n== discover-structure (enum path, compute-posterior reuse) ==")
(let [ds (prog/discover-structure kvars txk)]
  (assert-exact "best name" (:ds-best-name GOLDEN) (:name (:best ds)))
  (assert-exact "best log-ML" (:ds-best-logml GOLDEN) (:log-ml (:best ds)))
  (assert-exact "ranked log-MLs" (:ds-ranked-logmls GOLDEN) (mapv :log-ml (:ranked ds))))

(println "\n== builders: build-constraints / build-sequential-constraints / param-selection ==")
(assert-exact "build-constraints addrs+vals" (:bc GOLDEN)
              (cm->items (prog/build-constraints tx2 [:x :y])))
(assert-exact "build-sequential-constraints cumulative" (:bsc GOLDEN)
              (mapv cm->items (prog/build-sequential-constraints tx2 [:x :y])))
(assert-exact "param-selection addresses" (:psel GOLDEN)
              (pr-str (prog/param-selection [:x :y] edges-xy)))

(mx/force-gc!)
(report)
