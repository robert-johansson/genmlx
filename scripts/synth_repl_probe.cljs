(ns synth-repl-probe
  "EXPERIMENT (B) (beans genmlx-n74t / genmlx-77vv): does the minimal REPL synthesis
   LOOP solve an advanced model that whole-program best-of-16 got 0/16 on?

   For each of the three EXACT-scoreable advanced models (linreg-coupled / kalman-chain
   / hier-group-means) we start the genmlx.world.synth driver from a CRUDE covering
   model (the crudest rung of the experiment-A ladder) and give it a STRUCTURED move
   vocabulary — the structural upgrades a programmer would try (add the slope+intercept
   structure, couple the latent chain, split into per-group means) PLUS a distractor
   and the alternative structure (full-vs-no-intercept, independent-vs-hierarchical
   groups) — and let the EXACT oracle SELECT which edit to keep and WHEN to stop. No
   LLM: this isolates the load-bearing claim that a feedback loop with an exact verifier
   turns a 0/16 whole-program problem into a sequence of locally checkable edits.
   (Phase 2 SEARCHES this vocabulary; Phase 3 LEARNS the proposer.)

   We report, per model: the full per-step trajectory (edit, evidence, delta, method),
   the loop's final evidence vs the crude start and the whole-program best-of-16
   baseline, and that the oracle's selection is discriminating (it rejects the
   distractor; for hier it prefers INDEPENDENT over hierarchical groups, the
   higher-evidence structure for these well-separated groups — experiment A). NB the
   driver is GREEDY: it commits to the first improving structural move, so e.g. linreg
   locks into the no-intercept branch — a greedy local optimum, NOT a global-Occam
   claim; joint structure+noise search (Phase 2 SMC) may prefer a fuller model.

   Run: bun run --bun nbb scripts/synth_repl_probe.cljs"
  (:require [genmlx.world.synth :as syn]
            [clojure.string :as str]))

(def os   (js/require "os"))
(def path (js/require "path"))
(def fs   (js/require "fs"))
(defn home [& xs] (apply (.-join path) (.homedir os) xs))
(def out-dir (home "genmlx-loop-artifacts" "particle"))

(defn fx [x] (when (and x (js/isFinite x)) (.toFixed (js/Number x) 2)))

;; ---------------------------------------------------------------------------
;; Small spec builders shared by the per-task proposers.
;; ---------------------------------------------------------------------------

(defn- cur-noise [sp] (last (:args (first (:obs sp)))))   ; current obs noise scale
(defn- has-latent? [sp sym] (some #(= sym (:sym %)) (:latents sp)))
(defn- mx-mul [a b] (list 'mx/multiply a b))
(defn- mx-add [a b] (list 'mx/add a b))
(defn- mx-scalar [x] (list 'mx/scalar x))

;; A SHARED-noise refinement: set EVERY obs site to the same scale g. One shared
;; scale is a single hyperparameter (type-II maximum-marginal-likelihood over it), so
;; the loop picks the best grid value in ONE step — keeping the trajectory focused on
;; the STRUCTURAL step rather than a long noise-tuning tail, and matching the
;; shared-sigma rungs of experiment A. The grid is a COARSE, hand-provided vocabulary
;; (Phase 2 searches it; Phase 3 learns the proposer); the loop stops on the evidence
;; plateau over whatever grid it is given, so a finer grid would land slightly higher.
(defn- set-all-noise [sp g] (reduce #(syn/set-noise %1 %2 g) sp (map :addr (:obs sp))))
(defn- shared-noise-moves [sp grid]
  (for [g grid :when (not= g (cur-noise sp))]
    {:edit :set-noise :desc (str "shared obs noise -> " g) :spec' (set-all-noise sp g)}))

;; ===========================================================================
;; LINREG — x=0..6, y ~ linear (slope~1, intercept~1). Crude start: one shared
;; mean. Structural move: introduce slope (+ optional intercept) and re-point each
;; obs at slope*xj(+intercept). Occam test: the no-intercept model out-scores the
;; full one for these priors (experiment A), so the oracle should PREFER it.
;; ===========================================================================

(def linreg-obs {:y0 1.1 :y1 2.0 :y2 2.7 :y3 4.2 :y4 4.8 :y5 6.1 :y6 6.9})
(def linreg-ys  [:y0 :y1 :y2 :y3 :y4 :y5 :y6])

(def linreg-init
  (syn/spec [(syn/latent 'mu "gaussian" [0 10])]
            (for [k linreg-ys] (syn/obs k "gaussian" ['mu 2.5]))))

(defn- linreg-build [intercept? noise]
  (let [mean (fn [j] (if intercept?
                       (mx-add (mx-mul 'slope (mx-scalar (double j))) 'intercept)
                       (mx-mul 'slope (mx-scalar (double j)))))]
    (syn/spec (cond-> [(syn/latent 'slope "gaussian" [0 3])]
                intercept? (conj (syn/latent 'intercept "gaussian" [0 5])))
              (map-indexed (fn [j k] (syn/obs k "gaussian" [(mean j) noise])) linreg-ys))))

(defn linreg-propose [sp _fb]
  (let [n (cur-noise sp)]
    (if-not (has-latent? sp 'slope)
      ;; structural step: offer full-linear, no-intercept-linear, and a distractor
      ;; (tighten the shared-mean's noise WITHOUT fixing the structure).
      [{:edit :add-linear        :desc "add slope+intercept, linear mean" :spec' (linreg-build true n)}
       {:edit :add-linear-noint  :desc "add slope only (no intercept)"    :spec' (linreg-build false n)}
       {:edit :distractor-tighten :desc "tighten shared-mean noise -> 1 (no structure)"  :spec' (set-all-noise sp 1)}]
      ;; refinement: tune the (now linear) observation noise (shared scale).
      (shared-noise-moves sp [0.3 0.5 0.7 1 1.5 2.5]))))

;; ===========================================================================
;; KALMAN — smooth series; AR(1) coupling is the structural step. Crude start: iid
;; latents. Structural move: couple z_t ~ N(coef*z_{t-1}, 0.5) over a coef grid.
;; ===========================================================================

(def kalman-obs {:y0 0.4 :y1 0.9 :y2 1.3 :y3 1.1 :y4 0.6})

(def kalman-init
  (syn/spec (for [t (range 5)] (syn/latent (symbol (str "z" t)) "gaussian" [0 1]))
            (for [t (range 5)] (syn/obs (keyword (str "y" t)) "gaussian" [(symbol (str "z" t)) 0.7]))))

(defn- kalman-build [coef proc-noise]
  (syn/spec (cons (syn/latent 'z0 "gaussian" [0 1])
                  (for [t (range 1 5)]
                    (syn/latent (symbol (str "z" t)) "gaussian"
                                [(mx-mul (mx-scalar coef) (symbol (str "z" (dec t)))) proc-noise])))
            (for [t (range 5)] (syn/obs (keyword (str "y" t)) "gaussian" [(symbol (str "z" t)) 0.7]))))

(defn- coupled? [sp] (seq? (first (:args (second (:latents sp))))))  ; z1 mean is a form

(defn kalman-propose [sp _fb]
  (if-not (coupled? sp)
    ;; structural step: offer AR(1) coupling over a coefficient grid (0.3 is a weak,
    ;; wrong coupling distractor; 0.9 is the data-warranted one).
    (for [c [0.3 0.6 0.9 1.0]]
      {:edit :couple :desc (str "AR(1) couple, coef " c) :spec' (kalman-build c 0.5)})
    ;; refinement: tune the process noise (a single shared scale).
    (for [pn [0.1 0.2 0.3 0.4 0.5 0.7 1.0]]
      {:edit :set-proc-noise :desc (str "process noise -> " pn)
       :spec' (syn/spec (cons (syn/latent 'z0 "gaussian" [0 1])
                              (for [t (range 1 5)]
                                (syn/latent (symbol (str "z" t)) "gaussian"
                                            [(first (:args (nth (:latents sp) t))) pn])))
                        (:obs sp))})))

;; ===========================================================================
;; HIERARCHICAL — 3 well-separated groups. Crude start: single shared mean.
;; Structural move: per-group means (independent) OR partial-pooled (hierarchical).
;; Occam test: for well-separated groups the INDEPENDENT model out-scores the
;; hierarchical one (experiment A), so the oracle should prefer it.
;; ===========================================================================

(def hier-obs {:a0 5.2 :a1 4.8 :a2 5.5 :b0 -1.1 :b1 -0.7 :b2 -1.4 :c0 2.1 :c1 2.6 :c2 1.9})
(def hier-groups {'mu-a [:a0 :a1 :a2] 'mu-b [:b0 :b1 :b2] 'mu-c [:c0 :c1 :c2]})

(def hier-init
  (syn/spec [(syn/latent 'mu "gaussian" [0 5])]
            (for [k (keys hier-obs)] (syn/obs k "gaussian" ['mu 2]))))

(defn- hier-build [pooled? noise]
  (syn/spec (cond-> (for [g (keys hier-groups)]
                      (syn/latent g "gaussian" (if pooled? ['grand 2] [0 5])))
              pooled? (->> (cons (syn/latent 'grand "gaussian" [0 5]))))
            (for [[g ks] hier-groups, k ks] (syn/obs k "gaussian" [g noise]))))

(defn hier-propose [sp _fb]
  (if-not (has-latent? sp 'mu-a)
    [{:edit :indep-groups  :desc "per-group independent means" :spec' (hier-build false 1)}
     {:edit :hierarchical  :desc "partial-pooled (hierarchical) means" :spec' (hier-build true 1)}]
    (shared-noise-moves sp [0.3 0.5 0.7 1 1.5 2])))

;; ===========================================================================
;; Run the loop on each task; report the trajectory + the comparison.
;; ===========================================================================

;; :ref-rung = the experiment-A rung for the CORRECT structure at fixed noise=1
;; (no-intercept linear / AR(0.9) / independent groups). It is a REPORTING reference,
;; NOT the success gate: it lets us see the structural evidence the loop reaches. The
;; loop typically ends a bit ABOVE it because it also tunes the shared observation
;; noise (type-II maximum-marginal-likelihood over ONE fixed scale the rung held at 1)
;; — that is hyperparameter fitting, not added model structure, so it is not directly
;; comparable to the rung; the STRUCTURAL step is the result.
;;
;; :bestof16 is the whole-program baseline from the PRIOR run (NOT recomputed here):
;; particle_advanced_probe on the SAME data + SAME oracle — the student got 0/16 on
;; all four advanced models; the teacher got 1/4 and that one fit at ~ -35 (junk).
;; See ~/genmlx-loop-artifacts/particle/advanced_probe.json. The :obs below are
;; byte-identical to that probe's :observations, so the comparison is apples-to-apples.
(def tasks
  [{:id :linreg-coupled :init linreg-init :obs linreg-obs :propose linreg-propose
    :ref-rung -11.05 :bestof16 "student 0/16 ; teacher fit -35 (junk)"}
   {:id :kalman-chain :init kalman-init :obs kalman-obs :propose kalman-propose
    :ref-rung -5.37 :bestof16 "student 0/16"}
   {:id :hier-group-means :init hier-init :obs hier-obs :propose hier-propose
    :ref-rung -15.79 :bestof16 "student 0/16"}])

;; The STRUCTURAL edits — the key structure the crude covering model lacks and that
;; whole-program best-of-16 produced 0 valid candidates for.
(def structural-edits #{:add-linear :add-linear-noint :couple :indep-groups :hierarchical})
(defn- structural-step? [traj] (boolean (some #(structural-edits (:edit %)) traj)))

;; "solved" = the loop ACCEPTED the key structural edit (driven by exact oracle
;; evidence, over distractors + the structural alternative), IMPROVED on the crude
;; start, and SELF-TERMINATED on an evidence plateau (not :stuck / :max-steps). We do
;; NOT count the monotone climb as a criterion: the driver only appends an improving
;; step, so a monotone trajectory is true BY CONSTRUCTION, not evidence of success.
(defn- solved? [{:keys [start final trajectory stop-reason]}]
  (and (structural-step? trajectory)
       (> final start)
       (= :plateau stop-reason)))

(defn run-task [{:keys [id init obs propose ref-rung bestof16]}]
  (println (str "\n================  " (name id) "  ================"))
  (let [res  (syn/synthesize {:init-spec init :observations obs :propose propose
                              :max-steps 12 :plateau-eps 0.05 :n-particles 4000})
        traj (:trajectory res)
        start (:evidence (first traj))
        final (:evidence (last traj))
        row  {:id id :start start :final final :ref-rung ref-rung
              :stop-reason (:stop-reason res) :steps (:steps res)
              :trajectory (mapv #(select-keys % [:step :edit :desc :evidence :delta :method]) traj)
              :final-code (:code res)}]
    (println (str (.padEnd "step" 6) (.padEnd "edit" 24) (.padEnd "evidence" 11)
                  (.padEnd "delta" 9) "method"))
    (doseq [r traj]
      (println (str (.padEnd (str (:step r)) 6)
                    (.padEnd (str (name (:edit r))) 24)
                    (.padEnd (or (fx (:evidence r)) "--") 11)
                    (.padEnd (or (fx (:delta r)) "--") 9)
                    (str (some-> (:method r) name)))))
    (println (str "\n  stop-reason: " (name (:stop-reason res)) " after " (:steps res) " accepted edits"))
    (println (str "  structural edit accepted: " (structural-step? traj)
                  " (the structure the crude model + whole-program best-of-16 lacked)"))
    (println (str "  crude start evidence    : " (fx start)))
    (println (str "  LOOP final evidence     : " (fx final)
                  "   (vs exp-A fixed-noise=1 structural rung ~ " ref-rung
                  "; any excess = shared-noise type-II ML, not added structure)"))
    (println (str "  whole-program best-of-16: " bestof16))
    (println (str "  => loop improvement over crude start: " (fx (- final start)) " nats"
                  "   SOLVED? " (solved? row)))
    (println (str "  final model:\n   " (:code res)))
    row))

(println "\n###  EXPERIMENT (B): does the REPL loop solve an advanced model?  ###")
(let [results (mapv run-task tasks)]
  (println "\n================  VERDICT  ================")
  (println (str (.padEnd "model" 20) (.padEnd "start" 9) (.padEnd "final" 9)
                (.padEnd "struct?" 9) (.padEnd "stop" 9) "solved?"))
  (doseq [r results]
    (println (str (.padEnd (name (:id r)) 20)
                  (.padEnd (str (fx (:start r))) 9)
                  (.padEnd (str (fx (:final r))) 9)
                  (.padEnd (str (structural-step? (:trajectory r))) 9)
                  (.padEnd (name (:stop-reason r)) 9)
                  (solved? r))))
  (let [n-solved (count (filter solved? results))]
    (println (str "\n  => the REPL loop SOLVES " n-solved "/" (count results)
                  " advanced models: from a crude covering model it ACCEPTS the key"
                  " structural edit (oracle-selected over distractors + the structural"
                  " alternative) and self-terminates — where whole-program best-of-16 got 0/16."))
    (println "\n  HONEST CAVEATS (this isolates the loop, it is not the whole story):")
    (println "   - the proposer is a STRUCTURED move-vocabulary, NOT an LLM — deliberately, to")
    (println "     isolate the loop-vs-cliff claim from generation noise (Phase 3 learns the proposer).")
    (println "   - GREEDY commits to the first improving STRUCTURAL move and cannot reconsider it")
    (println "     (e.g. linreg locks into no-intercept; a full-intercept model may score higher under")
    (println "     JOINT structure+noise search) -> exactly why Phase 2 is SMC/beam over edits.")
    (println "   - the shared-noise refinement is type-II ML over ONE fixed scale (hyperparameter")
    (println "     fitting, not structure); the STRUCTURAL jump is the result, not the noise tuning.")
    (when-not (.existsSync fs out-dir) (.mkdirSync fs out-dir #js {:recursive true}))
    (.writeFileSync fs (home "genmlx-loop-artifacts" "particle" "synth_repl_probe.json")
                    (js/JSON.stringify (clj->js {:results results :n-solved n-solved
                                                 :n-tasks (count results)}) nil 2))
    (println "  wrote particle/synth_repl_probe.json")))
