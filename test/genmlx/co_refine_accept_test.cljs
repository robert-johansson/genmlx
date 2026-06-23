;; @tier slow
(ns genmlx.co-refine-accept-test
  "Acceptance for the CO-REFINED ACCEPT rule integrated into the production driver
   (R1 / genmlx-8smp, the R2 prerequisite). NATIVE-FREE — conjugate partial models score
   the EXACT analytical marginal, so every assertion is reproducible (no policy LLM).

   R-1 proved the multi-latent cliff is an ACCEPT-ORDERING bug, not a no-monotone-path
   wall: a structural edit is correct but, scored at its EMITTED σ (~1), sits in a deep
   evidence valley, so the strict ratchet rejects it before σ can rescue it. R1 validated
   the fix at the SCORING level (scripts/r1_accept_fix.cljs). THIS file pins the same fix
   inside the real `genmlx.world.synth/synthesize` driver — the production accept path the
   R2 bake-off and the harvest loop actually run:

     :co-refine-sigma? false  -> score each candidate at its emitted σ (legacy ratchet)
     :co-refine-sigma? true   -> score each candidate at its grid-best shared obs σ

   PARTS:
     A — incremental cliff (per-group-means, varying-slopes): a one-structural-edit-at-a-
         time proposer walks the crude->gold path. The strict ratchet STALLS at the first
         valley rung; the co-refined accept CLIMBS to gold (>= the solve bar).
     B — over-acceptance guard: on data from the SIMPLER model (shared-mean), the co-
         refined accept must NOT accept a spurious extra-latent split (Occam / the exact
         marginal likelihood penalizes the unwarranted latent). Without this the fix would
         launder over-fitting as a win.
     C — default-unchanged: with :co-refine-sigma? absent/false the driver behaves exactly
         as before (no silent behavior change on the legacy path).

   Run: bun run --bun nbb test/genmlx/co_refine_accept_test.cljs"
  (:require [genmlx.world.synth :as s]
            [genmlx.world.curriculum :as cur]))

(def ^:private pass (atom 0))
(def ^:private fail (atom 0))
(defn assert-true [label v]
  (if v (do (swap! pass inc) (println "  PASS" label))
      (do (swap! fail inc) (println "  FAIL" label))))
(defn- fx [x] (if (and x (js/isFinite x)) (.toFixed (js/Number x) 2) (str x)))

;; ---------------------------------------------------------------------------
;; The incremental crude->gold structural paths (byte-identical to r1_accept_fix):
;; rung 0 = crude (shared mean, σ 3.0); rung k = the structure added one group at a time
;; (σ 1.0, the scale a structure-focused proposer emits). The VALLEY is in the
;; intermediate rungs at σ=1.0; a one-rung-at-a-time proposer must pass through them.
;; ---------------------------------------------------------------------------
(defn- lin-mean [s-sym x i-sym]
  (list 'mx/add (list 'mx/multiply s-sym (list 'mx/scalar x)) i-sym))

(defn pgm-path [{:keys [groups pp]}]
  (vec (for [k (range (inc (count groups)))]
         (let [own (set (take k groups)) shared? (some (complement own) groups)
               sig (if (zero? k) 3.0 1.0)]
           (s/spec (concat (when shared? [(s/latent 'mu "gaussian" [0 5])])
                           (for [gn groups :when (own gn)] (s/latent (symbol (str "m-" gn)) "gaussian" [0 5])))
                   (for [gn groups i (range pp)]
                     (s/obs (keyword (str gn i)) "gaussian"
                            [(if (own gn) (symbol (str "m-" gn)) 'mu) sig])))))))

(defn vs-path [{:keys [groups pp]}]
  (vec (for [k (range (inc (count groups)))]
         (let [own (set (take k groups)) shared? (some (complement own) groups)
               sig (if (zero? k) 3.0 1.0)]
           (s/spec (concat (when shared? [(s/latent 'mu "gaussian" [0 5])])
                           (mapcat (fn [gn] [(s/latent (symbol (str "s-" gn)) "gaussian" [0 3])
                                             (s/latent (symbol (str "i-" gn)) "gaussian" [0 5])])
                                   (filter own groups)))
                   (for [gn groups x (range pp)]
                     (s/obs (keyword (str gn x)) "gaussian"
                            [(if (own gn) (lin-mean (symbol (str "s-" gn)) x (symbol (str "i-" gn))) 'mu) sig])))))))

;; A one-rung-at-a-time proposer: it locates the current spec on `path` by its set of
;; LATENT addresses (σ lives on obs, so co-refinement does not change which rung we are on)
;; and offers ONLY the next rung. This is the faithful analogue of an LLM that adds one
;; piece of structure per step — so the strict ratchet meets the intermediate valley.
(defn- lat-addrs [spec] (set (map :addr (:latents spec))))
(defn path-proposer [path]
  (let [by-addrs (into {} (map-indexed (fn [i sp] [(lat-addrs sp) i]) path))]
    (fn [spec _fb]
      (when-let [i (by-addrs (lat-addrs spec))]
        (when (< (inc i) (count path))
          [{:edit :rung :desc (str "rung->" (inc i)) :spec' (nth path (inc i))}])))))

(defn- run [path obs co-refine?]
  (s/synthesize {:init-spec (first path) :observations obs :propose (path-proposer path)
                 :max-steps 10 :plateau-eps 0.05 :n-particles 2000
                 :co-refine-sigma? co-refine?}))

;; ---------------------------------------------------------------------------
(def C (cur/generate-curriculum {:round 0 :instances-per-family 12}))
(defn- a-task [fam] (first (filter #(= fam (:family %)) (:tasks C))))

(println "\n=== PART A — incremental cliff: strict ratchet STALLS, co-refine CLIMBS ===")
(doseq [[label fam pathfn] [["per-group-means" :per-group-means pgm-path]
                            ["varying-slopes (the 0/9 cliff)" :varying-slopes vs-path]]]
  (let [task   (a-task fam)
        obs    (:observations task)
        bar    (:solve-bar task)
        path   (pathfn (:true-params task))
        strict (run path obs false)
        coref  (run path obs true)
        se     (:evidence (:feedback strict))
        ce     (:evidence (:feedback coref))]
    (println (str "\n  " label "  (bar=" (fx bar) ", " (count path) " rungs)"))
    (println (str "    STRICT    final=" (fx se) "  steps=" (:steps strict) " stop=" (name (:stop-reason strict))
                  "  solved=" (boolean (and se (>= se bar)))))
    (println (str "    CO-REFINE final=" (fx ce) "  steps=" (:steps coref) " stop=" (name (:stop-reason coref))
                  "  solved=" (boolean (and ce (>= ce bar)))))
    (assert-true (str label ": strict ratchet does NOT reach the bar (stalls in the valley)")
                 (not (and se (>= se bar))))
    (assert-true (str label ": co-refined accept REACHES the bar (climbs to gold)")
                 (boolean (and ce (>= ce bar))))
    (assert-true (str label ": co-refine strictly beats strict") (and ce se (> ce (+ se 0.05))))
    (assert-true (str label ": co-refine accepted >= 1 structural edit") (>= (:steps coref) 1))))

(println "\n=== PART B — over-acceptance guard: co-refine must NOT add a spurious latent ===")
(let [task  (a-task :shared-mean)
      obs   (:observations task)
      addrs (vec (keys obs))
      half  (quot (count addrs) 2)
      crude (s/spec [(s/latent 'mu "gaussian" [0 5])] (for [k addrs] (s/obs k "gaussian" ['mu 1.0])))
      ;; a spurious split of the (structureless) data into two arbitrary groups with own means
      split (s/spec [(s/latent 'mu1 "gaussian" [0 5]) (s/latent 'mu2 "gaussian" [0 5])]
                    (map-indexed (fn [i k] (s/obs k "gaussian" [(if (< i half) 'mu1 'mu2) 1.0])) addrs))
      prop  (fn [spec _fb] (when (= 1 (count (:latents spec))) [{:edit :spurious :desc "split" :spec' split}]))
      res   (s/synthesize {:init-spec crude :observations obs :propose prop
                           :max-steps 4 :plateau-eps 0.05 :n-particles 2000 :co-refine-sigma? true})
      n-lat (count (:latents (:spec res)))]
  (println (str "  shared-mean data: co-refine driver ended with " n-lat
                " latent(s), stop=" (name (:stop-reason res)) ", evidence=" (fx (:evidence (:feedback res)))))
  (assert-true "co-refine does NOT accept the spurious 2-latent split (Occam holds)" (= 1 n-lat))
  (assert-true "co-refine plateaus at the crude shared-mean model" (= :plateau (:stop-reason res))))

(println "\n=== PART C — default (co-refine off) is unchanged: legacy ratchet behavior ===")
(let [task (a-task :per-group-means)
      obs  (:observations task)
      path (pgm-path (:true-params task))
      off1 (run path obs false)
      off2 (s/synthesize {:init-spec (first path) :observations obs :propose (path-proposer path)
                          :max-steps 10 :plateau-eps 0.05 :n-particles 2000})] ;; co-refine absent
  (assert-true "co-refine-sigma? false == co-refine-sigma? absent (no silent default change)"
               (= (fx (:evidence (:feedback off1))) (fx (:evidence (:feedback off2))))))

;; ---------------------------------------------------------------------------
(println (str "\n================  " @pass " passed, " @fail " failed  ================"))
(when (pos? @fail) (js/process.exit 1))
