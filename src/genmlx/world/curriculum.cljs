(ns genmlx.world.curriculum
  "Phase-3 enabler for the north star (beans genmlx-77vv / genmlx-ilna): the
   REPL-synthesis CURRICULUM GENERATOR — a graded, oracle-grounded bank of
   STRUCTURED-MODEL-FITTING tasks whose value is the multi-step construction.

   WHY (the bottleneck). Phase 3 (genmlx-oexl) learns the cheap proposer (the
   :edit distribution of the programmer-GF) by SFT on REPL traces, and a learned
   proposer needs traces AT SCALE — but the north-star probe only has ~4 hand-built
   tasks. This namespace produces ~100-200 reproducible tasks so the 35B loop can
   run each → genmlx.world.repl-corpus harvests propose-eval-revise transitions →
   SFT the 0.8B. Its COMPLEXITY SPREAD (one-shot-ties → loop-wins) is also the
   resource-rational training signal the Phase-4 metareasoner allocates over.

   DISTINCT from genmlx.world.distill-gen (the whole-program distillation task-gen,
   bean genmlx-7473). distill-gen grades a single conjugate covering model; this
   grades how much MULTI-STEP STRUCTURE the data warrants, the thing the loop is for.

   ORACLE-GROUNDED, NO LLM (native-free in the synth sense — it reuses synth/check
   and never loads the policy LLM). Data is SAMPLED from a known ground-truth
   structured model with a pure seeded PRNG; difficulty (crude / gold / solve-bar)
   is set by the EXACT Bayesian model-evidence oracle (synth/check → the analytical
   p/generate marginal), not a judge. Seeded ⇒ byte-reproducible.

   WHAT THE FAMILIES ARE (all LINEAR-GAUSSIAN ⇒ exactly scoreable, verified):
     :shared-mean      (c1)  repeated measurements of one quantity
     :linear           (c1)  response linear in an input
     :per-group-means  (c2)  independent group levels
     :ar1              (c2)  a smoothly-evolving latent chain (Kalman)
     :segmented        (c2)  two independent regimes (a composition; HELD-OUT family)
     :varying-slopes   (c3)  an independent line per group (the loop-wins regime)
   Centered hierarchical / shared-slope (ancova) families are EXCLUDED: the L3
   eliminator misroutes a latent-in-latent prior to a broken :kalman path and a
   shared latent coupling many obs to a non-finite marginal (bug genmlx-iraj). A
   reproducibility guard at calibration rejects any family the oracle cannot score
   deterministically + exactly, so the exclusion is enforced, not merely assumed.

   Sections:
     0. Pure seeded PRNG          — mulberry32 + Box–Muller, eager + reproducible
     1. Rendering helpers          — mx form constructors + crude covering model
     2. Family templates           — the structured vocabulary as data
     3. Calibration                — crude / gold / solve-bar via the exact oracle
     4. Instance generation        — sample → data → calibrate → well-posed record
     5. Curriculum assembly + split — graded bank, leakage-safe task-level holdout
     6. Consumer adapters          — ->probe-task + the no-LLM family-proposer"
  (:require [genmlx.world.synth :as syn]
            [clojure.string :as str]))

;; ===========================================================================
;; 0. Pure seeded PRNG — mulberry32 uniforms + Box–Muller normals.
;;
;; NATIVE-FREE and independent of MLX's RNG, so the sampled data is byte-identical
;; across runs and toolchains for a given seed. CLJS bit-ops are SIGNED 32-bit, so
;; every shift is `unsigned-bit-shift-right` and the final word is coerced unsigned
;; before the divide; multiplies use `js/Math.imul` (true 32-bit). Outputs are EAGER
;; vectors built with loop/recur — no lazy seq, so chunking can never desync a stream.
;; ===========================================================================

(defn- u32
  "Coerce x to an unsigned 32-bit double in [0, 2^32)."
  [x]
  (unsigned-bit-shift-right x 0))

(defn uniforms
  "An EAGER vector of `n` mulberry32 uniforms in [0,1) from 32-bit `seed`.
   Reference mulberry32 (Tommy Ettinger), transcribed with unsigned shifts."
  [seed n]
  (loop [a   (bit-or seed 0)
         i   0
         acc (transient [])]
    (if (>= i n)
      (persistent! acc)
      (let [a (bit-or (+ a 0x6D2B79F5) 0)
            t a
            t (js/Math.imul (bit-xor t (unsigned-bit-shift-right t 15)) (bit-or t 1))
            t (bit-xor (+ t (js/Math.imul (bit-xor t (unsigned-bit-shift-right t 7))
                                          (bit-or t 61)))
                       t)
            r (bit-xor t (unsigned-bit-shift-right t 14))]
        (recur a (inc i) (conj! acc (/ (u32 r) 4294967296.0)))))))

(defn normals
  "An EAGER vector of `n` N(0,1) draws from `seed` via Box–Muller. Consumes 2
   uniforms per normal in fixed pairs (the cos branch); u1 is clamped to (0,1] so
   log(0) cannot produce a NaN. Deterministic and reproducible."
  [seed n]
  (let [us (uniforms seed (* 2 (max 1 n)))]
    (vec (for [k (range n)]
           (let [u1 (max (nth us (* 2 k)) 1e-12)
                 u2 (nth us (inc (* 2 k)))]
             (* (js/Math.sqrt (* -2.0 (js/Math.log u1)))
                (js/Math.cos (* 2.0 js/Math.PI u2))))))))

(defn mix-seed
  "Fold a sequence of ints into one 32-bit seed (mulberry32 mixing). Injective enough
   for distinct (round, family, instance, role) tuples and overflow-safe."
  [& xs]
  (reduce (fn [h x]
            (let [h (bit-or (+ h (bit-or x 0) 0x9E3779B9) 0)
                  h (js/Math.imul (bit-xor h (unsigned-bit-shift-right h 16)) 0x85EBCA6B)
                  h (js/Math.imul (bit-xor h (unsigned-bit-shift-right h 13)) 0xC2B2AE35)]
              (bit-xor h (unsigned-bit-shift-right h 16))))
          0x2545F491 xs))

(defn round1
  "Round to one decimal — the established probe-data convention. Keeps :observations
   canonical/hashable and the SFT corpus low-entropy."
  [x]
  (/ (js/Math.round (* (double x) 10.0)) 10.0))

;; ===========================================================================
;; 1. Rendering helpers + the crude covering model.
;; ===========================================================================

(defn- mul [a b] (list 'mx/multiply a b))
(defn- add [a b] (list 'mx/add a b))
(defn- sc  [x]   (list 'mx/scalar (double x)))

(def ^:const crude-sigma
  "The fixed, untuned noise of the crude covering model — the structureless baseline
   the solve-bar is measured against. True sigmas are sampled strictly below it so a
   real gap always exists."
  3.0)

(def sigma-grid
  "The shared-σ refinement grid the loop's hybrid uses (genmlx.world.synth /
   synth_llm_probe). gold = the best-fitting correct-structure model over this grid."
  [0.1 0.2 0.3 0.5 0.7 1.0 1.5 2.5])

(defn crude-spec
  "The crude covering model for any task: one shared latent, every observed address an
   obs site at the fixed crude σ. Identical role to the loop's init-spec."
  [observations]
  (syn/spec [(syn/latent 'mu "gaussian" [0 10])]
            (for [k (keys observations)] (syn/obs k "gaussian" ['mu crude-sigma]))))

;; ===========================================================================
;; 2. Family templates — the structured vocabulary as data.
;;
;; Each template is a map of pure fns over a sampled `params` (true continuous values
;; + structural knobs + the ordered obs addresses). `:gold` interprets its `noise` arg
;; as the family's single tunable scale (process noise for :ar1, observation σ
;; elsewhere). `:complexity` is the structural-decision BAND (the loop-vs-one-shot
;; proxy); `:n-latents` the latent count (a continuous proxy). The number of GROUP /
;; POINT knobs is sampled per instance for variety.
;; ===========================================================================

(defn- pick-int [u lo hi] (+ lo (int (* u (inc (- hi lo))))))   ; uniform int in [lo,hi]
(def ^:private group-names ["a" "b" "c" "d"])

(defn- sample-sigma
  "A true observation σ in [0.3, 1.5] — comfortably below crude-sigma so the gap is
   real for every family (incl. :shared-mean, whose only lever is the scale)."
  [u]
  (round1 (+ 0.3 (* u 1.2))))

(def family-defs
  "The template registry. Order is the family index used in seed mixing + split."
  [;; --- :shared-mean (c1) — repeated measurements of one quantity ---
   ;; The ONLY non-structural family: its gold IS the crude shape (one shared latent),
   ;; so its difficulty is purely the noise scale and :struct-gap is ~0 by design. The
   ;; :structural? false flag exempts it from the min-struct-gap well-posedness gate.
   {:family :shared-mean :complexity 1 :structural? false
    :sample (fn [seed]
              (let [us (uniforms seed 3)
                    n  (pick-int (nth us 0) 4 7)
                    mu (round1 (* 6.0 (- (nth us 1) 0.5)))]
                {:n n :mu mu :sigma (sample-sigma (nth us 2))
                 :addrs (mapv #(keyword (str "y" %)) (range n))}))
    :data (fn [{:keys [n mu sigma addrs]} seed]
            (let [z (normals (mix-seed seed 991) n)]
              (zipmap addrs (map #(round1 (+ mu (* sigma %))) z))))
    :gold (fn [{:keys [addrs]} noise]
            (syn/spec [(syn/latent 'mu "gaussian" [0 10])]
                      (for [k addrs] (syn/obs k "gaussian" ['mu noise]))))
    :n-latents (fn [_] 1)
    :desc (fn [{:keys [n]}]
            (str "You have " n " repeated measurements y0.." (dec n)
                 " of a single quantity taken under identical conditions. "
                 "Model the quantity that produced them."))}

   ;; --- :linear (c1) — response linear in an input ---
   {:family :linear :complexity 1
    :sample (fn [seed]
              (let [us (uniforms seed 4)
                    n  (pick-int (nth us 0) 5 8)
                    slope (round1 (* (if (< (nth us 2) 0.5) -1 1) (+ 0.6 (* (nth us 1) 1.8))))
                    intercept (round1 (* 4.0 (- (nth us 3) 0.5)))]
                {:n n :slope slope :intercept intercept
                 :sigma (sample-sigma (nth (uniforms (mix-seed seed 5) 1) 0))
                 :xs (vec (range n)) :addrs (mapv #(keyword (str "y" %)) (range n))}))
    :data (fn [{:keys [slope intercept sigma xs addrs]} seed]
            (let [z (normals (mix-seed seed 991) (count xs))]
              (zipmap addrs (map (fn [x zi] (round1 (+ (* slope x) intercept (* sigma zi)))) xs z))))
    :gold (fn [{:keys [xs addrs]} noise]
            (syn/spec [(syn/latent 'slope "gaussian" [0 3]) (syn/latent 'intercept "gaussian" [0 5])]
                      (map (fn [x k] (syn/obs k "gaussian" [(add (mul 'slope (sc x)) 'intercept) noise]))
                           xs addrs)))
    :n-latents (fn [_] 2)
    :desc (fn [{:keys [n]}]
            (str "You have " n " observations y0.." (dec n) ". Observation yj is the response "
                 "measured at input x = j (so x = 0,1,.." (dec n) "). Model how the response "
                 "depends on the input x."))}

   ;; --- :per-group-means (c2) — independent group levels ---
   {:family :per-group-means :complexity 2
    :sample (fn [seed]
              (let [us (uniforms seed 2)
                    g  (pick-int (nth us 0) 2 4)
                    pp (pick-int (nth us 1) 3 4)
                    ;; group values spread out (well separated) so the structure wins
                    vs (mapv (fn [k] (round1 (* 4.0 (- k (/ (dec g) 2.0)))))
                             (range g))
                    gs (subvec group-names 0 g)]
                {:g g :pp pp :values vs :groups gs
                 :sigma (sample-sigma (nth (uniforms (mix-seed seed 5) 1) 0))
                 :addrs (vec (for [gn gs i (range pp)] (keyword (str gn i))))}))
    :data (fn [{:keys [g pp values groups sigma addrs]} seed]
            (let [z (normals (mix-seed seed 991) (* g pp))]
              (zipmap addrs
                      (map-indexed (fn [idx k]
                                     (let [gi (quot idx pp)]
                                       (round1 (+ (nth values gi) (* sigma (nth z idx))))))
                                   addrs))))
    :gold (fn [{:keys [groups pp] :as p} noise]
            (syn/spec (for [gn groups] (syn/latent (symbol (str "m-" gn)) "gaussian" [0 5]))
                      (for [gn groups i (range pp)]
                        (syn/obs (keyword (str gn i)) "gaussian" [(symbol (str "m-" gn)) noise]))))
    :n-latents (fn [{:keys [g]}] g)
    :desc (fn [{:keys [g groups pp]}]
            (str "You have observations in " g " groups (" (str/join ", " groups) "). Each group "
                 "has " pp " measurements g0.." (dec pp) " (e.g. " (first groups) "0 is group "
                 (first groups) " measurement 0). The groups differ from one another. "
                 "Model the group structure."))}

   ;; --- :ar1 (c2) — a smoothly-evolving latent chain (Kalman) ---
   {:family :ar1 :complexity 2
    :sample (fn [seed]
              (let [us (uniforms seed 3)
                    n  (pick-int (nth us 0) 4 6)
                    coef (round1 (+ 0.6 (* (nth us 1) 0.35)))    ; 0.6..0.95
                    proc (round1 (+ 0.3 (* (nth us 2) 0.5)))]    ; 0.3..0.8
                {:n n :coef coef :proc proc :obs-noise 0.7
                 :addrs (mapv #(keyword (str "y" %)) (range n))}))
    :data (fn [{:keys [n coef proc obs-noise addrs]} seed]
            (let [zp (normals (mix-seed seed 991) n)
                  zo (normals (mix-seed seed 992) n)
                  ;; z0 ~ N(0,1) matches the gold model's z0 prior, so gold is the TRUE
                  ;; data-generating model; later z_t = coef*z_{t-1} + proc*noise.
                  zs (reductions (fn [z t] (+ (* coef z) (* proc (nth zp t)))) (nth zp 0)
                                 (range 1 n))]
              (zipmap addrs (map (fn [z zi] (round1 (+ z (* obs-noise zi)))) zs zo))))
    :gold (fn [{:keys [n coef obs-noise addrs]} noise]   ; `noise` tunes the PROCESS scale
            (syn/spec (cons (syn/latent 'z0 "gaussian" [0 1])
                            (for [t (range 1 n)]
                              (syn/latent (symbol (str "z" t)) "gaussian"
                                          [(mul (sc coef) (symbol (str "z" (dec t)))) noise])))
                      (map-indexed (fn [t k] (syn/obs k "gaussian" [(symbol (str "z" t)) obs-noise])) addrs)))
    :n-latents (fn [{:keys [n]}] n)
    :desc (fn [{:keys [n]}]
            (str "You have a time series of " n " observations y0.." (dec n) " recorded at "
                 "successive times t = 0,1,.." (dec n) ". The underlying signal evolves "
                 "smoothly over time. Model the series."))}

   ;; --- :segmented (c2, HELD-OUT composition) — two independent regimes ---
   {:family :segmented :complexity 2
    :sample (fn [seed]
              (let [us (uniforms seed 4)
                    nf (pick-int (nth us 0) 3 4)
                    nl (pick-int (nth us 1) 3 4)
                    level (round1 (* 4.0 (- (nth us 2) 0.5)))
                    slope (round1 (* (if (< (nth us 3) 0.5) -1 1)
                                     (+ 0.7 (* (nth (uniforms (mix-seed seed 6) 1) 0) 1.5))))
                    base  (round1 (* 4.0 (- (nth (uniforms (mix-seed seed 7) 1) 0) 0.5)))]
                {:nf nf :nl nl :level level :slope slope :base base
                 :sigma (sample-sigma (nth (uniforms (mix-seed seed 5) 1) 0))
                 :addrs (mapv #(keyword (str "r" %)) (range (+ nf nl)))}))
    :data (fn [{:keys [nf nl level slope base sigma addrs]} seed]
            (let [z (normals (mix-seed seed 991) (+ nf nl))]
              (zipmap addrs
                      (map-indexed (fn [t k]
                                     (let [m (if (< t nf) level (+ (* slope (- t nf)) base))]
                                       (round1 (+ m (* sigma (nth z t)))))) addrs))))
    :gold (fn [{:keys [nf nl addrs]} noise]
            (syn/spec [(syn/latent 'level "gaussian" [0 10])
                       (syn/latent 'slope "gaussian" [0 3])
                       (syn/latent 'base "gaussian" [0 5])]
                      (concat (for [t (range nf)] (syn/obs (nth addrs t) "gaussian" ['level noise]))
                              (for [t (range nf (+ nf nl))]
                                (syn/obs (nth addrs t) "gaussian"
                                         [(add (mul 'slope (sc (- t nf))) 'base) noise])))))
    :n-latents (fn [_] 3)
    ;; held-out family: the desc is deliberately bare index-semantics (no regime hint), so
    ;; the structure must be discovered from the data + oracle, not read off the prompt.
    :desc (fn [{:keys [nf nl]}]
            (str "You have a sequence of " (+ nf nl) " measurements r0.." (dec (+ nf nl))
                 " recorded one after another (index 0.." (dec (+ nf nl)) "). Model the sequence."))}

   ;; --- :varying-slopes (c3) — an independent line per group (loop-wins) ---
   {:family :varying-slopes :complexity 3
    :sample (fn [seed]
              (let [us (uniforms seed 2)
                    g  (pick-int (nth us 0) 2 3)
                    pp 4
                    gs (subvec group-names 0 g)
                    ;; per-group slope (|.|>=0.6) + intercept, drawn from disjoint sub-streams
                    sl (vec (for [k (range g)]
                              (let [u (nth (uniforms (mix-seed seed 10 k) 2) 0)
                                    s (nth (uniforms (mix-seed seed 10 k) 2) 1)]
                                (round1 (* (if (< s 0.5) -1 1) (+ 0.6 (* u 1.8)))))))
                    it (vec (for [k (range g)]
                              (round1 (* 4.0 (- (nth (uniforms (mix-seed seed 11 k) 1) 0) 0.5)))))]
                {:g g :pp pp :groups gs :slopes sl :intercepts it
                 :sigma (sample-sigma (nth (uniforms (mix-seed seed 5) 1) 0))
                 :addrs (vec (for [gn gs i (range pp)] (keyword (str gn i))))}))
    :data (fn [{:keys [g pp groups slopes intercepts sigma addrs]} seed]
            (let [z (normals (mix-seed seed 991) (* g pp))]
              (zipmap addrs
                      (map-indexed (fn [idx k]
                                     (let [gi (quot idx pp) x (rem idx pp)]
                                       (round1 (+ (* (nth slopes gi) x) (nth intercepts gi)
                                                  (* sigma (nth z idx)))))) addrs))))
    :gold (fn [{:keys [groups pp]} noise]
            (syn/spec (mapcat (fn [gn] [(syn/latent (symbol (str "s-" gn)) "gaussian" [0 3])
                                        (syn/latent (symbol (str "i-" gn)) "gaussian" [0 5])]) groups)
                      (for [gn groups i (range pp)]
                        (syn/obs (keyword (str gn i)) "gaussian"
                                 [(add (mul (symbol (str "s-" gn)) (sc i)) (symbol (str "i-" gn))) noise]))))
    :n-latents (fn [{:keys [g]}] (* 2 g))
    :desc (fn [{:keys [g groups pp]}]
            (str "You have observations in " g " groups (" (str/join ", " groups) "). Each group has "
                 pp " measurements g0.." (dec pp) " taken at inputs x = 0,1,.." (dec pp)
                 " (so " (first groups) "0 is group " (first groups) " at x=0). Within each group "
                 "the response changes with the input x. Model how the response depends on x in "
                 "each of the groups."))}])

(def family-index
  "Family keyword → its index (used in seed mixing + a stable split key)."
  (into {} (map-indexed (fn [i d] [(:family d) i]) family-defs)))

(def family-by-key
  (into {} (map (juxt :family identity)) family-defs))

;; ===========================================================================
;; 3. Calibration — crude / gold / solve-bar via the exact oracle.
;; ===========================================================================

(defn- score
  "Score a spec's rendered code against `observations`; {:ev <log-evidence> :method}."
  [spec observations]
  (let [fb (syn/check (syn/render spec) observations {:n-particles 0})]
    {:ev (when (syn/scored? fb) (:evidence fb)) :method (:method fb) :code (:code fb)}))

(defn- crude-spec-at
  "The crude shared-mean covering model at an arbitrary σ (for the crude-tuned sweep)."
  [observations sigma]
  (syn/spec [(syn/latent 'mu "gaussian" [0 10])]
            (for [k (keys observations)] (syn/obs k "gaussian" ['mu sigma]))))

(defn- best-over-grid
  "Score `build` at every grid σ (+ true σ); the highest finite-evidence result, with
   its :sigma and rendered :code. nil if none score."
  [build observations true-sigma]
  (->> (distinct (conj sigma-grid true-sigma))
       (map (fn [g] (assoc (score (build g) observations) :sigma g)))
       (filter (comp some? :ev))
       (sort-by (comp - :ev))
       first))

(defn calibrate
  "Run the exact oracle to grade one instance from the family's `build` (σ → spec), the
   data's `true-sigma`, and whether the family is `structural?`. Returns
     {:crude :crude-tuned :gold :gold-scale :ground-truth-code :method :exact?
      :solve-bar :gap :struct-gap}
   or nil if it could not be cleanly graded.

   :crude       evidence of the untuned shared-mean covering model (σ=3)
   :crude-tuned best shared-mean over the σ-grid (best STRUCTURELESS model)
   :gold        best correct-structure model over the σ-grid (the data-warranted optimum)
   :gold-scale  the winning σ-grid value (the OBSERVATION scale for most families; the
                PROCESS-noise scale for :ar1, whose build tunes the latent chain noise)
   :exact?      gold scored EXACT/Kalman AND reproduces on a re-score (the guard that
                rejects the broken-eliminator families, genmlx-iraj)
   :solve-bar   for STRUCTURAL families (crude-tuned+gold)/2 — provably ABOVE the best
                structureless model, so clearing it REQUIRES structure (not just noise
                tuning). For non-structural :shared-mean (crude+gold)/2 — there is no
                structure to find, only the noise scale to tune from the σ=3 default.
   :gap         gold−crude (headline)        :struct-gap gold−crude-tuned (pure structure)"
  [true-sigma build observations structural?]
  (let [crude-r (score (crude-spec observations) observations)
        ct-r    (best-over-grid #(crude-spec-at observations %) observations true-sigma)
        gold-r  (best-over-grid build observations true-sigma)]
    (when (and (:ev crude-r) (:ev gold-r) (:ev ct-r))
      (let [;; reproducibility guard: re-score the winning gold code; require agreement.
            re      (:ev (score (build (:sigma gold-r)) observations))
            method  (:method gold-r)
            exact?  (boolean (and re (contains? #{:exact :kalman} method)
                                  (< (js/Math.abs (- re (:ev gold-r))) 1e-3)))
            crude   (:ev crude-r) ct (:ev ct-r) g (:ev gold-r)
            ;; the bar sits halfway from the best STRUCTURELESS model (crude-tuned) to
            ;; gold for structural families, so a structureless model can never clear it.
            bar-base (if structural? ct crude)]
        {:crude crude :crude-tuned ct :gold g :gold-scale (:sigma gold-r)
         :ground-truth-code (:code gold-r) :method method :exact? exact?
         :solve-bar (/ (+ bar-base g) 2.0) :gap (- g crude) :struct-gap (- g ct)}))))

;; ===========================================================================
;; 4. Instance generation — sample → data → calibrate → well-posed record.
;; ===========================================================================

(def ^:const default-min-gap
  "Minimum gold−crude (nats) for a task to be well-posed — comfortably above float32
   evidence noise so the bar strictly brackets."
  3.0)

(def ^:const default-min-struct-gap
  "Minimum gold−crude-tuned (nats): the structure must beat even the BEST structureless
   model, so 'solved' means found-the-structure, not noise-tuning."
  1.0)

(defn- well-posed?
  "A task is well-posed iff the oracle graded it exactly+reproducibly, the headline gap
   clears min-gap, the bar strictly brackets (crude < bar < gold), AND (for STRUCTURAL
   families only) the structure beats the best structureless model by min-struct-gap AND
   the best structureless model does NOT clear the bar (crude-tuned < bar — guaranteed by
   the bar definition, asserted here as a guard). Non-structural :shared-mean is exempt
   from both structure gates (its struct-gap is ~0 by design)."
  [{:keys [crude crude-tuned solve-bar gold exact? gap struct-gap]} structural? min-gap min-struct-gap]
  (boolean (and exact?
                (>= gap min-gap)
                (or (not structural?)
                    (and (>= struct-gap min-struct-gap) (< crude-tuned solve-bar)))
                (< crude solve-bar) (< solve-bar gold))))

(defn gen-instance
  "Generate one well-posed task for `family` at (round, instance), or nil if it could
   not be made well-posed within `:max-retries` fresh param draws (counted by the
   caller as a drop). The samplers are signal-biased (|slope|≥0.6, separated groups,
   σ≤1.5), so the first attempt usually clears the gap; each retry re-samples from a
   fresh derived seed as a rare safety net."
  [family round instance {:keys [min-gap min-struct-gap max-retries]
                          :or {min-gap default-min-gap min-struct-gap default-min-struct-gap
                               max-retries 6}}]
  (let [{:keys [sample data gold complexity structural?] :as fdef
         :or {structural? true}} (family-by-key family)
        fidx (family-index family)]
    (loop [attempt 0]
      (if (> attempt max-retries)
        nil
        (let [seed   (mix-seed round fidx instance attempt)
              params (sample seed)
              obs    (data params seed)
              cal    (calibrate (or (:sigma params) (:proc params) 1.0) #(gold params %) obs structural?)]
          (if (and cal (well-posed? cal structural? min-gap min-struct-gap))
            (merge {:id (str (name family) "-r" round "-i" instance)
                    :family family :complexity complexity
                    :n-latents ((:n-latents fdef) params)
                    :task-desc ((:desc fdef) params)
                    :observations obs
                    ;; full ordered params (incl :addrs/:xs) so family-proposer can
                    ;; rebuild the exact gold structure with no hash-map key-order risk
                    :true-params params
                    :seed seed}
                   (select-keys cal [:ground-truth-code :crude :crude-tuned :gold
                                     :gold-scale :solve-bar :gap :struct-gap :exact? :method]))
            (recur (inc attempt))))))))

;; ===========================================================================
;; 5. Curriculum assembly + leakage-safe task-level split.
;;
;; TWO eval cohorts, reported separately (the rrps lesson — don't conflate claims):
;;   :within  — every stride-th instance of a TRAIN family (same-distribution
;;              generalization; each eval task has same-family train siblings)
;;   :family  — an entire HELD-OUT family (:segmented by default): a stronger
;;              COMPOSITIONAL/OOD test (its sub-structures flat+line appear in train,
;;              so its difficulty is bounded by that overlap — named, not hidden).
;; :eval-task-ids is the post-(name) STRING set repl-corpus/sft match on, so the split
;; is not a silent no-op when ids round-trip through (name id).
;; ===========================================================================

(defn assign-split
  "Tag each task :split :train|:eval and :cohort :within|:family|nil. A task is eval iff
   its family is held out OR it is a stride-th instance within its (train) family."
  [tasks {:keys [eval-families eval-stride] :or {eval-families #{:segmented} eval-stride 4}}]
  (->> tasks
       (group-by :family)
       (mapcat (fn [[fam fam-tasks]]
                 (if (contains? eval-families fam)
                   (map #(assoc % :split :eval :cohort :family) fam-tasks)
                   (map-indexed (fn [i t]
                                  (if (zero? (mod (inc i) eval-stride))
                                    (assoc t :split :eval :cohort :within)
                                    (assoc t :split :train :cohort nil)))
                                fam-tasks))))
       vec))

(defn generate-curriculum
  "Build a graded, leakage-safe curriculum.

   opts:
     :round              ReST-EM round (seed base + part of every :id); default 0
     :instances-per-family  instances attempted per family per round; default 20
     :families           families to include; default all in family-defs
     :eval-families      whole held-out families (compositional cohort); default #{:segmented}
     :eval-stride        within-family holdout stride; default 4
     :min-gap :min-struct-gap :max-retries  passed to gen-instance

   Returns {:tasks :train-tasks :eval-tasks :eval-task-ids :by-family :by-complexity
            :summary}. :eval-task-ids is the (comp name :id) STRING set repl-corpus
   consumes. Growth: bump :round (and/or :instances-per-family) for a fresh,
   reproducible, larger batch."
  ([] (generate-curriculum {}))
  ([{:keys [round instances-per-family families eval-families eval-stride]
     :or   {round 0 instances-per-family 20 eval-families #{:segmented} eval-stride 4}
     :as   opts}]
   (let [fams    (or families (mapv :family family-defs))
         per-fam (into {}
                       (for [fam fams]
                         [fam (let [gen (keep #(gen-instance fam round % opts) (range instances-per-family))]
                                {:tasks (vec gen)
                                 :drops (- instances-per-family (count gen))})]))
         tasks   (assign-split (vec (mapcat (comp :tasks val) per-fam))
                               {:eval-families eval-families :eval-stride eval-stride})
         train   (filterv #(= :train (:split %)) tasks)
         evals   (filterv #(= :eval (:split %)) tasks)
         eval-ids (into #{} (map (comp name :id)) evals)
         by-complexity (into (sorted-map)
                             (for [[c ts] (group-by :complexity tasks)]
                               [c {:n (count ts)
                                   :mean-gap (when (seq ts) (/ (reduce + (map :gap ts)) (count ts)))
                                   :mean-struct-gap (when (seq ts) (/ (reduce + (map :struct-gap ts)) (count ts)))
                                   :mean-n-latents (when (seq ts) (/ (reduce + (map :n-latents ts)) (count ts)))}]))]
     {:tasks tasks :train-tasks train :eval-tasks evals :eval-task-ids eval-ids
      :by-family (into (sorted-map)
                       (for [[fam {:keys [tasks drops]}] per-fam]
                         [fam {:n (count tasks) :drops drops
                               :complexity (:complexity (family-by-key fam))}]))
      :by-complexity by-complexity
      :summary {:round round
                :n-tasks (count tasks) :n-train (count train) :n-eval (count evals)
                :eval-within (count (filter #(= :within (:cohort %)) evals))
                :eval-family (count (filter #(= :family (:cohort %)) evals))
                :held-out-families (vec eval-families)
                :n-families (count fams)
                :total-drops (reduce + (map (comp :drops val) per-fam))}})))

;; ===========================================================================
;; 6. Consumer adapters.
;; ===========================================================================

(defn ->probe-task
  "Project a canonical task into the shape scripts/synth_llm_probe.cljs consumes
   ({:id :obs :task-desc :np :solve-bar :crude :gold}). The SOLE place :observations
   is aliased to :obs — the canonical record keeps one source of truth."
  [{:keys [id observations task-desc solve-bar crude gold]}]
  {:id id :obs observations :task-desc task-desc :np 0
   :solve-bar solve-bar :crude crude :gold gold})

(defn family-proposer
  "A no-LLM structured move-set for a task, so an end-to-end harvest can run WITHOUT the
   policy LLM. It is the COLD-START / PLUMBING-VALIDATION proposer (the real proposer is
   the Phase-3 learned LLM): it offers the family's structural edit (crude → the correct
   structure at a neutral σ) and, once structured, the shared-σ grid — yielding a
   MULTI-step trajectory so repl-corpus harvests >1 transition. Its rows are a pipeline
   smoke test, NOT a training corpus (a one-leap crude→gold proposer teaches nothing the
   north-star §12 cares about).

   `task` is a canonical curriculum record; it carries the FULL ordered :true-params, so
   the gold structure is rebuilt exactly (no hash-map key-order risk). Returns
   (fn [spec feedback] -> [candidates]). For :shared-mean (gold == crude shape) there is
   no structural step, so it goes straight to noise refinement. The refinement branch
   tunes the OBSERVATION scale; for :ar1 (whose tunable is the latent PROCESS noise) the
   structural leap alone clears the bar and obs-refinement is a near-no-op — reaching the
   exact gold-scale is the real (Phase-3 LLM) proposer's job, not this validation stub's."
  [{:keys [family true-params]}]
  (let [{:keys [gold]} (family-by-key family)
        gold-spec     (gold true-params 1.0)
        gold-n-latent (count (:latents gold-spec))]
    (fn [spec _feedback]
      (if (< (count (:latents spec)) gold-n-latent)
        ;; structural step: adopt the correct structure at a neutral σ; the oracle keeps
        ;; it iff it improves evidence.
        [{:edit :add-structure :desc (str "adopt " (name family) " structure")
          :spec' gold-spec}]
        ;; refinement: tune the shared observation scale over the grid.
        (for [g sigma-grid :when (not= g (last (:args (first (:obs spec)))))]
          {:edit :set-noise :desc (str "shared obs scale -> " g)
           :spec' (reduce #(syn/set-noise %1 %2 g) spec (map :addr (:obs spec)))})))))
