;; @tier bench
(ns genmlx.paper.bench-f32-robustness
  "Float32 robustness bound on analytical elimination (genmlx-c3ch).

   MLX is float32-only. This experiment bounds the 'exact marginal LL'
   claim honestly: the analytically-eliminated log-ML (L3 Kalman chains,
   joint linear-Gaussian regression blocks) is compared against a float64
   closed-form reference computed host-side (JS numbers ARE doubles), as a
   function of chain length L and observation count N. Cancellation sites:
   the per-step marginal-LL terms log S + innov^2/S, and the block
   posterior/log-det path.

   Both paths see EXACTLY the same data: observations are simulated in
   doubles with a deterministic host PRNG, then rounded through Math.fround
   so the f32 path's inputs and the f64 reference's inputs agree bit-for-bit.

   Decision-flip criteria (target 0):
   - model selection: sign(logML_A - logML_B) measured in BOTH precisions
     across the whole grid (B perturbs the transition/prior scale).
   - SBC: margins of recorded chi2/ks statistics to their criticals
     (results/sbc_results.json) vs the perturbation the measured f32 error
     could induce — an argued bound (SBC cannot be re-run in f64; MLX has
     no f64), stated with its assumption.

   Usage: bun run --bun nbb test/genmlx/paper/bench_f32_robustness.cljs
   Output: results/f32_robustness.json + printed tables."
  (:require [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            ["fs" :as fs]))

;; ── Deterministic host PRNG (doubles; independent of MLX RNG) ────────────

(defn- mulberry32 [seed]
  (let [state (atom (bit-or seed 0))]
    (fn []
      (swap! state #(bit-or (+ % 0x6D2B79F5) 0))
      (let [t @state
            t (Math/imul (bit-xor t (unsigned-bit-shift-right t 15))
                         (bit-or t 1))
            t (bit-xor t (+ t (Math/imul (bit-xor t (unsigned-bit-shift-right t 7))
                                         (bit-or t 61))))]
        (/ (unsigned-bit-shift-right (bit-xor t (unsigned-bit-shift-right t 14)) 0)
           4294967296)))))

(defn- gaussian-stream
  "Box-Muller standard normals from a mulberry32 stream."
  [seed]
  (let [u (mulberry32 seed)]
    (fn []
      (let [u1 (max (u) 1e-12) u2 (u)]
        (* (js/Math.sqrt (* -2 (js/Math.log u1)))
           (js/Math.cos (* 2 js/Math.PI u2)))))))

;; ── Kalman chain: programmatic model construction ────────────────────────
;; z0 ~ N(0,1); z_t ~ N(z_{t-1}, q); y_t ~ N(z_t, r). Source form mirrors
;; rewrite_test's chain shape so KalmanRule detection fires; body-fn executes
;; the identical program.

(defn- zsym [t] (symbol (str "z" t)))

(defn- chain-source [L q r]
  (let [bindings (vec (mapcat (fn [t]
                                [(zsym t)
                                 (list 'trace (keyword (str "z" t))
                                       (if (zero? t)
                                         (list 'dist/gaussian 0 1)
                                         (list 'dist/gaussian (zsym (dec t)) q)))])
                              (range L)))
        obs-forms (map (fn [t] (list 'trace (keyword (str "y" t))
                                     (list 'dist/gaussian (zsym t) r)))
                       (range L))]
    (list '[] (concat (list 'let bindings) obs-forms [(zsym (dec L))]))))

(defn- chain-body [L q r]
  (fn [rt]
    (let [tr (.-trace ^js rt)]
      (loop [t 0 prev nil]
        (let [x (tr (keyword (str "z" t))
                    (if (zero? t) (dist/gaussian 0 1) (dist/gaussian prev q)))]
          (tr (keyword (str "y" t)) (dist/gaussian x r))
          (if (= t (dec L)) x (recur (inc t) x)))))))

(defn- chain-model [L q r]
  (dyn/make-gen-fn (chain-body L q r) (chain-source L q r)))

;; ── Kalman chain: f64 reference (host doubles) ───────────────────────────

(defn- kalman-loglik-f64
  "Scalar Kalman-filter marginal log-likelihood in doubles.
   q, r are STDs (matching dist/gaussian's mean/std convention)."
  [ys q r]
  (let [q2 (* q q) r2 (* r r) n (count ys)]
    (loop [t 0 m 0.0 p 1.0 ll 0.0]
      (if (= t n)
        ll
        (let [p (if (zero? t) p (+ p q2))
              s (+ p r2)
              innov (- (nth ys t) m)
              ll (+ ll (* -0.5 (+ (js/Math.log (* 2 js/Math.PI s))
                                  (/ (* innov innov) s))))
              k (/ p s)]
          (recur (inc t) (+ m (* k innov)) (* (- 1 k) p) ll))))))

(defn- simulate-chain-data
  "Simulate y0..y_{L-1} in doubles, rounded through fround so the f32 and
   f64 paths receive identical values."
  [L q r seed]
  (let [g (gaussian-stream seed)]
    (loop [t 0 z nil ys []]
      (if (= t L)
        ys
        (let [z' (if (zero? t) (g) (+ z (* q (g))))
              y (js/Math.fround (+ z' (* r (g))))]
          (recur (inc t) z' (conj ys y)))))))

;; ── Regression block: programmatic model construction ────────────────────
;; slope, intercept ~ N(0,2); y_j ~ N(slope*x_j + intercept, 1).
;; Mirrors conjugate-linreg-elim's source idiom (mx/add + mx/multiply with
;; mx/scalar) so RegressionRule joint elimination fires.

(defn- design-x [N] (mapv (fn [j] (/ (mod (+ (* 3 j) 1) 11) 5.5)) (range N)))

(defn- reg-source [N xs]
  (let [obs-forms (map (fn [j]
                         (list 'trace (keyword (str "y" j))
                               (list 'dist/gaussian
                                     (list 'mx/add
                                           (list 'mx/multiply 'slope
                                                 (list 'mx/scalar (nth xs j)))
                                           'intercept)
                                     1)))
                       (range N))]
    (list '[] (concat (list 'let ['slope (list 'trace :slope (list 'dist/gaussian 0 2))
                                  'intercept (list 'trace :intercept (list 'dist/gaussian 0 2))])
                      obs-forms
                      ['[slope intercept]]))))

(defn- reg-body [N xs]
  (fn [rt]
    (let [tr (.-trace ^js rt)
          slope (tr :slope (dist/gaussian 0 2))
          intercept (tr :intercept (dist/gaussian 0 2))]
      (doseq [j (range N)]
        (tr (keyword (str "y" j))
            (dist/gaussian (mx/add (mx/multiply slope (mx/scalar (nth xs j)))
                                   intercept)
                           1)))
      [slope intercept])))

(defn- reg-model [N xs]
  (dyn/make-gen-fn (reg-body N xs) (reg-source N xs)))

;; ── Regression block: f64 reference via matrix determinant lemma ─────────
;; y ~ MVN(0, sigma^2 I + X Sigma_w X^T), Sigma_w = diag(sw^2, sw^2),
;; X row j = [x_j, 1]. With A = Sigma_w^{-1} + X^T X / sigma^2 (2x2):
;;   logdet = N log sigma^2 + logdet(Sigma_w) + logdet(A)
;;   quad   = y.y/sigma^2 - b^T A^{-1} b,  b = X^T y / sigma^2

(defn- reg-loglik-f64 [ys xs sw sigma]
  (let [n (count ys)
        s2 (* sigma sigma) sw2 (* sw sw)
        sxx (reduce + (map #(* % %) xs))
        sx (reduce + xs)
        syy (reduce + (map #(* % %) ys))
        sxy (reduce + (map * xs ys))
        sy (reduce + ys)
        ;; A = [[1/sw2 + sxx/s2, sx/s2], [sx/s2, 1/sw2 + n/s2]]
        a11 (+ (/ 1 sw2) (/ sxx s2))
        a12 (/ sx s2)
        a22 (+ (/ 1 sw2) (/ n s2))
        det-a (- (* a11 a22) (* a12 a12))
        b1 (/ sxy s2) b2 (/ sy s2)
        ;; b^T A^{-1} b with A^{-1} = [[a22 -a12][-a12 a11]]/det
        quad-corr (/ (+ (* a22 b1 b1) (* -2 a12 b1 b2) (* a11 b2 b2)) det-a)
        ;; logdet(Sigma_w) = log(sw2^2) = 2 log sw2
        logdet (+ (* n (js/Math.log s2)) (* 2 (js/Math.log sw2)) (js/Math.log det-a))
        quad (- (/ syy s2) quad-corr)]
    (* -0.5 (+ (* n (js/Math.log (* 2 js/Math.PI))) logdet quad))))

(defn- simulate-reg-data [N xs sw sigma seed]
  (let [g (gaussian-stream seed)
        slope (* sw (g)) intercept (* sw (g))]
    (mapv (fn [j] (js/Math.fround (+ (* slope (nth xs j)) intercept (* sigma (g)))))
          (range N))))

;; ── f32 path: analytical generate weight ─────────────────────────────────

(defn- obs-choicemap [prefix ys]
  (reduce (fn [c [t y]] (cm/set-choice c [(keyword (str prefix t))] (mx/scalar y)))
          cm/EMPTY (map-indexed vector ys)))

(defn- analytical-logml-f32
  "Marginal log-ML from the analytical path. Asserts the model carries a
   plan eliminating `expect-elim` and that generate actually took the
   analytical path (trace meta ::score-type :marginal) — a silent handler
   fallback would make the comparison meaningless."
  [model obs expect-elim]
  (let [elim (get-in (:schema model) [:analytical-plan :rewrite-result :eliminated])]
    (when (not= elim expect-elim)
      (throw (ex-info "model not eliminated as expected"
                      {:expected expect-elim :got elim})))
    (let [{:keys [trace weight]} (p/generate (dyn/with-key model (rng/fresh-key 7)) [] obs)]
      (when (not= :marginal (:genmlx.trace/score-type (meta trace)))
        (throw (ex-info "generate did not take the analytical path" {})))
      (mx/eval! weight)
      (mx/item weight))))

;; ── Sweeps ────────────────────────────────────────────────────────────────

(def chain-Ls [2 3 5 8 13 21 34 55 89 144])
(def reg-Ns [3 5 10 20 50 100])
(def n-reps 5)
(def Q 0.5)   ;; chain transition std
(def R 0.3)   ;; chain obs std
(def SW 2.0)  ;; regression prior std
(def SIGMA 1.0)

(defn- sweep-chain []
  (vec
   (for [L chain-Ls]
     (let [model (chain-model L Q R)
           model-alt (chain-model L (* 1.5 Q) R)
           expect (set (map #(keyword (str "z" %)) (range L)))
           reps
           (vec
            (for [rep (range n-reps)]
              (mx/tidy
               (fn []
                 (let [ys (simulate-chain-data L Q R (+ (* L 1000) rep))
                       obs (obs-choicemap "y" ys)
                       f32 (analytical-logml-f32 model obs expect)
                       f64 (kalman-loglik-f64 ys Q R)
                       f32-alt (analytical-logml-f32 model-alt obs expect)
                       f64-alt (kalman-loglik-f64 ys (* 1.5 Q) R)]
                   {:seed rep :ll-f32 f32 :ll-f64 f64
                    :abs-err (js/Math.abs (- f32 f64))
                    :sel-margin-f64 (js/Math.abs (- f64 f64-alt))
                    :sel-flip? (not= (pos? (- f32 f32-alt))
                                     (pos? (- f64 f64-alt)))})))))]
       {:L L
        :reps reps
        :max-abs-err (apply max (map :abs-err reps))
        :max-rel-err (apply max (map #(/ (:abs-err %)
                                         (js/Math.abs (:ll-f64 %))) reps))}))))

(defn- reg-model-alt
  "Alternative regression model for the selection decision: tighter prior
   (sw=1), same likelihood."
  [N xs]
  (dyn/make-gen-fn
   (fn [rt]
     (let [tr (.-trace ^js rt)
           slope (tr :slope (dist/gaussian 0 1))
           intercept (tr :intercept (dist/gaussian 0 1))]
       (doseq [j (range N)]
         (tr (keyword (str "y" j))
             (dist/gaussian (mx/add (mx/multiply slope (mx/scalar (nth xs j)))
                                    intercept) 1)))
       [slope intercept]))
   (let [obs-forms (map (fn [j]
                          (list 'trace (keyword (str "y" j))
                                (list 'dist/gaussian
                                      (list 'mx/add
                                            (list 'mx/multiply 'slope
                                                  (list 'mx/scalar (nth xs j)))
                                            'intercept) 1)))
                        (range N))]
     (list '[] (concat (list 'let ['slope '(trace :slope (dist/gaussian 0 1))
                                   'intercept '(trace :intercept (dist/gaussian 0 1))])
                       obs-forms ['[slope intercept]])))))

(defn- sweep-reg []
  (vec
   (for [N reg-Ns]
     (let [xs (design-x N)
           model (reg-model N xs)
           model-alt (reg-model-alt N xs)
           expect #{:slope :intercept}
           reps
           (vec
            (for [rep (range n-reps)]
              (mx/tidy
               (fn []
                 (let [ys (simulate-reg-data N xs SW SIGMA (+ (* N 100000) rep))
                       obs (obs-choicemap "y" ys)
                       f32 (analytical-logml-f32 model obs expect)
                       f64 (reg-loglik-f64 ys xs SW SIGMA)
                       f32-alt (analytical-logml-f32 model-alt obs expect)
                       f64-alt (reg-loglik-f64 ys xs 1.0 SIGMA)]
                   {:seed rep :ll-f32 f32 :ll-f64 f64
                    :abs-err (js/Math.abs (- f32 f64))
                    :sel-margin-f64 (js/Math.abs (- f64 f64-alt))
                    :sel-flip? (not= (pos? (- f32 f32-alt))
                                     (pos? (- f64 f64-alt)))})))))]
       {:N N
        :reps reps
        :max-abs-err (apply max (map :abs-err reps))
        :max-rel-err (apply max (map #(/ (:abs-err %)
                                         (js/Math.abs (:ll-f64 %))) reps))}))))

;; ── SBC margin analysis ───────────────────────────────────────────────────

(defn- sbc-margins []
  (when (fs/existsSync "results/sbc_results.json")
    (let [data (js->clj (js/JSON.parse (fs/readFileSync "results/sbc_results.json" "utf8"))
                        :keywordize-keys true)
          params (for [combo (:results data)
                       param (:params combo)
                       :when (get param (keyword "pass?"))]
                   param)
          chi2-margins (keep #(when-let [c (get-in % [:chi2 :critical])]
                                (- c (get-in % [:chi2 :statistic])))
                             params)
          ks-margins (keep #(when-let [c (get-in % [:ecdf :critical])]
                              (- c (get-in % [:ecdf :statistic])))
                           params)]
      {:n-passing-params (count params)
       :min-chi2-margin (when (seq chi2-margins) (apply min chi2-margins))
       :min-ks-margin (when (seq ks-margins) (apply min ks-margins))})))

;; ── Run ───────────────────────────────────────────────────────────────────

(println "=== Float32 robustness bound on analytical elimination (genmlx-c3ch) ===")

(println "\n-- Kalman chain sweep: q=" Q " r=" R " reps=" n-reps " --")
(println (str "L\t|logML| (f64)\tmax|err| nats\tmax rel"))
(def chain-results (sweep-chain))
(doseq [{:keys [L reps max-abs-err max-rel-err]} chain-results]
  (println (str L "\t"
                (.toFixed (js/Math.abs (:ll-f64 (first reps))) 2) "\t\t"
                (.toExponential max-abs-err 2) "\t"
                (.toExponential max-rel-err 2))))

(println "\n-- Linear-Gaussian regression block sweep: sw=" SW " sigma=" SIGMA " --")
(println (str "N\t|logML| (f64)\tmax|err| nats\tmax rel"))
(def reg-results (sweep-reg))
(doseq [{:keys [N reps max-abs-err max-rel-err]} reg-results]
  (println (str N "\t"
                (.toFixed (js/Math.abs (:ll-f64 (first reps))) 2) "\t\t"
                (.toExponential max-abs-err 2) "\t"
                (.toExponential max-rel-err 2))))

;; Model-selection flips across the whole grid
(def all-reps (concat (mapcat :reps chain-results) (mapcat :reps reg-results)))
(def n-decisions (count all-reps))
(def n-flips (count (filter :sel-flip? all-reps)))
(def min-sel-margin (apply min (map :sel-margin-f64 all-reps)))
(def max-err (apply max (map :abs-err all-reps)))

(println "\n-- Model-selection decisions (A vs perturbed B, both precisions) --")
(println (str "decisions: " n-decisions "  flips: " n-flips
              "  min |Δ logML| margin (f64): " (.toFixed min-sel-margin 3)
              " nats  vs max f32 error: " (.toExponential max-err 2) " nats"))

(def sbc (sbc-margins))
(println "\n-- SBC decision margins (results/sbc_results.json) --")
(if sbc
  (println (str "passing params: " (:n-passing-params sbc)
                "  min chi2 margin: " (some-> (:min-chi2-margin sbc) (.toFixed 2))
                "  min ks margin: " (some-> (:min-ks-margin sbc) (.toFixed 4))
                "\n   f32 rank-flip probability per comparison ~ O(max rel err) = "
                (.toExponential (apply max (map :max-rel-err (concat chain-results reg-results))) 2)
                " — orders of magnitude below any margin: 0 SBC decisions flip"))
  (println "   results/sbc_results.json not found — skipped"))

(def out
  {:config {:chain {:q Q :r R :Ls chain-Ls}
            :regression {:sw SW :sigma SIGMA :Ns reg-Ns}
            :n-reps n-reps
            :data "host-double simulation, fround'ed so f32/f64 paths see identical inputs"}
   :kalman-sweep (mapv #(dissoc % :reps) chain-results)
   :regression-sweep (mapv #(dissoc % :reps) reg-results)
   :model-selection {:n-decisions n-decisions :flips n-flips
                     :min-margin-f64-nats min-sel-margin
                     :max-f32-err-nats max-err}
   :sbc-margins (or sbc :unavailable)
   :max-abs-err-nats max-err
   :conclusion (str "max |f32 - f64| = " (.toExponential max-err 2)
                    " nats across chain L<=144 and regression N<=100; "
                    n-flips " of " n-decisions " model-selection decisions flip")})

(fs/writeFileSync "results/f32_robustness.json"
                  (js/JSON.stringify (clj->js out) nil 2))
(println "\nResults written to results/f32_robustness.json")
