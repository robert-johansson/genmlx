;; @tier fast core
(ns genmlx.score-type-test
  "genmlx-lbae: score-type as enforced metadata.

   Every trace-producing path tags its traces with :genmlx.trace/score-type
   (:joint | :marginal | :collapsed), combinators and splice propagate the
   tag from sub-traces (no laundering), and joint-scoring boundaries
   (update/project/regenerate) convert :marginal traces (genmlx-pkmx) or
   THROW on unconvertible ones (:collapsed — empty choices, nothing to
   re-generate from). Trace-MH entry points assert their regenerate results
   are joint-scored, so a regressed/forgotten analytical strip throws
   instead of silently anchoring chains at the posterior mean (genmlx-540f:
   chi2 290.9 vs crit 21.67)."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.gfi :as gfi]
            [genmlx.trace :as tr]
            [genmlx.dist :as dist]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.combinators :as comb]
            [genmlx.serialize :as ser]
            [genmlx.inference.exact :as exact]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.inference.kernel :as kern]
            [genmlx.inference.importance :as imp])
  (:require-macros [genmlx.gen :refer [gen]]))

(def ^:private st-key :genmlx.trace/score-type)

(defn- score-type-error?
  "True iff (f) throws the score-type contract violation."
  [f]
  (try (f) false
       (catch :default e
         (= :score-type-mismatch (:genmlx/error (ex-data e))))))

(defn- conjugate-model
  "mu ~ N(0,1); y ~ N(mu,1). Normal-normal conjugate — static, L3-eliminable."
  []
  (gen []
    (let [mu (trace :mu (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
      (trace :y (dist/gaussian mu (mx/scalar 1)))
      mu)))

(defn- prefix-model
  "Static prefix + dynamic loop suffix — L1-M3 prefix path."
  []
  (gen [n]
    (let [mu (trace :mu (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
      (doseq [i (range n)]
        (trace (keyword (str "y" i)) (dist/gaussian mu (mx/scalar 1))))
      mu)))

(defn- discrete-model
  "coin ~ Bernoulli(0.5); obs ~ Bernoulli(coin ? 0.9 : 0.1) — enumerable."
  []
  (gen []
    (let [coin (trace :coin (dist/bernoulli 0.5))
          pr (mx/where coin (mx/scalar 0.9) (mx/scalar 0.1))]
      (trace :obs (dist/bernoulli pr))
      coin)))

;; Two independent conjugate pairs — same shape as trace_mh_elim_test/sbc
;; two-gaussians; statically eliminated #{:a :b}.
(defn- two-gaussians []
  (gen []
    (let [a (trace :a (dist/gaussian 0 2))
          b (trace :b (dist/gaussian 0 2))]
      (trace :obs-a (dist/gaussian a 1))
      (trace :obs-b (dist/gaussian b 1))
      [a b])))

;; Eliminated conjugate latent :mu (normal-normal via :y) PLUS a residual :tau
;; (Gamma prior, non-conjugate scale obs :w). Trace is :marginal; selecting :mu
;; RE-OPENS the eliminated pair (a wl1y decline → joint conversion), while
;; selecting the residual :tau does NOT re-open it (the analytical regenerate
;; still fires and returns a :marginal result).
(defn- conj+residual []
  (gen []
    (let [mu  (trace :mu (dist/gaussian 0 1))
          tau (trace :tau (dist/gamma-dist 2 1))]
      (trace :y (dist/gaussian mu 1))
      (trace :w (dist/gaussian 0 tau))
      mu)))

(def ^:private tg-obs
  (-> cm/EMPTY
      (cm/set-choice [:obs-a] (mx/scalar 1.0))
      (cm/set-choice [:obs-b] (mx/scalar -1.0))))

(defn- choice-val [trace addr]
  (let [v (cm/get-value (cm/get-submap (:choices trace) addr))]
    (mx/eval! v)
    (mx/item v)))

;; ============================================================
;; 1. trace.cljs helpers
;; ============================================================

(deftest helpers-accessor-and-tagging
  (let [t (tr/make-trace {:gen-fn nil :args [] :choices cm/EMPTY
                          :retval nil :score (mx/scalar 0.0)})]
    (testing "untagged trace defaults to :joint"
      (is (= :joint (tr/score-type t))))
    (testing "with-score-type round-trips"
      (is (= :marginal (tr/score-type (tr/with-score-type t :marginal))))
      (is (= :collapsed (tr/score-type (tr/with-score-type t :collapsed)))))
    (testing "with-score-type preserves other metadata"
      (let [t' (-> t (with-meta {:other 1}) (tr/with-score-type :marginal))]
        (is (= 1 (:other (meta t'))))))))

(deftest helpers-combine-score-types
  (testing "only path-unstable tags propagate through composition"
    (is (= :joint (tr/combine-score-types)))
    (is (= :joint (tr/combine-score-types :joint :joint)))
    (is (= :marginal (tr/combine-score-types :joint :marginal)))
    (is (= :marginal (tr/combine-score-types :marginal :joint)))
    (is (= :marginal (tr/combine-score-types :marginal :collapsed))))
  (testing "a :collapsed PART composes as :joint — enumerate blocks record no
            internal choices and re-derive their exact score consistently
            (the MCMC-around-an-enumerate-splice pattern, exact_test 41)"
    (is (= :joint (tr/combine-score-types :collapsed :joint)))
    (is (= :joint (tr/combine-score-types :joint :collapsed))))
  (testing "nil (untagged) counts as :joint"
    (is (= :joint (tr/combine-score-types nil nil)))
    (is (= :marginal (tr/combine-score-types nil :marginal)))))

(deftest helpers-assert-joint!
  (let [t (tr/make-trace {:gen-fn nil :args [] :choices cm/EMPTY
                          :retval nil :score (mx/scalar 0.0)})]
    (testing "passes through joint and untagged traces"
      (is (identical? t (tr/assert-joint! t :test-op)))
      (is (some? (tr/assert-joint! (tr/with-score-type t :joint) :test-op))))
    (testing "throws descriptive ex-info on non-joint traces"
      (is (score-type-error?
            #(tr/assert-joint! (tr/with-score-type t :marginal) :test-op)))
      (is (score-type-error?
            #(tr/assert-joint! (tr/with-score-type t :collapsed) :test-op)))
      (let [data (try (tr/assert-joint! (tr/with-score-type t :collapsed) :my-op)
                      nil
                      (catch :default e (ex-data e)))]
        (is (= :collapsed (:score-type data)) "ex-data carries the actual tag")
        (is (= :joint (:expected data)) "ex-data carries the expectation")
        (is (= :my-op (:op data)) "ex-data carries the consuming op")))))

;; ============================================================
;; 2. Producer matrix — every path tags EXPLICITLY
;; ============================================================

(deftest producer-handler-path-tags-joint
  (let [model (dyn/auto-key (gfi/strip-compiled (conjugate-model)))
        t (p/simulate model [])]
    (is (= :handler (:label (dyn/resolve-dispatch model :simulate))) "precondition")
    (is (contains? (meta t) st-key) "tag is explicit, not just defaulted")
    (is (= :joint (tr/score-type t)))))

(deftest producer-compiled-path-tags-joint
  (let [model (dyn/auto-key (conjugate-model))
        t (p/simulate model [])]
    (is (= :compiled (:label (dyn/resolve-dispatch model :simulate))) "precondition")
    (is (contains? (meta t) st-key))
    (is (= :joint (tr/score-type t)))
    (testing "compiled update and regenerate tag :joint too"
      (let [u (p/update model t (cm/choicemap :y (mx/scalar 2.0)))
            r (p/regenerate model t (sel/select :mu))]
        (is (= :joint (tr/score-type (:trace u))))
        (is (contains? (meta (:trace u)) st-key))
        (is (= :joint (tr/score-type (:trace r))))
        (is (contains? (meta (:trace r)) st-key))))))

(deftest producer-prefix-path-tags-joint
  (let [model (dyn/auto-key (prefix-model))
        t (p/simulate model [2])]
    (is (= :prefix (:label (dyn/resolve-dispatch model :simulate))) "precondition")
    (is (contains? (meta t) st-key))
    (is (= :joint (tr/score-type t)))))

(deftest producer-analytical-fired-tags-marginal
  (let [model (dyn/with-key (conjugate-model) (rng/fresh-key 7))
        {t :trace} (p/generate model [] (cm/choicemap :y (mx/scalar 1.5)))]
    (is (= :marginal (tr/score-type t)))))

(deftest producer-analytical-declined-tags-joint
  ;; genmlx-b470: fully-constrained prior → every handler falls through →
  ;; plain joint scoring, and the tag must say so EXPLICITLY.
  (let [model (dyn/with-key (conjugate-model) (rng/fresh-key 8))
        {t :trace} (p/generate model [] (cm/choicemap :mu (mx/scalar 0.3)
                                                      :y (mx/scalar 1.5)))]
    (is (contains? (meta t) st-key))
    (is (= :joint (tr/score-type t)))))

(deftest producer-enumerate-tags-collapsed
  (let [em (exact/enumerate (dyn/auto-key (discrete-model)))]
    (testing "generate"
      (let [{t :trace} (p/generate em [] (cm/choicemap :obs (mx/scalar 1.0)))]
        (is (= :collapsed (tr/score-type t)))))
    (testing "simulate"
      (is (= :collapsed (tr/score-type (p/simulate em [])))))))

(deftest producer-combinator-plain-tags-joint
  (let [kernel (dyn/auto-key (gen [x]
                  (trace :y (dist/gaussian (mx/scalar 0) (mx/scalar 1)))))
        mapped (dyn/auto-key (comb/map-combinator kernel))
        t (p/simulate mapped [[1.0 2.0]])]
    (is (contains? (meta t) st-key))
    (is (= :joint (tr/score-type t)))))

(deftest producer-combinator-laundering-tags-marginal
  ;; A Map over a conjugate-but-uncompilable kernel: Map falls back to the
  ;; per-element handler path, each sub-generate fires the analytical path →
  ;; marginal sub-scores summed into the parent total. The parent trace must
  ;; carry the tag — this is the §3.3 laundering hole. Gamma-Poisson is the
  ;; real such cell: a conjugate family whose prior has no noise transform.
  (let [kernel (dyn/auto-key (gen [x]
                  (let [lam (trace :lam (dist/gamma-dist (mx/scalar 2) (mx/scalar 2)))]
                    (trace :y (dist/poisson lam))
                    lam)))
        mapped (dyn/auto-key (comb/map-combinator kernel))
        obs (-> cm/EMPTY
                (cm/set-choice [0 :y] (mx/scalar 3.0))
                (cm/set-choice [1 :y] (mx/scalar 1.0)))
        {t :trace} (p/generate mapped [[1.0 2.0]] obs)]
    (is (seq (:auto-handlers (:schema kernel)))
        "precondition: kernel is analytically eliminable")
    (is (nil? (:compiled-generate (:schema kernel)))
        "precondition: kernel is NOT compilable (gamma has no noise transform), so Map takes the handler fallback")
    (is (= :marginal (tr/score-type t))
        "marginal-ness propagates from sub-traces to the combinator trace")))

(deftest producer-combinator-compiled-elements-stay-joint
  ;; The complementary cell: a static conjugate kernel compiles, Map prefers
  ;; the compiled per-element path which scores JOINTLY — the analytical
  ;; sub-generate never runs, no laundering occurs, and :joint is correct.
  (let [kernel (dyn/auto-key (gen [x]
                  (let [mu (trace :mu (dist/gaussian (mx/scalar 0) (mx/scalar 1)))]
                    (trace :y (dist/gaussian mu (mx/scalar 1)))
                    mu)))
        mapped (dyn/auto-key (comb/map-combinator kernel))
        obs (-> cm/EMPTY
                (cm/set-choice [0 :y] (mx/scalar 1.5))
                (cm/set-choice [1 :y] (mx/scalar -0.5)))
        {t :trace} (p/generate mapped [[1.0 2.0]] obs)]
    (is (seq (:auto-handlers (:schema kernel)))
        "precondition: kernel is analytically eliminable")
    (is (some? (:compiled-generate (:schema kernel)))
        "precondition: kernel IS compilable — Map takes the compiled element path")
    (is (= :joint (tr/score-type t))
        "compiled element scores are joint; no laundering, tag says so")))

(deftest producer-splice-laundering-tags-marginal
  ;; Same hole through splice: execute-sub must propagate the sub-trace tag
  ;; and merge-sub-result must lub it into the parent state.
  (let [sub (conjugate-model)
        parent (dyn/with-key (gen [] (splice :sub sub)) (rng/fresh-key 11))
        obs (cm/set-submap cm/EMPTY :sub (cm/choicemap :y (mx/scalar 1.5)))
        {t :trace} (p/generate parent [] obs)]
    (is (seq (:auto-handlers (:schema sub)))
        "precondition: sub-gf is analytically eliminable")
    (is (= :marginal (tr/score-type t))
        "marginal-ness propagates through splice into the parent trace")
    (testing "splice-scores metadata coexists with the tag (no meta wipe)"
      (is (some? (:genmlx.dynamic/splice-scores (meta t)))
          "parent trace still carries splice scores"))))

(deftest producer-enumerate-splice-composes-joint
  ;; The certified mixed-inference pattern (exact_test 41): a parent model
  ;; splices an enumerate-wrapped discrete block and runs trace-MH on its
  ;; continuous sites. The collapsed block records no internal choices and
  ;; re-derives its exact score consistently in every op, so the PARENT is
  ;; joint-scored — it must not inherit :collapsed (that would make the
  ;; boundary guard reject a mathematically valid flow).
  (let [agent-gf (dyn/auto-key (gen [] (trace :choice (dist/bernoulli 0.6))))
        ;; laplace likelihood: keeps the parent NON-conjugate so the only
        ;; non-joint tag source is the enumerate splice itself
        parent (dyn/with-key
                 (gen []
                   (let [mu (trace :mu (dist/gaussian (mx/scalar 0) (mx/scalar 5)))]
                     (splice :agent (exact/enumerate agent-gf))
                     (trace :obs (dist/laplace mu (mx/scalar 1)))
                     mu))
                 (rng/fresh-key 41))
        {t :trace} (p/generate parent [] (cm/choicemap :obs (mx/scalar 2.0)))]
    (is (empty? (:auto-handlers (:schema parent)))
        "precondition: parent itself is not analytically eliminable")
    (is (= :joint (tr/score-type t))
        "parent with an enumerate splice is joint-scored, not collapsed")
    (testing "trace-MH over the continuous site flows"
      (let [t' ((kern/mh-kernel (sel/select :mu)) t (rng/fresh-key 42))]
        (is (tr/trace? t') "mh-kernel step runs without throwing"))
      (let [{rt :trace} (p/regenerate parent t (sel/select :mu))]
        (is (= :joint (tr/score-type rt)) "regenerate result stays joint")))))

(deftest producer-vectorized-tags-joint
  (let [model (dyn/auto-key (conjugate-model))
        vt (dyn/vsimulate model [] 4 (rng/fresh-key 3))]
    (is (contains? (meta vt) st-key))
    (is (= :joint (tr/score-type vt)))))

;; ============================================================
;; 3. Conversion at boundaries — laundered parents convert exactly
;; ============================================================

(deftest laundered-parent-converts-at-joint-boundaries
  ;; Oracle: the same choices re-generated on the fully stripped parent (a
  ;; joint trace, handler ground truth) pushed through the same op.
  (let [sub (conjugate-model)
        parent (dyn/with-key (gen [] (splice :sub sub)) (rng/fresh-key 11))
        obs (cm/set-submap cm/EMPTY :sub (cm/choicemap :y (mx/scalar 1.5)))
        {lt :trace} (p/generate parent [] obs)
        stripped (dyn/auto-key (gfi/strip-compiled parent))
        {jt :trace} (p/generate stripped [] (:choices lt))
        new-obs (cm/set-submap cm/EMPTY :sub (cm/choicemap :y (mx/scalar 2.0)))]
    (is (= :marginal (tr/score-type lt)) "precondition: parent is laundered-marginal")
    (is (= :joint (tr/score-type jt)) "oracle trace is joint")
    (testing "update converts and matches the joint oracle"
      (let [u-l (p/update parent lt new-obs)
            u-j (p/update stripped jt new-obs)
            _ (mx/materialize! (:weight u-l) (:weight u-j))]
        (is (< (js/Math.abs (- (mx/item (:weight u-l)) (mx/item (:weight u-j)))) 1e-4)
            "update weights match")
        (is (= :joint (tr/score-type (:trace u-l))) "converted result is joint")
        (is (contains? (meta (:trace u-l)) st-key) "and explicitly tagged")))
    (testing "project converts and matches the joint oracle"
      (let [p-l (p/project parent lt (sel/select :sub))
            p-j (p/project stripped jt (sel/select :sub))
            _ (mx/materialize! p-l p-j)]
        (is (< (js/Math.abs (- (mx/item p-l) (mx/item p-j))) 1e-4)
            "project values match")))))

;; ============================================================
;; 4. Boundary checks — unconvertible tags THROW
;; ============================================================

(deftest collapsed-traces-throw-at-joint-boundaries
  (let [model (dyn/auto-key (gfi/strip-compiled (conjugate-model)))
        {t :trace} (p/generate model [] (cm/choicemap :y (mx/scalar 1.5)))
        ct (tr/with-score-type t :collapsed)]
    (is (score-type-error? #(p/update model ct (cm/choicemap :y (mx/scalar 2.0))))
        "update on a :collapsed trace throws")
    (is (score-type-error? #(p/regenerate model ct (sel/select :mu)))
        "regenerate on a :collapsed trace throws")
    (is (score-type-error? #(p/project model ct (sel/select :mu)))
        "project on a :collapsed trace throws")
    (testing "unknown tags throw too (closed vocabulary at the boundary)"
      (is (score-type-error?
            #(p/update model (tr/with-score-type t :bogus)
                       (cm/choicemap :y (mx/scalar 2.0))))))))

(deftest enumerate-trace-into-trace-mh-throws
  ;; The bean's canonical example: trace-MH receiving a :collapsed trace.
  ;; Pre-lbae this silently no-ops (regenerate weight 0 → always accept).
  (let [em (exact/enumerate (dyn/auto-key (discrete-model)))
        {t :trace} (p/generate em [] (cm/choicemap :obs (mx/scalar 1.0)))]
    (is (= :collapsed (tr/score-type t)) "precondition")
    (is (score-type-error?
          #((kern/mh-kernel (sel/select :coin)) t (rng/fresh-key 2)))
        "kern/mh-kernel throws on a collapsed trace")
    (is (score-type-error?
          #(mcmc/mh-step t (sel/select :coin) (rng/fresh-key 3)))
        "mcmc/mh-step throws on a collapsed trace")))

(deftest marginal-init-traces-still-flow-through-trace-mh
  ;; The legitimate pkmx flow must KEEP working: SBC inits mh-cycle chains
  ;; with analytical-generate (marginal) traces and relies on conversion.
  (let [model (dyn/with-key (conjugate-model) (rng/fresh-key 7))
        {t :trace} (p/generate model [] (cm/choicemap :y (mx/scalar 1.5)))]
    (is (= :marginal (tr/score-type t)) "precondition")
    (let [t' ((kern/mh-kernel (sel/select :mu)) t (rng/fresh-key 9))]
      (is (tr/trace? t') "kernel step runs without throwing"))
    (let [t'' (mcmc/mh-step t (sel/select :mu) (rng/fresh-key 10))]
      (is (tr/trace? t'') "mh-step runs without throwing"))))

;; ============================================================
;; 5. Backstop teeth — the law guarding the check itself (540f)
;; ============================================================

(deftest backstop-protects-eliminated-model-when-strip-bypassed
  ;; The 540f safety GOAL: if the trace-MH analytical strip ever regresses, an
  ;; eliminated model must NOT silently anchor chains at the posterior mean
  ;; (chi2 290.9 vs crit 21.67). Two protections now hold that line, both
  ;; exercised here with the strip forced to identity (the regressed-strip case):
  ;;
  ;;  (1) wl1y re-opening decline — selecting an eliminated latent makes the
  ;;      analytical regenerate dispatcher DECLINE (the selection re-opens the
  ;;      block), routing to the joint handler + joint-rescore-marginal. The
  ;;      latent is resampled, not pinned: a CORRECT joint move, so the result is
  ;;      :joint and no throw is needed — the chain is not corrupted. (Pre-wl1y
  ;;      this case threw; the throw was the only protection. The decline is the
  ;;      stronger one: it makes the move correct, not merely loud.)
  ;;  (2) assert-joint! teeth — see backstop-throws-on-marginal-residual-move:
  ;;      any :marginal result that still reaches trace-MH throws loudly.
  (let [tg (dyn/auto-key (two-gaussians))
        {t :trace} (p/generate tg [] tg-obs)]
    (is (= #{:a :b} (get-in (:schema tg) [:analytical-plan :rewrite-result :eliminated]))
        "precondition: model is statically eliminated")
    (is (= :marginal (tr/score-type t)) "precondition: analytical generate fired")
    ;; Direct regenerate of an eliminated latent (the move mh-kernel runs after the
    ;; — here bypassed — strip): the wl1y decline routes it to the joint handler +
    ;; conversion, so the RESULT trace is :joint (not a marginal anchor) with a
    ;; finite weight.
    (let [{rt :trace rw :weight}
          (p/regenerate (dyn/auto-key (:gen-fn t)) t (sel/select :a))]
      (is (= :joint (tr/score-type rt))
          "regenerate result is joint — wl1y decline → conversion, not a marginal anchor")
      (is (js/isFinite (mx/item rw)) "regenerate weight is finite"))
    ;; Backstop: with the strip forced to identity, mh-kernel/mh-step therefore do
    ;; NOT trip assert-joint! — the regenerate result they assert on is joint. (A
    ;; :marginal result would still throw; see the residual-move test below.) The
    ;; returned trace is the accept/reject outcome, so it may be the old :marginal
    ;; trace on rejection — the point is only that no corruption-guard fired.
    (with-redefs [dyn/strip-analytical-path identity]
      (is (tr/trace? ((kern/mh-kernel (sel/select :a)) t (rng/fresh-key 540)))
          "mh-kernel runs without throwing (joint regenerate result)")
      (is (tr/trace? (mcmc/mh-step t (sel/select :a) (rng/fresh-key 541)))
          "mh-step runs without throwing (joint regenerate result)"))))

(deftest backstop-throws-on-marginal-residual-move-when-strip-bypassed
  ;; The assert-joint! teeth are intact: with the strip bypassed, a residual MH
  ;; move on a model with an eliminated block does NOT re-open it, so the
  ;; analytical regenerate fires and returns a :marginal result — trace-MH must
  ;; THROW, never silently accept a marginal-scored step (genmlx-540f / wl1y).
  (let [m (dyn/auto-key (conj+residual))
        {t :trace} (p/generate m [] (cm/choicemap :y (mx/scalar 1.5)))]
    (is (= :marginal (tr/score-type t)) "precondition: analytical generate fired (mu eliminated)")
    (with-redefs [dyn/strip-analytical-path identity]
      (is (score-type-error?
            #((kern/mh-kernel (sel/select :tau)) t (rng/fresh-key 542)))
          "mh-kernel throws on a marginal residual move when the strip is bypassed")
      (is (score-type-error?
            #(mcmc/mh-step t (sel/select :tau) (rng/fresh-key 543)))
          "mh-step throws on a marginal residual move when the strip is bypassed"))))

;; ============================================================
;; 6. Strip consolidation — importance sampling on eliminated models
;; ============================================================

(deftest importance-particles-are-joint-and-diverse
  (let [model (dyn/auto-key (two-gaussians))
        {:keys [traces]} (imp/importance-sampling
                           {:samples 10 :key (rng/fresh-key 5)} model [] tg-obs)]
    (is (every? #(= :joint (tr/score-type %)) traces)
        "IS particles are joint-scored (analytical path stripped)")
    (is (> (count (distinct (map #(choice-val % :a) traces))) 1)
        "particles are diverse, not pinned at the posterior mean")))

;; ============================================================
;; 7. Serialization honesty
;; ============================================================

(deftest serialization-tags-and-reloads-honestly
  (let [model (dyn/with-key (conjugate-model) (rng/fresh-key 7))
        {t :trace} (p/generate model [] (cm/choicemap :y (mx/scalar 1.5)))
        json (ser/save-trace t)
        parsed (js->clj (js/JSON.parse json) :keywordize-keys true)]
    (is (= :marginal (tr/score-type t)) "precondition")
    (is (= "marginal" (:score-type parsed))
        "saved JSON declares its score encoding (the saved score IS marginal)")
    (testing "load-trace re-generates: reloaded trace is freshly joint-scored"
      (let [t2 (ser/load-trace (dyn/auto-key model) json)]
        (is (= :joint (tr/score-type t2))
            "fully-constrained re-generate falls through to joint (b470)")))))

;; ============================================================
;; 8. GFI law registered
;; ============================================================

(deftest gfi-law-registered
  (is (contains? (set (map :name gfi/laws)) :score-type-soundness)
      "score-type soundness is part of the law catalog"))

(cljs.test/run-tests)
