(ns genmlx.gfi-compositional-test
  "Property-based verification that GFI laws are preserved by combinators.
   Tests thesis Section 2.1.5: Map, Unfold, and Switch preserve well-formedness.

   The central theorem: wrapping a well-formed generative function in a
   combinator yields a well-formed generative function. All 8 testable
   algebraic laws (simulate-score, generate-empty, generate-full,
   update-identity, update-density-ratio, project-all, project-none,
   propose-generate) must hold for the composite."
  (:require [cljs.test :as t]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.dynamic :as dyn]
            [genmlx.combinators :as comb]
            [genmlx.dist :as dist]
            [genmlx.gfi :as gfi]
            [genmlx.gfi-compiler :as compiler]
            [genmlx.gfi-gen :as model-gen]
            [genmlx.gfi-law-checkers :as laws]
            [genmlx.test-helpers :as h])
  (:require-macros [genmlx.gen :refer [gen]]
                   [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- spec->handler-gf
  "Compile a model spec to a DynamicGF with compiled paths stripped,
   forcing the handler (interpreter) execution path."
  [spec]
  (-> (compiler/spec->gf spec) gfi/strip-compiled dyn/auto-key))

(defn- spec->handler-gf-safe
  "Like spec->handler-gf but returns nil on compilation failure."
  [spec]
  (try (spec->handler-gf spec)
       (catch :default _ nil)))

(def ^:private law-checkers
  "The 9 law checkers applicable to combinator-wrapped generative functions.
   Includes hierarchical project-decomposition for combinator address paths."
  [laws/check-simulate-score-law
   laws/check-generate-empty-weight
   laws/check-generate-full-weight
   laws/check-update-identity-law
   laws/check-update-density-ratio
   laws/check-project-all-equals-score
   laws/check-project-none-equals-zero
   laws/check-propose-generate-consistency])

(defn- check-combinator-laws
  "Run all applicable law checkers plus hierarchical project-decomposition.
   Returns true when every law passes."
  [gf args]
  (let [results (conj (mapv #(% gf args) law-checkers)
                      (laws/check-project-decomposition-hierarchical gf args))
        failures (remove :pass? results)]
    (when (seq failures)
      (doseq [{:keys [law detail]} failures]
        (println "    FAIL" law ":" detail)))
    (empty? failures)))

;; ---------------------------------------------------------------------------
;; Map combinator generators
;; ---------------------------------------------------------------------------

(def gen-map-element-count
  "2-4 elements per Map invocation."
  (gen/choose 2 4))

(def gen-map-inputs
  "Generate Map arguments: a vector of mx/scalar values."
  (gen/fmap (fn [n] (mapv mx/scalar (repeat n 1.0)))
            gen-map-element-count))

;; ---------------------------------------------------------------------------
;; Unfold kernel pool
;; ---------------------------------------------------------------------------
;; Unfold kernels must accept [t state] and return a new state.
;; These cannot be generated from gen-kernel-spec (wrong arity),
;; so we build a small pool of hand-crafted kernels.

(def ^:private unfold-kernel-gaussian
  "Unfold kernel: x ~ N(state, 0.1), returns x as next state."
  (dyn/auto-key
    (gen [t state]
      (let [x (trace :x (dist/gaussian state 0.1))]
        (mx/eval! x)
        (mx/item x)))))

(def ^:private unfold-kernel-laplace
  "Unfold kernel: x ~ Laplace(state, 0.5), returns x as next state."
  (dyn/auto-key
    (gen [t state]
      (let [x (trace :x (dist/laplace state 0.5))]
        (mx/eval! x)
        (mx/item x)))))

(def ^:private unfold-kernel-two-site
  "Unfold kernel: z ~ N(state, 1), y ~ N(z, 0.5), returns y as next state."
  (dyn/auto-key
    (gen [t state]
      (let [z (trace :z (dist/gaussian state 1))]
        (mx/eval! z)
        (let [y (trace :y (dist/gaussian (mx/item z) 0.5))]
          (mx/eval! y)
          (mx/item y))))))

(def ^:private unfold-kernel-pool
  [unfold-kernel-gaussian unfold-kernel-laplace unfold-kernel-two-site])

(def gen-unfold-kernel
  "Generator: pick one kernel from the hand-built pool."
  (gen/elements unfold-kernel-pool))

(def gen-unfold-steps
  "2-4 steps for Unfold."
  (gen/choose 2 4))

;; ---------------------------------------------------------------------------
;; Map combinator: 8 laws on random kernels
;; [T] Section 2.1.5 — Map preserves GFI well-formedness
;; ---------------------------------------------------------------------------

(defspec map-preserves-simulate-score-law 30
  (prop/for-all [spec model-gen/gen-kernel-spec
                 inputs gen-map-inputs]
    (if-let [kernel (spec->handler-gf-safe spec)]
      (let [mapped (comb/map-combinator kernel)]
        (:pass? (laws/check-simulate-score-law mapped [inputs])))
      true)))

(defspec map-preserves-generate-empty-weight 30
  (prop/for-all [spec model-gen/gen-kernel-spec
                 inputs gen-map-inputs]
    (if-let [kernel (spec->handler-gf-safe spec)]
      (let [mapped (comb/map-combinator kernel)]
        (:pass? (laws/check-generate-empty-weight mapped [inputs])))
      true)))

(defspec map-preserves-generate-full-weight 30
  (prop/for-all [spec model-gen/gen-kernel-spec
                 inputs gen-map-inputs]
    (if-let [kernel (spec->handler-gf-safe spec)]
      (let [mapped (comb/map-combinator kernel)]
        (:pass? (laws/check-generate-full-weight mapped [inputs])))
      true)))

(defspec map-preserves-update-identity 30
  (prop/for-all [spec model-gen/gen-kernel-spec
                 inputs gen-map-inputs]
    (if-let [kernel (spec->handler-gf-safe spec)]
      (let [mapped (comb/map-combinator kernel)]
        (:pass? (laws/check-update-identity-law mapped [inputs])))
      true)))

(defspec map-preserves-update-density-ratio 30
  (prop/for-all [spec model-gen/gen-kernel-spec
                 inputs gen-map-inputs]
    (if-let [kernel (spec->handler-gf-safe spec)]
      (let [mapped (comb/map-combinator kernel)]
        (:pass? (laws/check-update-density-ratio mapped [inputs])))
      true)))

(defspec map-preserves-project-all-equals-score 30
  (prop/for-all [spec model-gen/gen-kernel-spec
                 inputs gen-map-inputs]
    (if-let [kernel (spec->handler-gf-safe spec)]
      (let [mapped (comb/map-combinator kernel)]
        (:pass? (laws/check-project-all-equals-score mapped [inputs])))
      true)))

(defspec map-preserves-project-none-equals-zero 30
  (prop/for-all [spec model-gen/gen-kernel-spec
                 inputs gen-map-inputs]
    (if-let [kernel (spec->handler-gf-safe spec)]
      (let [mapped (comb/map-combinator kernel)]
        (:pass? (laws/check-project-none-equals-zero mapped [inputs])))
      true)))

(defspec map-preserves-propose-generate-consistency 30
  (prop/for-all [spec model-gen/gen-kernel-spec
                 inputs gen-map-inputs]
    (if-let [kernel (spec->handler-gf-safe spec)]
      (let [mapped (comb/map-combinator kernel)]
        (:pass? (laws/check-propose-generate-consistency mapped [inputs])))
      true)))

(defspec map-preserves-project-decomposition 30
  (prop/for-all [spec model-gen/gen-kernel-spec
                 inputs gen-map-inputs]
    (if-let [kernel (spec->handler-gf-safe spec)]
      (let [mapped (comb/map-combinator kernel)]
        (:pass? (laws/check-project-decomposition-hierarchical mapped [inputs])))
      true)))

;; ---------------------------------------------------------------------------
;; Unfold combinator: 9 laws on random kernels
;; [T] Section 2.1.5 — Unfold preserves GFI well-formedness
;; ---------------------------------------------------------------------------

(defspec unfold-preserves-simulate-score-law 30
  (prop/for-all [kernel gen-unfold-kernel
                 n-steps gen-unfold-steps]
    (let [unfold (comb/unfold-combinator kernel)]
      (:pass? (laws/check-simulate-score-law unfold [n-steps 0.0])))))

(defspec unfold-preserves-generate-empty-weight 30
  (prop/for-all [kernel gen-unfold-kernel
                 n-steps gen-unfold-steps]
    (let [unfold (comb/unfold-combinator kernel)]
      (:pass? (laws/check-generate-empty-weight unfold [n-steps 0.0])))))

(defspec unfold-preserves-generate-full-weight 30
  (prop/for-all [kernel gen-unfold-kernel
                 n-steps gen-unfold-steps]
    (let [unfold (comb/unfold-combinator kernel)]
      (:pass? (laws/check-generate-full-weight unfold [n-steps 0.0])))))

(defspec unfold-preserves-update-identity 30
  (prop/for-all [kernel gen-unfold-kernel
                 n-steps gen-unfold-steps]
    (let [unfold (comb/unfold-combinator kernel)]
      (:pass? (laws/check-update-identity-law unfold [n-steps 0.0])))))

(defspec unfold-preserves-update-density-ratio 30
  (prop/for-all [kernel gen-unfold-kernel
                 n-steps gen-unfold-steps]
    (let [unfold (comb/unfold-combinator kernel)]
      (:pass? (laws/check-update-density-ratio unfold [n-steps 0.0])))))

(defspec unfold-preserves-project-all-equals-score 30
  (prop/for-all [kernel gen-unfold-kernel
                 n-steps gen-unfold-steps]
    (let [unfold (comb/unfold-combinator kernel)]
      (:pass? (laws/check-project-all-equals-score unfold [n-steps 0.0])))))

(defspec unfold-preserves-project-none-equals-zero 30
  (prop/for-all [kernel gen-unfold-kernel
                 n-steps gen-unfold-steps]
    (let [unfold (comb/unfold-combinator kernel)]
      (:pass? (laws/check-project-none-equals-zero unfold [n-steps 0.0])))))

(defspec unfold-preserves-propose-generate-consistency 30
  (prop/for-all [kernel gen-unfold-kernel
                 n-steps gen-unfold-steps]
    (let [unfold (comb/unfold-combinator kernel)]
      (:pass? (laws/check-propose-generate-consistency unfold [n-steps 0.0])))))

(defspec unfold-preserves-project-decomposition 30
  (prop/for-all [kernel gen-unfold-kernel
                 n-steps gen-unfold-steps]
    (let [unfold (comb/unfold-combinator kernel)]
      (:pass? (laws/check-project-decomposition-hierarchical unfold [n-steps 0.0])))))

;; ---------------------------------------------------------------------------
;; Switch combinator: 9 laws on random branch models
;; [T] Section 2.1.5 — Switch preserves GFI well-formedness
;; ---------------------------------------------------------------------------

(def gen-branch-index
  "Generator: 0 or 1 for two-branch Switch."
  (gen/elements [0 1]))

(defspec switch-preserves-simulate-score-law 30
  (prop/for-all [spec0 model-gen/gen-model-spec
                 spec1 model-gen/gen-model-spec
                 idx gen-branch-index]
    (let [b0 (spec->handler-gf-safe spec0)
          b1 (spec->handler-gf-safe spec1)]
      (if (and b0 b1)
        (let [sw (comb/switch-combinator b0 b1)]
          (:pass? (laws/check-simulate-score-law sw [idx])))
        true))))

(defspec switch-preserves-generate-empty-weight 30
  (prop/for-all [spec0 model-gen/gen-model-spec
                 spec1 model-gen/gen-model-spec
                 idx gen-branch-index]
    (let [b0 (spec->handler-gf-safe spec0)
          b1 (spec->handler-gf-safe spec1)]
      (if (and b0 b1)
        (let [sw (comb/switch-combinator b0 b1)]
          (:pass? (laws/check-generate-empty-weight sw [idx])))
        true))))

(defspec switch-preserves-generate-full-weight 30
  (prop/for-all [spec0 model-gen/gen-model-spec
                 spec1 model-gen/gen-model-spec
                 idx gen-branch-index]
    (let [b0 (spec->handler-gf-safe spec0)
          b1 (spec->handler-gf-safe spec1)]
      (if (and b0 b1)
        (let [sw (comb/switch-combinator b0 b1)]
          (:pass? (laws/check-generate-full-weight sw [idx])))
        true))))

(defspec switch-preserves-update-identity 30
  (prop/for-all [spec0 model-gen/gen-model-spec
                 spec1 model-gen/gen-model-spec
                 idx gen-branch-index]
    (let [b0 (spec->handler-gf-safe spec0)
          b1 (spec->handler-gf-safe spec1)]
      (if (and b0 b1)
        (let [sw (comb/switch-combinator b0 b1)]
          (:pass? (laws/check-update-identity-law sw [idx])))
        true))))

(defspec switch-preserves-update-density-ratio 30
  (prop/for-all [spec0 model-gen/gen-model-spec
                 spec1 model-gen/gen-model-spec
                 idx gen-branch-index]
    (let [b0 (spec->handler-gf-safe spec0)
          b1 (spec->handler-gf-safe spec1)]
      (if (and b0 b1)
        (let [sw (comb/switch-combinator b0 b1)]
          (:pass? (laws/check-update-density-ratio sw [idx])))
        true))))

(defspec switch-preserves-project-all-equals-score 30
  (prop/for-all [spec0 model-gen/gen-model-spec
                 spec1 model-gen/gen-model-spec
                 idx gen-branch-index]
    (let [b0 (spec->handler-gf-safe spec0)
          b1 (spec->handler-gf-safe spec1)]
      (if (and b0 b1)
        (let [sw (comb/switch-combinator b0 b1)]
          (:pass? (laws/check-project-all-equals-score sw [idx])))
        true))))

(defspec switch-preserves-project-none-equals-zero 30
  (prop/for-all [spec0 model-gen/gen-model-spec
                 spec1 model-gen/gen-model-spec
                 idx gen-branch-index]
    (let [b0 (spec->handler-gf-safe spec0)
          b1 (spec->handler-gf-safe spec1)]
      (if (and b0 b1)
        (let [sw (comb/switch-combinator b0 b1)]
          (:pass? (laws/check-project-none-equals-zero sw [idx])))
        true))))

(defspec switch-preserves-propose-generate-consistency 30
  (prop/for-all [spec0 model-gen/gen-model-spec
                 spec1 model-gen/gen-model-spec
                 idx gen-branch-index]
    (let [b0 (spec->handler-gf-safe spec0)
          b1 (spec->handler-gf-safe spec1)]
      (if (and b0 b1)
        (let [sw (comb/switch-combinator b0 b1)]
          (:pass? (laws/check-propose-generate-consistency sw [idx])))
        true))))

(defspec switch-preserves-project-decomposition 30
  (prop/for-all [spec0 model-gen/gen-model-spec
                 spec1 model-gen/gen-model-spec
                 idx gen-branch-index]
    (let [b0 (spec->handler-gf-safe spec0)
          b1 (spec->handler-gf-safe spec1)]
      (if (and b0 b1)
        (let [sw (comb/switch-combinator b0 b1)]
          (:pass? (laws/check-project-decomposition-hierarchical sw [idx])))
        true))))

;; ---------------------------------------------------------------------------
;; Composite: combinator type x law matrix
;; [T] Section 2.1.5 — all combinators preserve all applicable laws
;; ---------------------------------------------------------------------------

(def gen-combinator-type
  "Generator: one of the three combinator types."
  (gen/elements [:map :unfold :switch]))

(defspec all-combinators-preserve-all-laws 30
  (prop/for-all [comb-type gen-combinator-type
                 kernel-spec model-gen/gen-kernel-spec
                 model-spec0 model-gen/gen-model-spec
                 model-spec1 model-gen/gen-model-spec
                 unfold-kernel gen-unfold-kernel
                 n-steps gen-unfold-steps
                 inputs gen-map-inputs
                 idx gen-branch-index]
    (case comb-type
      :map
      (if-let [kernel (spec->handler-gf-safe kernel-spec)]
        (check-combinator-laws (comb/map-combinator kernel) [inputs])
        true)

      :unfold
      (check-combinator-laws (comb/unfold-combinator unfold-kernel)
                             [n-steps 0.0])

      :switch
      (let [b0 (spec->handler-gf-safe model-spec0)
            b1 (spec->handler-gf-safe model-spec1)]
        (if (and b0 b1)
          (check-combinator-laws (comb/switch-combinator b0 b1) [idx])
          true)))))

;; ---------------------------------------------------------------------------
;; Splice weight decomposition [T] Proposition 2.3.2
;;
;; For a composed model P_outer that splices P_inner under address :sub:
;;   update(P_outer, t1, t2.choices).weight
;;     = update(P_inner, t1_inner, t2_inner.choices).weight
;;     + [outer_score(t2) - outer_score(t1)]
;;
;; where outer_score is the log-density of the outer-only trace sites
;; (everything except the splice sub-trace).
;;
;; Simpler testable form: the composed model must satisfy the same
;; update-identity and update-density-ratio laws as any GF. We also
;; verify that the basic GFI laws hold for splice-composed models
;; with randomly generated inner models.
;; ---------------------------------------------------------------------------

(defn- make-splice-model
  "Build a model that splices `inner` under :sub, then traces :bridge ~ N(a, 1)
   where a is the inner model's return value."
  [inner]
  (let [body-fn (fn [rt]
                  (let [splice-fn (.-splice rt)
                        trace-fn  (.-trace rt)
                        a (splice-fn :sub inner)]
                    (trace-fn :bridge (dist/gaussian a 1))))
        source '([] (let [a (splice :sub inner)]
                      (trace :bridge (dist/gaussian a 1))))]
    (dyn/auto-key (dyn/make-gen-fn body-fn source))))

(defspec splice-preserves-simulate-score-law 30
  (prop/for-all [spec model-gen/gen-model-spec]
    (if-let [inner (spec->handler-gf-safe spec)]
      (let [composed (make-splice-model inner)]
        (:pass? (laws/check-simulate-score-law composed [])))
      true)))

(defspec splice-preserves-generate-empty-weight 30
  (prop/for-all [spec model-gen/gen-model-spec]
    (if-let [inner (spec->handler-gf-safe spec)]
      (let [composed (make-splice-model inner)]
        (:pass? (laws/check-generate-empty-weight composed [])))
      true)))

(defspec splice-preserves-update-identity 30
  (prop/for-all [spec model-gen/gen-model-spec]
    (if-let [inner (spec->handler-gf-safe spec)]
      (let [composed (make-splice-model inner)]
        (:pass? (laws/check-update-identity-law composed [])))
      true)))

(defspec splice-preserves-update-density-ratio 30
  (prop/for-all [spec model-gen/gen-model-spec]
    (if-let [inner (spec->handler-gf-safe spec)]
      (let [composed (make-splice-model inner)]
        (:pass? (laws/check-update-density-ratio composed [])))
      true)))

(defspec splice-preserves-project-all-equals-score 30
  (prop/for-all [spec model-gen/gen-model-spec]
    (if-let [inner (spec->handler-gf-safe spec)]
      (let [composed (make-splice-model inner)]
        (:pass? (laws/check-project-all-equals-score composed [])))
      true)))

(defspec splice-preserves-propose-generate-consistency 30
  (prop/for-all [spec model-gen/gen-model-spec]
    (if-let [inner (spec->handler-gf-safe spec)]
      (let [composed (make-splice-model inner)]
        (:pass? (laws/check-propose-generate-consistency composed [])))
      true)))

;; ---------------------------------------------------------------------------
;; Splice weight DECOMPOSITION [T] Proposition 2.3.2
;;
;; The specific algebraic claim: for a composed model P that splices inner
;; model Q under address :sub and then traces :bridge:
;;
;;   update(P, t1, t2.choices).weight
;;     = update(Q, t1[:sub], t2[:sub].choices).weight
;;     + [log p(:bridge | t2[:sub].retval) - log p(:bridge | t1[:sub].retval)]
;;
;; In other words: the composed update weight decomposes into the inner
;; update weight plus the outer-only density ratio. This is the chain rule
;; for importance weights under composition.
;; ---------------------------------------------------------------------------

(defspec splice-weight-decomposes 30
  (prop/for-all [spec model-gen/gen-model-spec]
    (if-let [inner (spec->handler-gf-safe spec)]
      (let [composed (make-splice-model inner)
            ;; Two independent traces from the composed model
            t1 (p/simulate composed [])
            t2 (p/simulate composed [])
            ;; Composed update weight
            {:keys [weight]} (p/update composed t1 (:choices t2))
            w-composed (h/realize weight)
            ;; Inner sub-traces
            inner-choices-1 (cm/get-submap (:choices t1) :sub)
            inner-choices-2 (cm/get-submap (:choices t2) :sub)
            ;; Inner update weight: update inner model from t1[:sub] to t2[:sub]
            inner-t1 (p/simulate inner [])  ;; need a trace object for inner
            ;; Actually: use generate to build inner traces from the choices
            {:keys [trace]} (p/generate inner [] inner-choices-1)
            inner-trace-1 trace
            {:keys [weight]} (p/update inner inner-trace-1 inner-choices-2)
            w-inner (h/realize weight)
            ;; Bridge-only density ratio:
            ;; bridge ~ N(retval_inner, 1)
            ;; log p(bridge_2 | retval_2) - log p(bridge_1 | retval_1)
            ;; where bridge_i and retval_i come from trace i
            bridge-val-1 (h/realize (cm/get-value (cm/get-submap (:choices t1) :bridge)))
            bridge-val-2 (h/realize (cm/get-value (cm/get-submap (:choices t2) :bridge)))
            retval-1 (h/realize (:retval inner-trace-1))
            retval-2-trace (:trace (p/generate inner [] inner-choices-2))
            retval-2 (h/realize (:retval retval-2-trace))
            ;; log N(bridge; retval, 1) = -0.5*log(2pi) - 0.5*(bridge - retval)^2
            bridge-lp (fn [b r] (- (* -0.5 (js/Math.log (* 2 js/Math.PI)))
                                   (* 0.5 (js/Math.pow (- b r) 2))))
            w-bridge (- (bridge-lp bridge-val-2 retval-2)
                        (bridge-lp bridge-val-1 retval-1))
            ;; Prop 2.3.2: w_composed = w_inner + w_bridge
            expected (+ w-inner w-bridge)
            ok? (and (h/finite? w-composed) (h/finite? expected)
                     (h/close? w-composed expected 1e-2))]
        (when-not ok?
          (println "\n  DECOMPOSITION FAIL:"
                   "w-composed=" w-composed "w-inner=" w-inner
                   "w-bridge=" w-bridge "expected=" expected
                   "diff=" (js/Math.abs (- w-composed expected))))
        ok?)
      true)))

;; ---------------------------------------------------------------------------
;; Run
;; ---------------------------------------------------------------------------

(defn -main []
  (t/run-tests 'genmlx.gfi-compositional-test))

(-main)
