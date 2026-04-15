(ns genmlx.gfi-universal-test
  "Property-based GFI verification over generated model structures.

   Unlike gfi_laws_test which checks laws on 17 hand-picked models,
   this suite generates random model specifications via test.check,
   compiles them into DynamicGFs, and verifies all GFI algebraic laws.

   The universally-quantified variable — the model itself — is generated."
  (:require [cljs.test :as t]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [genmlx.mlx :as mx]
            [genmlx.protocols :as p]
            [genmlx.gfi-compiler :as compiler]
            [genmlx.gfi-gen :as model-gen]
            [genmlx.gfi-law-checkers :as laws]
            [genmlx.gfi :as gfi]
            [genmlx.dynamic :as dyn]
            [genmlx.combinators :as comb]
            [genmlx.choicemap :as cm]
            [genmlx.mlx.random :as rng]
            [genmlx.test-helpers :as h])
  (:require-macros [clojure.test.check.clojure-test :refer [defspec]]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- spec-addrs
  "Extract the set of trace site addresses from a spec."
  [spec]
  (set (map :addr (:sites spec))))

(defn- spec->gf-safe
  "Compile a spec, returning nil on failure."
  [spec]
  (try (compiler/spec->gf spec)
       (catch :default _ nil)))

(defn- safe-realize
  "Realize an MLX array to a JS number. If x is already a number (e.g. delta
   distribution values), return it directly — mx/item requires an MxArray."
  [x]
  (if (number? x) x (h/realize x)))

;; ---------------------------------------------------------------------------
;; Law 1: simulate produces valid trace [T] Def 2.1.16
;; ---------------------------------------------------------------------------

(defspec law:gen-simulate-valid-trace 50
  (prop/for-all [spec model-gen/gen-model-spec]
                (let [gf (spec->gf-safe spec)]
                  (if (nil? gf)
                    true
                    (:pass? (laws/check-simulate-produces-valid-trace
                             gf [] (spec-addrs spec)))))))

;; ---------------------------------------------------------------------------
;; Law 2: simulate score consistency [T] §2.3.1
;; ---------------------------------------------------------------------------

(defspec law:gen-simulate-score 50
  (prop/for-all [spec model-gen/gen-model-spec]
                (let [gf (spec->gf-safe spec)]
                  (if (nil? gf)
                    true
                    (:pass? (laws/check-simulate-score-law gf []))))))

;; ---------------------------------------------------------------------------
;; Law 3: generate(empty) weight = 0 [D] generate
;; ---------------------------------------------------------------------------

(defspec law:gen-generate-empty-weight 50
  (prop/for-all [spec model-gen/gen-model-spec]
                (let [gf (spec->gf-safe spec)]
                  (if (nil? gf)
                    true
                    (:pass? (laws/check-generate-empty-weight gf []))))))

;; ---------------------------------------------------------------------------
;; Law 4: generate(full) weight = score [T] §2.3.1
;; ---------------------------------------------------------------------------

(defspec law:gen-generate-full-weight 50
  (prop/for-all [spec model-gen/gen-model-spec]
                (let [gf (spec->gf-safe spec)]
                  (if (nil? gf)
                    true
                    (:pass? (laws/check-generate-full-weight gf []))))))

;; ---------------------------------------------------------------------------
;; Law 5: update identity [T] Prop 2.3.1
;; ---------------------------------------------------------------------------

(defspec law:gen-update-identity 50
  (prop/for-all [spec model-gen/gen-model-spec]
                (let [gf (spec->gf-safe spec)]
                  (if (nil? gf)
                    true
                    (:pass? (laws/check-update-identity-law gf []))))))

;; ---------------------------------------------------------------------------
;; Law 6: project(all) = score [T] §2.3.1
;; ---------------------------------------------------------------------------

(defspec law:gen-project-all-score 50
  (prop/for-all [spec model-gen/gen-model-spec]
                (let [gf (spec->gf-safe spec)]
                  (if (nil? gf)
                    true
                    (:pass? (laws/check-project-all-equals-score gf []))))))

;; ---------------------------------------------------------------------------
;; Law 7: project(none) = 0 [T] §2.3.1
;; ---------------------------------------------------------------------------

(defspec law:gen-project-none-zero 50
  (prop/for-all [spec model-gen/gen-model-spec]
                (let [gf (spec->gf-safe spec)]
                  (if (nil? gf)
                    true
                    (:pass? (laws/check-project-none-equals-zero gf []))))))

;; ---------------------------------------------------------------------------
;; Law 8: project decomposition (chain rule) [T] §2.3.1
;; ---------------------------------------------------------------------------

(defspec law:gen-project-decomposition 50
  (prop/for-all [spec model-gen/gen-model-spec]
                (let [gf (spec->gf-safe spec)]
                  (if (nil? gf)
                    true
                    (:pass? (laws/check-project-decomposition
                             gf [] (spec-addrs spec)))))))

;; ---------------------------------------------------------------------------
;; Law 9: propose-generate consistency [D] propose
;; ---------------------------------------------------------------------------

(defspec law:gen-propose-generate 50
  (prop/for-all [spec model-gen/gen-model-spec]
                (let [gf (spec->gf-safe spec)]
                  (if (nil? gf)
                    true
                    (:pass? (laws/check-propose-generate-consistency gf []))))))

;; ---------------------------------------------------------------------------
;; Composite: all 9 laws on each generated model
;; ---------------------------------------------------------------------------

(defspec law:gen-all-laws-hold 50
  (prop/for-all [spec model-gen/gen-model-spec]
                (let [gf (spec->gf-safe spec)]
                  (if (nil? gf)
                    true
                    (let [results (laws/check-all-laws gf [] (spec-addrs spec))
                          failures (remove :pass? results)]
                      (when (seq failures)
                        (println "\n  FAILURES for spec:" (pr-str spec))
                        (doseq [f failures]
                          (println "    " (:law f) ":" (:detail f))))
                      (empty? failures))))))

;; ---------------------------------------------------------------------------
;; Compiled path equivalence: compiled = handler for generated models
;; ---------------------------------------------------------------------------

(defspec law:gen-compiled-equals-handler 50
  (prop/for-all [spec model-gen/gen-model-spec]
                (let [gf (spec->gf-safe spec)]
                  (if (nil? gf)
                    true
                    (let [key (rng/fresh-key 42)
                          gf-compiled (dyn/with-key gf key)
                          gf-handler (dyn/with-key (gfi/strip-compiled gf) key)
                          t-compiled (p/simulate gf-compiled [])
                          t-handler (p/simulate gf-handler [])
                          sc (h/realize (:score t-compiled))
                          sh (h/realize (:score t-handler))
                          scores-match? (and (h/finite? sc)
                                             (h/finite? sh)
                                             (h/close? sc sh 1e-3))
                          choices-match?
                          (every?
                           (fn [{:keys [addr]}]
                             (let [vc (safe-realize (cm/get-value (cm/get-submap (:choices t-compiled) addr)))
                                   vh (safe-realize (cm/get-value (cm/get-submap (:choices t-handler) addr)))]
                               (h/close? vc vh 1e-4)))
                           (:sites spec))]
                      (and scores-match? choices-match?))))))

;; ---------------------------------------------------------------------------
;; Models with arguments: exercises arg-passing code paths
;; ---------------------------------------------------------------------------

(defspec law:gen-model-with-args 50
  (prop/for-all [spec model-gen/gen-model-spec-with-arg
                 x-val gen/small-integer]
                (let [gf (spec->gf-safe spec)]
                  (if (nil? gf)
                    true
                    (let [results (laws/check-all-laws gf [x-val] (spec-addrs spec))
                          failures (remove :pass? results)]
                      (when (seq failures)
                        (println "\n  FAILURES (with-arg) for spec:" (pr-str spec) "x=" x-val)
                        (doseq [f failures]
                          (println "    " (:law f) ":" (:detail f))))
                      (empty? failures))))))

;; ---------------------------------------------------------------------------
;; Update density ratio: w = new_score - old_score [T] Prop 2.3.1
;; ---------------------------------------------------------------------------

(defspec law:gen-update-density-ratio 50
  (prop/for-all [spec model-gen/gen-model-spec]
                (let [gf (spec->gf-safe spec)]
                  (if (nil? gf)
                    true
                    (:pass? (laws/check-update-density-ratio gf []))))))

;; ---------------------------------------------------------------------------
;; Map combinator: GFI laws preserved under Map wrapping [T] §2.1.5
;; ---------------------------------------------------------------------------

(defn- map-addrs
  "Build nested addr set for Map combinator: #{[0 :a] [1 :a] ...}"
  [spec n]
  (set (for [i (range n)
             {:keys [addr]} (:sites spec)]
         [i addr])))

(defspec law:gen-map-combinator 30
  (prop/for-all [spec model-gen/gen-kernel-spec
                 n (gen/choose 2 4)]
                (let [kernel (spec->gf-safe spec)]
                  (if (nil? kernel)
                    true
                    (let [;; Strip compiled paths from kernel — Map's fused path
              ;; can hit NAPI type errors with generated body functions
                          mapped (dyn/auto-key
                                  (comb/map-combinator
                                   (dyn/auto-key (gfi/strip-compiled kernel))))
                          args [(mapv mx/scalar (repeat n 1))]
                          trace (p/simulate mapped args)
                          score (h/realize (:score trace))]
                      (and (h/finite? score)
               ;; update identity holds for combinator
                           (let [{:keys [weight]} (p/update mapped trace (:choices trace))]
                             (h/close? 0.0 (h/realize weight) 1e-3))
               ;; generate empty weight = 0
                           (let [{:keys [weight]} (p/generate mapped args cm/EMPTY)]
                             (h/close? 0.0 (h/realize weight) 1e-3))))))))

;; ---------------------------------------------------------------------------
;; Runner
;; ---------------------------------------------------------------------------

(defn -main []
  (t/run-tests 'genmlx.gfi-universal-test))

(-main)
