;; @tier fast core
(ns genmlx.update-weight-test
  "Thesis/Gen.jl update-weight convention (genmlx-8l8j).

   update(t, sigma).weight = (non-fresh score of t') - score(t), where the
   non-fresh score counts every constrained or retained site under the new
   parameters. Freshly sampled addresses (drawn from the internal proposal)
   cancel out of the weight; removed and overwritten old choices are charged
   via the recorded old score. Equivalently:

       weight = (score' - score) - project(t', fresh-addresses)

   The previous convention returned the raw score delta, which wrongly
   included the log-probs of freshly sampled latents — biasing SMC whenever
   an update extends a model's structure. The batched vupdate path used a
   third convention (constrained sums only), which missed retained sites
   whose parents changed. All paths now agree on the thesis convention."
  (:require-macros [genmlx.gen :refer [gen]])
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.protocols :as p]
            [genmlx.dynamic :as dyn]
            [genmlx.dist :as dist]
            [genmlx.dist.core :as dc]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.combinators :as comb]
            [genmlx.vmap :as vmap]
            [genmlx.gfi :as gfi]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]))

(defn- lp [d v] (h/realize (dc/dist-log-prob d v)))

;; ---------------------------------------------------------------------------
;; Structure change: branch flip removes one address and freshly samples another
;; ---------------------------------------------------------------------------

(def gate-model
  (gen []
    (let [b (trace :b (dist/bernoulli 0.5))]
      (if (pos? (mx/item b))
        (trace :w (dist/gaussian 0 1))
        (trace :y (dist/gaussian 2 1))))))

(deftest branch-flip-update-weight
  (testing "flipping :b removes :y (charged via old score) and freshly samples :w (cancels)"
    (let [obs (cm/choicemap :b (mx/scalar 0.0) :y (mx/scalar 1.5))
          {t0 :trace} (p/generate (dyn/with-key gate-model (rng/fresh-key 7)) [] obs)
          {:keys [trace weight]} (p/update (dyn/with-key gate-model (rng/fresh-key 8))
                                           t0 (cm/choicemap :b (mx/scalar 1.0)))
          ;; lp_b(1) - (lp_b(0) + lp_y(1.5)) = -lp_y(1.5)
          expected (- (lp (dist/gaussian 2 1) (mx/scalar 1.5)))]
      (is (cm/has-value? (cm/get-submap (:choices trace) :w)) "new branch sampled :w")
      (is (h/close? expected (h/realize weight) 1e-3)
          "weight excludes the fresh :w log-prob"))))

(deftest branch-flip-round-trip
  (testing "forward + backward weights cancel up to the fresh forward sample"
    (let [obs (cm/choicemap :b (mx/scalar 0.0) :y (mx/scalar 1.5))
          {t0 :trace} (p/generate (dyn/with-key gate-model (rng/fresh-key 17)) [] obs)
          {t1 :trace w-fwd :weight discard :discard}
          (p/update (dyn/with-key gate-model (rng/fresh-key 18))
                    t0 (cm/choicemap :b (mx/scalar 1.0)))
          fresh-lp (h/realize (p/project (dyn/with-key gate-model (rng/fresh-key 19))
                                         t1 (sel/select :w)))
          {t2 :trace w-bwd :weight}
          (p/update (dyn/with-key gate-model (rng/fresh-key 20)) t1 discard)]
      (is (h/close? (h/realize (:score t0)) (h/realize (:score t2)) 1e-3)
          "backward update with the discard restores the original score")
      (is (h/close? (- fresh-lp)
                    (+ (h/realize w-fwd) (h/realize w-bwd))
                    1e-3)
          "w_fwd + w_bwd = -project(t', fresh)"))))

;; ---------------------------------------------------------------------------
;; Structure extension: a flipped gate brings a fresh latent into existence
;; ---------------------------------------------------------------------------

(def extend-model
  (gen []
    (let [more (trace :more (dist/bernoulli 0.3))]
      (when (pos? (mx/item more))
        (trace :extra (dist/gaussian 0 1)))
      (trace :obs (dist/gaussian 0 1)))))

(deftest extension-update-weight-excludes-fresh-latent
  (testing "SMC-style extension: fresh latent cancels, weight = gate lp delta"
    (let [obs (cm/choicemap :more (mx/scalar 0.0) :obs (mx/scalar 0.7))
          {t0 :trace} (p/generate (dyn/with-key extend-model (rng/fresh-key 27)) [] obs)
          {:keys [trace weight]} (p/update (dyn/with-key extend-model (rng/fresh-key 28))
                                           t0 (cm/choicemap :more (mx/scalar 1.0)))
          expected (- (js/Math.log 0.3) (js/Math.log 0.7))]
      (is (cm/has-value? (cm/get-submap (:choices trace) :extra)) "fresh :extra sampled")
      (is (h/close? expected (h/realize weight) 1e-3)
          "weight is independent of the fresh :extra value"))))

;; ---------------------------------------------------------------------------
;; Fixed structure: constrained site feeds a retained downstream site
;; ---------------------------------------------------------------------------

(def dep-model
  (gen []
    (let [x (trace :x (dist/gaussian 0 1))]
      (trace :y (dist/gaussian x 1)))))

(deftest dependent-retained-scalar-and-batched-agree
  (testing "scalar update: weight = full lp delta including retained :y under new :x"
    (let [t (p/simulate (dyn/with-key dep-model (rng/fresh-key 37)) [])
          x-old (cm/get-choice (:choices t) [:x])
          y (cm/get-choice (:choices t) [:y])
          new-x (mx/scalar 0.5)
          {:keys [weight]} (p/update (dyn/with-key dep-model (rng/fresh-key 38))
                                     t (cm/choicemap :x new-x))
          expected (- (+ (lp (dist/gaussian 0 1) new-x)
                         (lp (dist/gaussian new-x 1) y))
                      (+ (lp (dist/gaussian 0 1) x-old)
                         (lp (dist/gaussian x-old 1) y)))]
      (is (h/close? expected (h/realize weight) 1e-3))))
  (testing "batched vupdate: same convention per particle"
    (let [n 4
          vt (dyn/vsimulate dep-model [] n (rng/fresh-key 47))
          new-x (mx/scalar 0.5)
          {:keys [weight]} (dyn/vupdate dep-model vt (cm/choicemap :x new-x)
                                        (rng/fresh-key 48))
          xs (cm/get-choice (:choices vt) [:x])
          ys (cm/get-choice (:choices vt) [:y])]
      (mx/eval! weight)
      (is (= [n] (mx/shape weight)) "vupdate weight is [N]-shaped")
      (doseq [i (range n)]
        (let [x-i (mx/index xs i)
              y-i (mx/index ys i)
              expected (- (+ (lp (dist/gaussian 0 1) new-x)
                             (lp (dist/gaussian new-x 1) y-i))
                          (+ (lp (dist/gaussian 0 1) x-i)
                             (lp (dist/gaussian x-i 1) y-i)))]
          (is (h/close? expected (mx/realize (mx/index weight i)) 1e-3)
              (str "particle " i " matches the scalar convention")))))))

;; ---------------------------------------------------------------------------
;; Update through a splice with a fresh child site
;; ---------------------------------------------------------------------------

(def gated-sub
  (gen [mu]
    (let [g (trace :g (dist/bernoulli 0.25))]
      (if (pos? (mx/item g))
        (trace :z (dist/gaussian mu 1))
        mu))))

(def splice-gate-model
  (gen []
    (let [x (trace :x (dist/gaussian 0 10))]
      (splice :sub gated-sub x))))

(deftest update-through-splice-fresh-and-removed
  (testing "flipping the child gate on: fresh child :z cancels"
    (let [obs (-> cm/EMPTY
                  (cm/set-choice [:x] (mx/scalar 1.0))
                  (cm/set-choice [:sub :g] (mx/scalar 0.0)))
          {t0 :trace} (p/generate (dyn/with-key splice-gate-model (rng/fresh-key 57)) [] obs)
          {:keys [trace weight]} (p/update (dyn/with-key splice-gate-model (rng/fresh-key 58))
                                           t0 (cm/set-choice cm/EMPTY [:sub :g] (mx/scalar 1.0)))
          expected (- (js/Math.log 0.25) (js/Math.log 0.75))]
      (is (some? (cm/get-choice (:choices trace) [:sub :z])) "fresh child :z sampled")
      (is (h/close? expected (h/realize weight) 1e-3))))
  (testing "flipping the child gate off: removed child :z charged via old score"
    (let [obs (-> cm/EMPTY
                  (cm/set-choice [:x] (mx/scalar 1.0))
                  (cm/set-choice [:sub :g] (mx/scalar 1.0))
                  (cm/set-choice [:sub :z] (mx/scalar 0.4)))
          {t0 :trace} (p/generate (dyn/with-key splice-gate-model (rng/fresh-key 67)) [] obs)
          {:keys [weight]} (p/update (dyn/with-key splice-gate-model (rng/fresh-key 68))
                                     t0 (cm/set-choice cm/EMPTY [:sub :g] (mx/scalar 0.0)))
          expected (- (js/Math.log 0.75)
                      (+ (js/Math.log 0.25)
                         (lp (dist/gaussian (mx/scalar 1.0) 1) (mx/scalar 0.4))))]
      (is (h/close? expected (h/realize weight) 1e-3)))))

;; ---------------------------------------------------------------------------
;; Combinators: Map, Unfold (dynamic kernels => handler fallback paths), Switch
;; ---------------------------------------------------------------------------

(def map-kernel
  (gen [mu]
    (let [g (trace :g (dist/bernoulli 0.25))]
      (if (pos? (mx/item g))
        (trace :z (dist/gaussian mu 1))
        mu))))

(deftest map-update-weight-dynamic-kernel
  (testing "Map update: element 0's gate flips on, fresh :z cancels; element 1 untouched"
    (let [mgf (comb/map-combinator (dyn/auto-key map-kernel))
          obs (-> cm/EMPTY
                  (cm/set-choice [0 :g] (mx/scalar 0.0))
                  (cm/set-choice [1 :g] (mx/scalar 0.0)))
          {t0 :trace} (p/generate mgf [[(mx/scalar 0.0) (mx/scalar 0.0)]] obs)
          {:keys [weight]} (p/update mgf t0
                                     (cm/set-choice cm/EMPTY [0 :g] (mx/scalar 1.0)))
          expected (- (js/Math.log 0.25) (js/Math.log 0.75))]
      (is (h/close? expected (h/realize weight) 1e-3)))))

(def unfold-kernel
  (gen [t state]
    (let [g (trace :g (dist/bernoulli 0.25))]
      (when (pos? (mx/item g))
        (trace :z (dist/gaussian state 1)))
      state)))

(deftest unfold-update-weight-dynamic-kernel
  (testing "Unfold update: step 0 prefix-skipped, step 1 gate flips (fresh :z cancels), step 2 retained"
    (let [ugf (comb/unfold-combinator (dyn/auto-key unfold-kernel))
          obs (-> cm/EMPTY
                  (cm/set-choice [0 :g] (mx/scalar 0.0))
                  (cm/set-choice [1 :g] (mx/scalar 0.0))
                  (cm/set-choice [2 :g] (mx/scalar 0.0)))
          {t0 :trace} (p/generate ugf [3 (mx/scalar 0.0)] obs)
          {:keys [weight]} (p/update ugf t0
                                     (cm/set-choice cm/EMPTY [1 :g] (mx/scalar 1.0)))
          expected (- (js/Math.log 0.25) (js/Math.log 0.75))]
      (is (h/close? expected (h/realize weight) 1e-3)))))

(deftest switch-branch-flip-update-weight
  (testing "Switch flip: generate weight of constrained new branch minus old branch score"
    (let [b0 (dyn/auto-key (gen [] (trace :u (dist/gaussian 0 1))))
          b1 (dyn/auto-key (gen [] (trace :v (dist/gaussian 5 1))))
          sgf (comb/switch-combinator b0 b1)
          t (p/simulate sgf [0])
          u-old (cm/get-choice (:choices t) [:u])
          flipped (assoc t :args [1])
          {:keys [trace weight]} (p/update sgf flipped
                                           (cm/choicemap :v (mx/scalar 4.5)))
          expected (- (lp (dist/gaussian 5 1) (mx/scalar 4.5))
                      (lp (dist/gaussian 0 1) u-old))]
      (is (cm/has-value? (cm/get-submap (:choices trace) :v)) "new branch active")
      (is (h/close? expected (h/realize weight) 1e-3)))))

(deftest vmap-update-weight
  (testing "vmap update: per-element thesis weights, retained :b re-scored under new :a"
    (let [kernel (gen []
                   (let [a (trace :a (dist/gaussian 0 1))]
                     (trace :b (dist/gaussian a 1))))
          vgf (vmap/vmap-gf (dyn/auto-key kernel) :axis-size 3)
          t (p/simulate vgf [])
          new-a (mx/scalar 0.5)
          {:keys [weight]} (p/update vgf t (cm/choicemap :a new-a))
          old-as (cm/get-choice (:choices t) [:a])
          bs (cm/get-choice (:choices t) [:b])
          expected (reduce + (for [i (range 3)]
                               (let [a-i (mx/index old-as i)
                                     b-i (mx/index bs i)]
                                 (- (+ (lp (dist/gaussian 0 1) new-a)
                                       (lp (dist/gaussian new-a 1) b-i))
                                    (+ (lp (dist/gaussian 0 1) a-i)
                                       (lp (dist/gaussian a-i 1) b-i))))))]
      (is (h/close? expected (h/realize weight) 1e-3)))))

;; ---------------------------------------------------------------------------
;; GFI law: update-fresh-cancellation
;; ---------------------------------------------------------------------------

(deftest update-fresh-cancellation-law
  (testing "law holds on fixed-structure and structure-changing models"
    (is (true? (:pass? (gfi/check-law :update-fresh-cancellation
                                      (dyn/auto-key dep-model) [])))
        "fixed structure: reduces to density ratio")
    (is (true? (:pass? (gfi/check-law :update-fresh-cancellation
                                      (dyn/auto-key gate-model) [])))
        "branch model: fresh project term exercised")))

(cljs.test/run-tests)
