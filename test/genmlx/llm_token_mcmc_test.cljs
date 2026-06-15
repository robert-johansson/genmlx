;; @tier fast
(ns genmlx.llm-token-mcmc-test
  "genmlx-3ob2 / fayo: full 7-op GFI over token traces validated by TV-to-exact
   on a TINY synthetic categorical 'token' model (MODEL-FREE — no LLM/GPU). The
   grammar (ab|ac|ba|bc) is COUPLED: t1's valid set depends on t0, so a
   regenerate move over t0 with t1 retained must rescore the grammar-coupled
   retained site — the C8 case. The fast per-site weight would be wrong; the
   forced general retained-only path is exact, which MCMC-vs-exact pins."
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.dynamic :as dyn]
            [genmlx.mlx.random :as rng]
            [genmlx.inference.mcmc :as mcmc]
            [genmlx.edit :as edit]
            [genmlx.llm.grammar :as grammar])
  (:require-macros [genmlx.gen :refer [gen]]))

;; tokens: a=0 b=1 c=2.  per-site logits (same at every site; grammar masks).
(def L [0.4 0.1 -0.2])
(def logits (mx/array L))
(def token-index ["a" "b" "c"])
(def A 0) (def B 1) (def C 2)

;; coupled grammar: t0 in {a,b}; t1 in {b,c} after a, {a,c} after b.
(def constraint
  {:dfa (grammar/compile-regex "(ab|ac|ba|bc)")
   :token-index token-index :eos-id 999 :masks nil})

(def base-model (gen [] (trace :t0 (dist/categorical logits))
                        (trace :t1 (dist/categorical logits)) nil))
(def cmodel (grammar/constrain base-model constraint))
;; Keyed alias: project/update/edit run ensure-key even when they don't sample
;; (deterministic replay), so they need a key on the gf; a fixed one keeps them
;; reproducible.
(def kmodel (dyn/with-key cmodel (rng/fresh-key 7)))

;; ---- exact distribution over the 4 valid sequences -------------------------
(defn- masked-softmax [allowed tok]
  (let [z (reduce + (map #(js/Math.exp (nth L %)) allowed))]
    (/ (js/Math.exp (nth L tok)) z)))
;; t0 ∈ {a,b}; t1 ∈ {b,c} (after a) or {a,c} (after b)
(def p-t0 {A (masked-softmax [A B] A) B (masked-softmax [A B] B)})
(def exact
  {[A B] (* (p-t0 A) (masked-softmax [B C] B))
   [A C] (* (p-t0 A) (masked-softmax [B C] C))
   [B A] (* (p-t0 B) (masked-softmax [A C] A))
   [B C] (* (p-t0 B) (masked-softmax [A C] C))})

(defn- tv [p q]
  (* 0.5 (reduce + (map (fn [k] (js/Math.abs (- (get p k 0.0) (get q k 0.0))))
                        (into (set (keys p)) (keys q))))))

(defn- pair [trace]
  [(mx/item (cm/get-value (cm/get-submap (:choices trace) :t0)))
   (mx/item (cm/get-value (cm/get-submap (:choices trace) :t1)))])

(defn- empirical [pairs]
  (let [n (count pairs)]
    (into {} (map (fn [[k v]] [k (/ v n)])) (frequencies pairs))))

;; ---- exact per-site masked log-probs (the INDEPENDENT oracle: closed form,
;;      no function-under-test) -------------------------------------------------
(defn- allowed-t1 [t0] (if (= t0 A) [B C] [A C]))      ; grammar: after a→{b,c}, after b→{a,c}
(defn- lp0 [t0]    (js/Math.log (masked-softmax [A B] t0)))
(defn- lp1 [t0 t1] (js/Math.log (masked-softmax (allowed-t1 t0) t1)))
(defn- choice [cmap addr] (mx/item (cm/get-value (cm/get-submap cmap addr))))

;; A deterministic constrained trace with the given (t0,t1): fully-constrained
;; generate is an exact draw. Reused by the update/project/edit op tests.
(defn- known-trace [t0 t1]
  (:trace (p/generate kmodel []
                      (cm/from-map {:t0 (mx/array t0) :t1 (mx/array t1)}))))

;; ---------------------------------------------------------------------------
(deftest simulate-matches-exact
  (testing "constrained simulate distribution == exact (per-step masked product)"
    (let [traces (mapv (fn [i] (p/simulate (dyn/with-key cmodel (rng/fresh-key (+ 1000 i))) []))
                       (range 3000))
          emp (empirical (map pair traces))]
      (println "  exact:" (pr-str (into {} (map (fn [[k v]] [k (.toFixed v 3)]) exact))))
      (println "  emp:  " (pr-str (into {} (map (fn [[k v]] [k (.toFixed v 3)]) emp))))
      (is (every? #{[A B] [A C] [B A] [B C]} (keys emp)) "only valid sequences appear")
      (is (< (tv exact emp) 0.04) (str "TV(simulate, exact) = " (tv exact emp))))))

(deftest assess-and-generate-weights
  (testing "assess weight == exact joint log-prob for each valid sequence"
    (doseq [[[t0 t1] pe] exact]
      (let [w (mx/item (:weight (p/assess (dyn/with-key cmodel (rng/fresh-key 1)) []
                                          (cm/from-map {:t0 (mx/array t0) :t1 (mx/array t1)}))))]
        (is (< (js/Math.abs (- w (js/Math.log pe))) 1e-4)
            (str "assess " [t0 t1] " w=" w " expected=" (js/Math.log pe))))))
  (testing "fully-constrained generate weight == full joint log-prob"
    (let [{:keys [trace weight]} (p/generate (dyn/with-key cmodel (rng/fresh-key 7)) []
                                             (cm/from-map {:t0 (mx/array A) :t1 (mx/array B)}))]
      (is (< (js/Math.abs (- (mx/item weight) (js/Math.log (exact [A B])))) 1e-4)
          "generate weight = log p(ab)"))))

(deftest mcmc-over-coupled-site-matches-exact-posterior
  (testing "C8: regenerate t0 with t1 retained rescores the grammar-coupled site"
    ;; observe t1 = c (in both ac and bc); infer t0.
    (let [post-unnorm {A (exact [A C]) B (exact [B C])}
          z (+ (post-unnorm A) (post-unnorm B))
          exact-post {A (/ (post-unnorm A) z) B (/ (post-unnorm B) z)}
          traces (mcmc/mh {:samples 4000 :burn 500 :selection (sel/select :t0)
                           :key (rng/fresh-key 123)}
                          cmodel [] (cm/from-map {:t1 (mx/array C)}))
          t0s (map (fn [t] (mx/item (cm/get-value (cm/get-submap (:choices t) :t0)))) traces)
          emp (let [n (count t0s)] (into {} (map (fn [[k v]] [k (/ v n)]) (frequencies t0s))))]
      (println "  exact-post p(t0|t1=c):" (pr-str (into {} (map (fn [[k v]] [k (.toFixed v 3)]) exact-post))))
      (println "  mcmc emp:             " (pr-str (into {} (map (fn [[k v]] [k (.toFixed v 3)]) emp))))
      (is (every? #{A B} (keys emp)) "t1=c retained; t0 stays in {a,b}")
      (is (< (js/Math.abs (- (get emp A 0.0) (exact-post A))) 0.05)
          (str "MCMC p(t0=a|t1=c)=" (get emp A 0.0) " vs exact " (exact-post A))))))

;; ===========================================================================
;; B.1 (fayo): the remaining GFI ops over token traces — project / update /
;; propose / edit — each pinned to the SAME independent masked-softmax oracle
;; (the function under test never appears in the ground truth). With the
;; simulate/assess/generate/regenerate tests above, this completes the
;; "LLM-as-GF validated across ALL GFI operations over token traces" claim.
;; ===========================================================================

(deftest project-selected-densities-match-exact
  (testing "project(selection) == Σ selected masked log-probs; additive to score"
    (doseq [[t0 t1] [[A B] [A C] [B A] [B C]]]
      (let [tr   (known-trace t0 t1)
            p0   (mx/item (p/project kmodel tr (sel/select :t0)))
            p1   (mx/item (p/project kmodel tr (sel/select :t1)))
            pall (mx/item (p/project kmodel tr sel/all))]
        (is (< (js/Math.abs (- p0 (lp0 t0))) 1e-4)    (str "project :t0 " [t0 t1] " = " p0))
        (is (< (js/Math.abs (- p1 (lp1 t0 t1))) 1e-4) (str "project :t1 " [t0 t1] " = " p1))
        (is (< (js/Math.abs (- pall (+ (lp0 t0) (lp1 t0 t1)))) 1e-4) "project all == full score")
        (is (< (js/Math.abs (- pall (+ p0 p1))) 1e-4) "additive: all == :t0 + :t1")))))

(deftest update-changed-site-weight-matches-exact
  (testing "update t1 b->c (t0=a retained): weight == Δ(t1 term); discard = old t1"
    (let [tr (known-trace A B)
          {:keys [trace weight discard]} (p/update kmodel tr (cm/from-map {:t1 (mx/array C)}))
          exp (- (lp1 A C) (lp1 A B))]
      (is (= [(choice (:choices trace) :t0) (choice (:choices trace) :t1)] [A C]) "t1->c, t0 retained")
      (is (< (js/Math.abs (- (mx/item weight) exp)) 1e-4)
          (str "update weight " (mx/item weight) " vs exact " exp))
      (is (= B (choice discard :t1)) "discard carries old t1=b"))))

(deftest propose-weight-equals-joint-and-matches-exact
  (testing "propose: each weight == masked joint log-prob; empirical == exact"
    (let [props (mapv (fn [i] (p/propose (dyn/with-key cmodel (rng/fresh-key (+ 5000 i))) []))
                      (range 3000))
          bad   (filter (fn [{:keys [choices weight]}]
                          (let [t0 (choice choices :t0) t1 (choice choices :t1)]
                            (>= (js/Math.abs (- (mx/item weight) (+ (lp0 t0) (lp1 t0 t1)))) 1e-4)))
                        props)
          emp   (empirical (map (fn [{:keys [choices]}] [(choice choices :t0) (choice choices :t1)]) props))]
      (println "  propose emp:" (pr-str (into {} (map (fn [[k v]] [k (.toFixed v 3)]) emp))))
      (is (zero? (count bad)) (str (count bad) "/3000 propose weights != joint log-prob"))
      (is (every? #{[A B] [A C] [B A] [B C]} (keys emp)) "only valid sequences proposed")
      (is (< (tv exact emp) 0.04) (str "TV(propose, exact) = " (tv exact emp))))))

(deftest edit-constraint-is-update-and-reverses
  (testing "ConstraintEdit t1 b->c == update; backward-request restores the trace"
    (let [tr (known-trace A B)
          {:keys [trace weight discard backward-request]}
          (edit/edit kmodel tr (edit/constraint-edit (cm/from-map {:t1 (mx/array C)})))
          exp (- (lp1 A C) (lp1 A B))]
      (is (= [(choice (:choices trace) :t0) (choice (:choices trace) :t1)] [A C]) "forward edit sets t1=c")
      (is (< (js/Math.abs (- (mx/item weight) exp)) 1e-4) "edit weight == update weight")
      (is (= B (choice discard :t1)) "discard old t1=b")
      (is (instance? edit/ConstraintEdit backward-request) "backward-request is a ConstraintEdit")
      ;; Reversibility (the SMCP3 contract): backward edit restores (a,b), weights cancel.
      (let [{bt :trace bw :weight} (edit/edit kmodel trace backward-request)]
        (is (= [(choice (:choices bt) :t0) (choice (:choices bt) :t1)] [A B]) "backward restores (a,b)")
        (is (< (js/Math.abs (+ (mx/item weight) (mx/item bw))) 1e-4) "round-trip weights cancel")))))

(cljs.test/run-tests)
