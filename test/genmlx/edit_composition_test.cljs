;; @tier medium
(ns genmlx.edit-composition-test
  "Gate-1 of the relational resource-rational program-synthesis program
   (genmlx-uobz): CompositeEdit — sequential composition of typed edits whose
   backward-request reverses the whole chain.

   This is the substrate for COMBINATORIAL ENTAILMENT (RFT): a relation never
   applied as a single move (e.g. More∘More, or a sign-flip ∘ More) is DERIVED
   by chaining moves whose individual inverses are known, and the entailed
   reverse path — compose(bwd e_n, …, bwd e_1) — reconstructs the original
   trace. The Crel/Cfunc control program rests on this: deriving a never-seen
   composite relation from individually-seen ones is the one mechanism a cache
   provably cannot reproduce.

   PRE-REGISTERED GATE: roundtrip reconstruction of every choice value to
   <= 1e-5, with forward+backward weight cancellation to <= 1e-3, over chains of
   ConstraintEdit (install / coordination) and structure-preserving
   ArgsUpdateEdit (More/Less prior-location shift). Both constituents are
   value-lossless by the :edit-backward-request-roundtrip and
   :update-args-roundtrip GFI laws; this test proves losslessness COMPOSES.

   If this fails, the combinatorial-entailment claim has no substrate and the
   broader thesis drops to strong-analogy (the P0 STOP rule).

   NOTE: all MLX values are realized to clj numbers IMMEDIATELY on construction
   (not lazily inside `is`) — across many deftests the membrane's proactive
   buffer-count sweep (genmlx-5ucd) can free lazily-held trace buffers."
  (:require [cljs.test :refer [deftest is testing] :as t]
            [genmlx.test-helpers :as h]
            [genmlx.mlx :as mx]
            [genmlx.dist :as dist]
            [genmlx.dynamic :as dyn]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.edit :as edit])
  (:require-macros [genmlx.gen :refer [gen]]))

;; Model with an argument: x ~ N(m, 1); y ~ N(x, 1). The arg m lets
;; ArgsUpdateEdit act as a More/Less relation (shift the prior location);
;; ConstraintEdit installs values (a Same/coordination move). y depends on x,
;; so the retained-vs-transformed bookkeeping is genuinely non-trivial. This is
;; exactly the model of the :update-args-roundtrip law (gfi.cljs).
(def model
  (dyn/auto-key
   (gen [m]
     (let [x (trace :x (dist/gaussian m 1))]
       (trace :y (dist/gaussian x 1))
       x))))

(defn xv [tr] (mx/item (cm/get-choice (:choices tr) [:x])))
(defn yv [tr] (mx/item (cm/get-choice (:choices tr) [:y])))
(defn sv [tr] (mx/item (:score tr)))
;; Edit weights are MxArrays for MLX-backed GFs but a plain number for the
;; degenerate empty composite (w-add starts at 0); handle both.
(defn wv [r] (let [w (:weight r)] (if (number? w) w (mx/item w))))
(defn fresh-seed [] (p/simulate model [0.0]))

(deftest composite-equals-sequential
  (testing "CompositeEdit applies edits left-to-right; trace and summed weight match manual sequencing"
    (let [t0 (fresh-seed)
          e1 (edit/constraint-edit (cm/choicemap :x (mx/scalar 1.25)))
          e2 (edit/constraint-edit (cm/choicemap :y (mx/scalar 2.5)))
          r1 (edit/edit model t0 e1)
          r2 (edit/edit model (:trace r1) e2)
          rc (edit/edit model t0 (edit/composite-edit e1 e2))
          seq-x (xv (:trace r2)) seq-y (yv (:trace r2))
          wseq (+ (wv r1) (wv r2))
          cx (xv (:trace rc)) cy (yv (:trace rc)) cw (wv rc)
          bwd? (instance? edit/CompositeEdit (:backward-request rc))]
      (is (h/close? cx 1.25 1e-6) "composite installs x=1.25")
      (is (h/close? cy 2.5 1e-6) "composite installs y=2.5")
      (is (h/close? cx seq-x 1e-6) "composite x == sequential x")
      (is (h/close? cy seq-y 1e-6) "composite y == sequential y")
      (is (h/close? cw wseq 1e-4) "composite weight == sum of sequential weights")
      (is bwd? "backward-request is a CompositeEdit"))))

(deftest composite-roundtrip-constraints
  (testing "backward of a ConstraintEdit composite reconstructs the original trace; weights cancel"
    (let [t0 (fresh-seed)
          ox (xv t0) oy (yv t0) os (sv t0)
          e1 (edit/constraint-edit (cm/choicemap :x (mx/scalar 1.25)))
          e2 (edit/constraint-edit (cm/choicemap :y (mx/scalar 2.5)))
          fwd (edit/edit model t0 (edit/composite-edit e1 e2))
          fw (wv fwd)
          bwd (edit/edit model (:trace fwd) (:backward-request fwd))
          bx (xv (:trace bwd)) by (yv (:trace bwd)) bs (sv (:trace bwd)) bw (wv bwd)]
      (is (h/close? bx ox 1e-5) "x reconstructed to <=1e-5")
      (is (h/close? by oy 1e-5) "y reconstructed to <=1e-5")
      (is (h/close? bs os 1e-4) "score reconstructed")
      (is (h/close? (+ fw bw) 0.0 1e-3) "forward + backward weights cancel"))))

(deftest composite-args-combinatorial-entailment
  (testing "two composed arg-shifts (More∘More) equal one direct shift, and the composite reverses to the origin"
    (let [t0 (fresh-seed)
          ox (xv t0)
          e1 (edit/args-update-edit [1.0] cm/EMPTY)    ; m: 0 -> 1
          e2 (edit/args-update-edit [2.0] cm/EMPTY)    ; m: 1 -> 2
          fwd (edit/edit model t0 (edit/composite-edit e1 e2))  ; never applied as one move
          fx (xv (:trace fwd)) fargs (:args (:trace fwd)) fw (wv fwd)
          direct (edit/edit model t0 (edit/args-update-edit [2.0] cm/EMPTY))
          dw (wv direct)
          bwd (edit/edit model (:trace fwd) (:backward-request fwd))
          bx (xv (:trace bwd)) bargs (:args (:trace bwd)) bw (wv bwd)]
      ;; x is RETAINED through update-with-args, so the composite reaches the
      ;; same state as the direct More-by-2 relation: combinatorial entailment
      ;; (composing two More steps entails the larger More relation, exactly).
      (is (h/close? fx ox 1e-6) "x retained through composite")
      (is (= [2.0] fargs) "composite reaches m=2")
      (is (h/close? fw dw 1e-4) "More∘More weight == direct More-by-2 weight (telescoping entailment)")
      (is (h/close? bx ox 1e-5) "backward reconstructs x to <=1e-5")
      (is (= [0.0] bargs) "backward restores m=0")
      (is (h/close? (+ fw bw) 0.0 1e-3) "weights cancel"))))

(deftest composite-mixed-roundtrip
  (testing "mixed ConstraintEdit ∘ ArgsUpdateEdit composite reverses to the original"
    (let [t0 (fresh-seed)
          ox (xv t0) oy (yv t0)
          e1 (edit/constraint-edit (cm/choicemap :y (mx/scalar 3.0)))  ; install y
          e2 (edit/args-update-edit [1.5] cm/EMPTY)                    ; shift m
          fwd (edit/edit model t0 (edit/composite-edit e1 e2))
          fy (yv (:trace fwd)) fargs (:args (:trace fwd)) fw (wv fwd)
          bwd (edit/edit model (:trace fwd) (:backward-request fwd))
          bx (xv (:trace bwd)) by (yv (:trace bwd)) bargs (:args (:trace bwd)) bw (wv bwd)]
      (is (h/close? fy 3.0 1e-6) "forward installed y=3.0")
      (is (= [1.5] fargs) "forward reached m=1.5")
      (is (h/close? bx ox 1e-5) "x reconstructed")
      (is (h/close? by oy 1e-5) "y reconstructed to original")
      (is (= [0.0] bargs) "args restored")
      (is (h/close? (+ fw bw) 0.0 1e-3) "weights cancel"))))

(deftest composite-deep-roundtrip
  (testing "a 3-edit chain (More ∘ sign-flip-install ∘ More) reconstructs to <=1e-5"
    (let [t0 (fresh-seed)
          ox (xv t0) oy (yv t0) os (sv t0)
          e1 (edit/args-update-edit [0.7] cm/EMPTY)
          e2 (edit/constraint-edit (cm/choicemap :x (mx/scalar -1.3)))  ; Opposite-ish flip
          e3 (edit/args-update-edit [2.2] cm/EMPTY)
          fwd (edit/edit model t0 (edit/composite-edit e1 e2 e3))
          fx (xv (:trace fwd)) fw (wv fwd)
          bwd-req (:backward-request fwd)
          n-bwd (count (:edits bwd-req))
          bwd (edit/edit model (:trace fwd) bwd-req)
          bx (xv (:trace bwd)) by (yv (:trace bwd)) bargs (:args (:trace bwd)) bs (sv (:trace bwd)) bw (wv bwd)]
      (is (instance? edit/CompositeEdit bwd-req) "backward is a CompositeEdit")
      (is (= 3 n-bwd) "backward has 3 reversed edits")
      (is (h/close? fx -1.3 1e-6) "forward installed x=-1.3")
      (is (h/close? bx ox 1e-5) "x reconstructed to <=1e-5")
      (is (h/close? by oy 1e-5) "y reconstructed to <=1e-5")
      (is (= [0.0] bargs) "args restored to origin")
      (is (h/close? bs os 1e-4) "score reconstructed")
      (is (h/close? (+ fw bw) 0.0 1e-3) "weights cancel over the full chain"))))

(deftest composite-empty-is-identity
  (testing "an empty composite is the identity edit (zero weight, trace unchanged)"
    (let [t0 (fresh-seed)
          ox (xv t0)
          fwd (edit/edit model t0 (edit/composite-edit))
          fx (xv (:trace fwd)) fw (wv fwd)
          bwd? (instance? edit/CompositeEdit (:backward-request fwd))]
      (is (h/close? fx ox 1e-6) "x unchanged")
      (is (h/close? fw 0.0 1e-9) "zero weight")
      (is bwd? "backward is a CompositeEdit"))))

(t/run-tests)
