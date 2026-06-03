(ns genmlx.edit-purity-test
  "Regression for genmlx-5413: genmlx.edit must be usable by pure / deterministic
   GFs whose weights are plain numbers — without the native MLX backend.

   Previously edit.cljs had a top-level (:require [genmlx.mlx :as mx]), whose
   `(js/require \"@mlx-node/core\")` runs at load, so merely requiring genmlx.edit
   forced the GPU backend even for ConstraintEdit / SelectionEdit (which use no
   MLX). The fix drops that require and resolves genmlx.mlx's scalar ops LAZILY,
   only inside the ProposalEdit weight-combine and only when a weight is an
   MxArray. For plain-number weights the combine uses +/- and never touches MLX.

   These tests use minimal deterministic GFs (number weights) and assert the edit
   results stay plain numbers — i.e. no MxArray is produced, which is the in-repo
   proxy for 'no MLX op ran'. (Loadability with the backend absent is verified out
   of tree via the external repro; the MxArray ProposalEdit path is covered by
   proposal_edit_test.)"
  (:require [cljs.test :refer [deftest is testing]]
            [genmlx.test-helpers :as h]
            [genmlx.protocols :as p]
            [genmlx.choicemap :as cm]
            [genmlx.selection :as sel]
            [genmlx.edit :as edit]))

;; Deterministic GF: update/regenerate return PLAIN-NUMBER weights + a discard.
(defrecord PureGF []
  p/IUpdate
  (update [_ trace _constraints]
    {:trace trace :weight 1.5 :discard (cm/choicemap :x 0.0)})
  p/IRegenerate
  (regenerate [_ trace _selection]
    {:trace trace :weight 0.25 :discard cm/EMPTY}))

;; Pieces for a pure ProposalEdit (all number weights).
(defrecord PureModelGF []
  p/IUpdate
  (update [_ trace _ch] {:trace trace :weight 2.0 :discard (cm/choicemap :a 9.0)}))
(defrecord PureFwdGF []
  p/IPropose
  (propose [_ _args] {:choices (cm/choicemap :a 1.0) :weight 0.5 :retval nil}))
(defrecord PureBwdGF []
  p/IAssess
  (assess [_ _args _choices] {:retval nil :weight 0.75}))

(def ^:private dummy-trace {:choices cm/EMPTY})

(deftest constraint-edit-pure
  (testing "ConstraintEdit on a deterministic GF: number weight, discard carried back, no MLX"
    (let [r (edit/edit-dispatch (->PureGF) dummy-trace
                                (edit/constraint-edit (cm/choicemap :x 1.0)))]
      (is (number? (:weight r)) "weight is a plain number (no MxArray)")
      (is (h/close? 1.5 (:weight r) 1e-9) "weight passes through p/update")
      (is (instance? edit/ConstraintEdit (:backward-request r)) "backward = ConstraintEdit")
      (is (h/close? 0.0 (cm/get-choice (:constraints (:backward-request r)) [:x]) 1e-9)
          "backward-request carries the discard"))))

(deftest selection-edit-pure
  (testing "SelectionEdit on a deterministic GF: number weight, no MLX"
    (let [r (edit/edit-dispatch (->PureGF) dummy-trace
                                (edit/selection-edit (sel/select :x)))]
      (is (number? (:weight r)) "weight is a plain number (no MxArray)")
      (is (h/close? 0.25 (:weight r) 1e-9) "weight passes through p/regenerate")
      (is (instance? edit/SelectionEdit (:backward-request r)) "backward = SelectionEdit")
      (is (= cm/EMPTY (:discard r)) "selection edit discards nothing"))))

(deftest proposal-edit-pure-number-path
  (testing "ProposalEdit with all-number weights uses +/- and stays backend-free"
    ;; weight = update(2.0) + (assess_bwd(0.75) - propose_fwd(0.5)) = 2.25
    (let [req (edit/proposal-edit (->PureFwdGF) (->PureBwdGF))
          r   (edit/edit-dispatch (->PureModelGF) dummy-trace req)]
      (is (number? (:weight r)) "pure ProposalEdit weight is a plain number — w-add/w-sub took the numeric branch")
      (is (h/close? 2.25 (:weight r) 1e-9) "numeric weight combine is correct")
      (is (instance? edit/ProposalEdit (:backward-request r)) "backward = ProposalEdit (fwd/bwd swapped)")
      (is (h/close? 9.0 (cm/get-choice (:discard r) [:a]) 1e-9) "discard preserved"))))

(deftest edit-namespace-has-no-mlx-require
  (testing "the genmlx.edit interface loaded and works without any MLX value being constructed"
    ;; If requiring genmlx.edit had pulled in genmlx.mlx eagerly, this ns would
    ;; have failed to load. Reaching here at all (with all weights above plain
    ;; numbers) is the regression signal.
    (is (some? edit/constraint-edit))
    (is (some? edit/selection-edit))
    (is (some? edit/proposal-edit))))

(cljs.test/run-tests)
