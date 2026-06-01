(ns genmlx.runtime
  "Runtime execution engine — the single mutable boundary above MLX.

   Provides run-handler which encapsulates all mutable state (a single
   volatile!) and exposes trace/splice/param operations as closures on
   a runtime object. The gen macro injects this runtime as a parameter,
   and model code accesses operations through local bindings.

   ## Architecture

   run-handler takes a pure transition function and an initial state,
   creates closures that thread state through a volatile!, and passes
   a runtime object to the body function:

       (run-handler transition init-state (fn [rt] ...body...))

   The runtime rt has three fields:
     .trace  — (fn [addr dist] -> value)
     .splice — (fn [addr gf & args] -> retval)
     .param  — (fn [name default] -> value)

   The volatile! never escapes this module."
  (:require [genmlx.handler :as h]
            [genmlx.choicemap :as cm]
            [genmlx.mlx :as mx]
            [genmlx.mlx.random :as rng]
            [genmlx.mlx.constants :refer [ZERO]]
            [genmlx.selection :as sel]
            [genmlx.protocols :as p]
            [genmlx.dist.core :as dc]))

;; Validation hook atom. No-op by default.
;; genmlx.dev/start! swaps this with a real validator.
(defonce validate-fn (atom (fn [_schema-key _value _context])))

(defn- non-empty-cm?
  "Is cm a present, non-empty choicemap?"
  [cm]
  (and cm (not= cm cm/EMPTY)))

(defn run-handler
  "Execute body-fn under a transition-based handler, returning final state map.

   transition: pure (fn [state addr dist] -> [value state'])
   init-state: initial handler state map (immutable)
   body-fn:    (fn [runtime] -> retval) — receives runtime object

   The only mutable state is a volatile! created and consumed here."
  [transition init-state body-fn]
  (@validate-fn :base-state init-state "run-handler init-state")
  (let [vol (volatile! init-state)

        ;; trace closure: sample/constrain at an address
        trace-fn
        (fn [addr dist]
          (let [[value state'] (transition @vol addr dist)]
            (vreset! vol state')
            (mx/auto-cleanup!)
            value))

        ;; param closure: read a trainable parameter
        param-fn
        (fn [name default-value]
          (let [default (if (mx/array? default-value) default-value (mx/scalar default-value))]
            (or (get-in @vol [:param-store :params name]) default)))

        ;; splice closure: call a sub-generative-function
        splice-fn
        (fn [addr gf & args]
          (let [state @vol]
            (if (:batched? state)
              ;; Batched mode
              (cond
                ;; DynamicGF: run sub-body under batched handler recursively
                (:body-fn gf)
                (let [[k1 k2] (rng/split (:key state))
                      n (:batch-size state)
                      sub-constraints (cm/get-submap (:constraints state) addr)
                      sub-old-choices (cm/get-submap (:old-choices state) addr)
                      sub-selection (when-let [s (:selection state)]
                                      (sel/get-subselection s addr))
                      sub-body-fn (:body-fn gf)
                      ;; Common init-state keys for every batched sub-execution.
                      base-init {:choices cm/EMPTY :score ZERO :key k2
                                 :batch-size n :batched? true}
                      ;; Choose transition + build init-state based on mode
                      [sub-transition sub-init-state]
                      (cond
                        sub-selection
                        [h/batched-regenerate-transition
                         (merge base-init
                                {:weight ZERO
                                 :selection sub-selection
                                 :old-choices (or sub-old-choices cm/EMPTY)})]

                        (non-empty-cm? sub-old-choices)
                        [h/batched-update-transition
                         (merge base-init
                                {:weight ZERO
                                 :constraints (or sub-constraints cm/EMPTY)
                                 :old-choices sub-old-choices
                                 :discard cm/EMPTY})]

                        (non-empty-cm? sub-constraints)
                        [h/batched-generate-transition
                         (merge base-init
                                {:weight ZERO
                                 :constraints sub-constraints})]

                        :else
                        [h/batched-simulate-transition base-init])
                      ;; Propagate param-store to sub-execution
                      sub-init-state (if-let [ps (:param-store state)]
                                       (assoc sub-init-state :param-store ps)
                                       sub-init-state)
                      ;; Recursive call to run-handler
                      sub-result (run-handler sub-transition sub-init-state
                                              (fn [rt] (apply sub-body-fn rt (vec args))))
                      ;; Merge into parent state
                      _ (@validate-fn :sub-result sub-result
                          (str "splice sub-result at " addr))
                      state' (-> state
                                 (assoc :key k1)
                                 (h/merge-sub-result addr sub-result))]
                  (vreset! vol state')
                  (:retval sub-result))

                ;; Combinator with batched fast path
                (satisfies? p/IBatchedSplice gf)
                (let [[state' retval] (p/batched-splice gf state addr (vec args))]
                  (vreset! vol state')
                  retval)

                ;; Combinator / other GFI: scalar fallback per particle
                (satisfies? p/IGenerativeFunction gf)
                (let [[state' retval] (h/combinator-batched-fallback state addr gf (vec args))]
                  (vreset! vol state')
                  retval)

                :else
                (throw (ex-info "splice of unsupported gen-fn type in batched mode"
                                {:addr addr})))

              ;; Scalar mode: delegate to executor
              (let [[sk ek] (rng/split (:key state))
                    _ (vreset! vol (assoc state :key ek))
                    sub-constraints (cm/get-submap (:constraints state) addr)
                    sub-old-choices (cm/get-submap (:old-choices state) addr)
                    sub-selection (when-let [s (:selection state)]
                                    (sel/get-subselection s addr))
                    old-splice-score (get (:old-splice-scores state) addr)
                    old-sub-meta (get (:old-nested-splice-scores state) addr)
                    sub-result ((:executor state) gf (vec args)
                                                  {:constraints sub-constraints
                                                   :old-choices sub-old-choices
                                                   :selection sub-selection
                                                   :key sk
                                                   :old-splice-score old-splice-score
                                                   :old-sub-splice-scores (:splice-scores old-sub-meta)
                                                   :old-sub-nested-splice-scores (:nested-splice-scores old-sub-meta)
                                                   :param-store (:param-store state)})]
                (@validate-fn :sub-result sub-result
                  (str "executor sub-result at " addr))
                (vswap! vol h/merge-sub-result addr sub-result)
                (:retval sub-result)))))

        ;; Build runtime object
        rt #js {:trace trace-fn :splice splice-fn :param param-fn}]

    (let [retval (body-fn rt)]
      (assoc @vol :retval retval))))
