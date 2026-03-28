(ns genmlx.mlx.bootstrap
  "Bootstrap jax-js with WebGPU backend.
   In browser: navigator.gpu is native, no polyfill needed.
   In Node.js/Bun: polyfill via Dawn (npm: webgpu).")

;; Polyfill navigator.gpu if not present (terminal via Dawn)
(when-not (exists? js/navigator.gpu)
  (let [dawn (js/require "webgpu")
        gpu  (.create dawn #js [])]
    (js/Object.assign js/globalThis (.-globals dawn))
    (js/Object.defineProperty js/globalThis "navigator"
      #js {:value #js {:gpu gpu} :writable true :configurable true})))

;; Load and initialize jax-js
(defonce ^:private jax-promise (js/import "@jax-js/jax"))

(defonce ^:private state (atom {:ready? false :jax nil}))

(defn init!
  "Initialize jax-js. Returns a Promise that resolves when ready.
   Must be called before any mx/ operations."
  []
  (-> jax-promise
      (.then (fn [jax]
               (-> (.init jax)
                   (.then (fn [devices]
                            (when (.includes devices "webgpu")
                              (.defaultDevice jax "webgpu"))
                            (reset! state {:ready? true :jax jax})
                            jax)))))))

(defn jax
  "Returns the jax-js module. Throws if not initialized."
  []
  (let [s @state]
    (when-not (:ready? s)
      (throw (js/Error. "genmlx.mlx.bootstrap: jax-js not initialized. Call (init!) first.")))
    (:jax s)))
