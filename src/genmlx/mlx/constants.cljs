(ns genmlx.mlx.constants
  "Shared cached MLX scalar constants and log constants.

   Defined once here and referenced everywhere, instead of re-allocating the
   same (mx/scalar 0.0) etc. in each consumer. MLX arrays are immutable —
   mx/add and friends always create new arrays — so a single cached scalar is
   safe to share across namespaces.

   This is a leaf namespace: it requires only genmlx.mlx, which every consumer
   already reaches, so it introduces no circular dependency."
  (:require [genmlx.mlx :as mx]))

;; Plain numeric log constant (host-side, for arithmetic before mx/scalar).
(def LOG-2PI (js/Math.log (* 2.0 js/Math.PI)))

;; Cached MLX scalar constants.
(def ZERO (mx/scalar 0.0))
(def ONE (mx/scalar 1.0))
(def TWO (mx/scalar 2.0))
(def THREE (mx/scalar 3.0))
(def HALF (mx/scalar 0.5))
(def NEG-INF (mx/scalar ##-Inf))
(def LOG-2 (mx/scalar (js/Math.log 2.0)))
(def LOG-2PI-HALF (mx/scalar (* 0.5 LOG-2PI)))
(def LOG-PI (mx/scalar (js/Math.log js/Math.PI)))
(def MLX-PI (mx/scalar js/Math.PI))
(def SQRT-TWO (mx/scalar (js/Math.sqrt 2.0)))
(def TINY (mx/scalar 1e-30))
