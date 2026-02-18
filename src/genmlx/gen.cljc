(ns genmlx.gen
  "The gen macro for defining generative functions.

   (gen [x]
     (let [slope (trace :slope (gaussian 0 10))]
       (+ (* slope x) (trace :noise (gaussian 0 1)))))

   Inside the body:
     (trace :addr dist)       -> sample/constrain at :addr
     (splice :addr gf args)   -> call sub-generative-function")

#?(:org.babashka/nbb
   (defmacro gen
     "Define a generative function from a parameter list and body.
      Returns a DynamicGF that implements the full GFI."
     [params & body]
     `(genmlx.dynamic/make-gen-fn
        (fn ~params ~@body)
        '~(list* params body))))
