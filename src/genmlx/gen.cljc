(ns genmlx.gen
  "The gen macro for defining generative functions.

   (gen [x]
     (let [slope (trace :slope (gaussian 0 10))]
       (+ (* slope x) (trace :noise (gaussian 0 1)))))

   Inside the body:
     (trace :addr dist)       -> sample/constrain at :addr
     (splice :addr gf args)   -> call sub-generative-function
     (param :name default)    -> read trainable parameter

   The macro injects a runtime parameter (ᐩrt) as the first argument
   and binds trace, splice, param as local names from the runtime object.")

#?(:org.babashka/nbb
   (defmacro gen
     "Define a generative function from a parameter list and body.
      Returns a DynamicGF that implements the full GFI.
      The body function receives a runtime object as its first argument,
      with trace, splice, param available as local bindings."
     [params & body]
     `(genmlx.dynamic/make-gen-fn
        (fn [~'ᐩrt ~@params]
          (let [~'trace  (.-trace ~'ᐩrt)
                ~'splice (.-splice ~'ᐩrt)
                ~'param  (.-param ~'ᐩrt)]
            ~@body))
        '~(list* params body))))
