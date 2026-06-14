(ns genmlx.world.net
  "The NETWORK/IO face of the Bun WORLD membrane (bean genmlx-gsoi) — a thin,
   honest boundary over Bun's network APIs, the SECOND effect membrane alongside
   `genmlx.mlx` (the COMPUTE membrane).

   Same architectural MOVE as mlx.cljs, different substrate GRADE:

     mlx-node : compute effect : eval! -> Metal     SYNC dispatch to a TRANSPARENT
                                                    substrate (exact; no model needed)
     Bun net  : world effect   : request/serve      ASYNC crossing to an AUTONOMOUS
                                                    OTHER (uncertain returns, can fail,
                                                    must be MODELLED — never GF'd)

   The PATTERN is shared (a thin membrane isolating one effect class, pure flow
   above); the substrate grade is NOT (mlx-node is your own values realized exactly;
   the network reaches a genuine other). The docstring states the mirror AND the
   difference on purpose — claiming net.cljs is 'just like eval!' would be a
   dishonest membrane claim.

   FACE 1 OF N, HTTP/JSON only. This is one face of the Bun membrane; the others are
   distinct (persistence = genmlx.memory via bun:sqlite; process/worker = the control
   scheduler via Worker/Bun.spawn). WebSocket / Bun.spawn-stdio / bun:ffi are FUTURE
   transports of THIS face, deliberately out of scope here. The reusable discipline
   the world faces share (so faces 2-3 don't reinvent it):
     1. ONE effect class, named and sectioned;
     2. live resources fenced as an explicit RESOURCE BOUNDARY with a blessed
        with-* scope carrying the try/finally cleanup (with-server here);
     3. pure synchronous flow BETWEEN crossings; async only AT the crossing.

   TWO effect shapes live here, honestly distinguished (this is NOT a single
   chokepoint like eval!):

   A. THE DATA CROSSING (the membrane proper) — `request`: one outbound POST + the
      parsed response, the thin mirror of eval!'s single dispatch. In GFI terms the
      OUTBOUND payload is an ACTUATOR (a sampled action committed to the world) and
      the INBOUND response is a SENSOR (the datum a model conditions on).

   B. THE SERVER LIFECYCLE — the RESOURCE BOUNDARY (`serve!`/`with-server`): a
      `Bun.serve` listener is a live, stateful, externally-observable OS resource —
      unlike eval!'s stateless dispatch. It is the one mutable boundary of this face
      (mirroring mlx.cljs's fenced 'Memory Management — the mutable boundary'). Use
      `with-server` (it guarantees teardown); `serve!` is the low-level escape hatch
      (parallel to raw eval! under tidy-run). serve! exists chiefly so tests/examples
      can stand up a localhost PEER — the agent-side FACE is the client (`request`);
      the server is scaffolding for the other side of the boundary.

   GOTCHA (load-bearing, empirically confirmed): a `Bun.serve` `fetch` handler MUST
   return a plain `Response` or a NATIVE `Promise` (`.then` chain). A promesa
   (`p/let`/`p/then`) promise is NOT `instanceof js/Promise`, so Bun does NOT await
   it — it falls through to a default HTTP 200 'Welcome to Bun' page (worse than a
   hang: the client then JSON-parses HTML and fails downstream). `serve!` therefore
   reads the request body via a native `.then` chain. Client-side promesa is fine.

   sync/async split: the network face is the ASYNC half of the Bun membrane —
   `request` returns a promise and the rollout loop is a promesa loop; the per-step
   decision math between crossings stays pure and synchronous (CLAUDE.md's
   'sync math, async events')."
  (:require [promesa.core :as p]))

(def ^:private Bun (.-Bun js/globalThis))

(defn available?
  "True when the Bun runtime network APIs backing this face are present (running
   under `bun`). Pure flow above can branch on this to skip cleanly off-Bun."
  []
  (boolean (and Bun (fn? (.-serve Bun)) (fn? (.-fetch js/globalThis)))))

;; ===========================================================================
;; A. THE DATA CROSSING — `request` (the agent's outbound face; the membrane)
;; ===========================================================================

(defn request
  "The agent's single neutral crossing OUT to an external service: POST `payload`
   (JSON) to `url`+`route`, parse the JSON response, return a promesa promise of the
   keywordized map. This is the sole DATA side effect of the network face — the thin
   mirror of mlx.cljs's eval! (one crossing), except ASYNC and to an autonomous
   OTHER. `route` is a path like \"/step\" or \"/query\"; the service shape (route
   names, payload keys) is a convention of the PEER, not baked into the membrane —
   higher layers (e.g. genmlx.agents.remote) compose `request` into their own seams.

   (Scope: POST/JSON only — the FIRST external-env transport. A heterogeneous
   third-party API, GET, or non-JSON body would compose a different crossing; that
   is a future increment, not this face's claim.)"
  ([url route] (request url route {}))
  ([url route payload]
   (-> (js/fetch (str url route)
                 #js {:method  "POST"
                      :headers #js {"content-type" "application/json"}
                      :body    (js/JSON.stringify (clj->js payload))})
       (p/then (fn [resp] (.json resp)))
       (p/then (fn [j] (js->clj j :keywordize-keys true))))))

;; ===========================================================================
;; B. SERVER LIFECYCLE — the RESOURCE BOUNDARY (scaffolding for the peer side)
;; ===========================================================================
;;
;; The ONE mutable boundary of this face: a live Bun.serve listener (a process-level
;; OS resource). It is created, used within a scope, and torn down — never escaping
;; pure flow. `with-server` is the blessed scope (try/finally teardown); `serve!` is
;; the escape hatch. A leaked listener never lets the process exit, so teardown is a
;; first-class requirement (a leak would wedge the slow-tier test cap like a hang).

(defn serve!
  "Stand up a localhost `Bun.serve` HTTP peer that routes EVERY POST to a single
   `handler` — a PURE function (fn [route payload-map] -> response-map) that touches
   no Bun and does no I/O. Returns {:url :port :server :stop}; `:stop` is a 0-arg fn
   that force-closes the live listener. `:port` 0 (default) ⇒ OS-assigned (read it
   back from `:port`, which makes concurrent/test servers collision-free). Binds
   127.0.0.1 by default.

   The handler reads the JSON body via a NATIVE `.then` chain (see the ns gotcha) and
   converts any handler exception into a 500 rather than crashing the listener.
   Prefer `with-server`, which guarantees teardown; reach for bare `serve!` only when
   you must manage the lifecycle yourself."
  ([handler] (serve! handler {}))
  ([handler {:keys [port host] :or {port 0 host "127.0.0.1"}}]
   (let [respond  (fn [m] (js/Response.json (clj->js m)))
         dispatch (fn [route body]
                    (try
                      (respond (handler route (js->clj body :keywordize-keys true)))
                      (catch :default e
                        (js/Response. (str "env error: " (.-message e))
                                      #js {:status 500}))))
         server   (.serve Bun
                    #js {:port     port
                         :hostname host
                         :fetch    (fn [req]
                                     (let [route (.-pathname (js/URL. (.-url req)))]
                                       ;; native .then (NOT promesa) — see ns gotcha.
                                       ;; the rejection arm covers an empty/absent body.
                                       (.then (.json req)
                                              (fn [b] (dispatch route b))
                                              (fn [_] (dispatch route #js {})))))})
         actual   (.-port server)]
     {:url    (str "http://" host ":" actual)
      :port   actual
      :server server
      :stop   (fn [] (.stop server true))})))

(defn with-server
  "[blessed scope] Stand up `handler`, call `(f url)` — which MUST return a promise —
   and GUARANTEE the listener is stopped afterwards, on success OR failure (the
   p/finally runs even when f's promise rejects). Returns the promise of `(f url)`'s
   result. This is the only place a server lifecycle should exist in tests/examples:
   no leaked listener (a leaked Bun.serve never exits)."
  ([handler f] (with-server handler {} f))
  ([handler opts f]
   (let [{:keys [url stop]} (serve! handler opts)]
     (-> (p/let [r (f url)] r)
         (p/finally (fn [& _] (stop)))))))
