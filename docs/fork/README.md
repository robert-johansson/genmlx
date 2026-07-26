# The fork patch tree — how GenMLX stays synced with mlx-node and MLX

> **Bean:** `genmlx-rjfm`. **Date:** 2026-07-25.
> **Provenance:** 13-agent investigation over the three trees, with every headline number
> re-verified by hand against the live repos. Measurements below are reproducible with the
> commands shown; where something was *not* verifiable without a build, it says so.
> **Companions:** [`SYNC-RUNBOOK.md`](SYNC-RUNBOOK.md) (the ritual),
> [`CONFLICT-LEDGER.md`](CONFLICT-LEDGER.md) (per-file resolution doctrine).

## The three repos

```
ml-explore/mlx ──────────────► mlx-node/mlx ──────────► mlx-node/mlx-node ──────► genmlx
  true upstream                pure MIRROR of              the Node/Rust           consumer
                               ml-explore + a side         binding layer
                               branch of NAX patches
        │                              │                          │
        └──► robert-johansson/mlx ◄────┘                          └──► robert-johansson/mlx-node
                    OUR FORK                                                  OUR FORK
```

`genmlx` pins `mlx-node` as a git submodule; `mlx-node` pins `mlx` as a git submodule. Our
mlx-node fork rewrites one line of `.gitmodules` so `crates/mlx-sys/mlx` resolves to
*our* MLX fork rather than mlx-node's.

**`mlx-node/mlx` main is a byte-identical mirror of ml-explore** — zero own commits. Their own
MLX patches (Apple NAX D256 SDPA) live on the side branch `perf/qwen-d256-sdpa`, and that is
what their releases pin. There is no hidden third patch layer.

## The measured asymmetry — and why the two forks use opposite strategies

Measured 2026-07-25 against mlx-node `v0.0.8` (`0ebeaa57`) and ml-explore `973e27f82`:

| | **mlx** (C++) | **mlx-node** (Rust/TS) |
|---|---|---|
| our delta | 15 commits, 11 files, +644/−78 | 78 commits, 110 files, +12 489/−856 |
| files upstream *also* touched | **0 of 11** | **42 of 110** |
| rebase cost | **clean, 15/15** | 9 stops, 38 files, **161 hunks** |
| merge cost | 0 conflicts | 1 stop, 14 files, **26 hunks** |
| **strategy** | **REBASE** | **MERGE-FORWARD** |

Two conclusions follow, and they are the whole design:

**The cost metric is both-touched lines, not total lines.** Our 68 mlx-node files that upstream
never touches carry *more* code (+7 266) than the 42 shared ones (+5 223) and produced **zero**
conflicts in both the merge and the rebase experiment. Rank every shrink decision by
both-touched lines removed, not by lines removed.

**Rebasing mlx-node costs 6.7× what merging costs**, because replaying 78 commits re-litigates
the *previous* sync that merge `72e752e9` already resolved by hand. The proof:
`packages/server/src/presets.ts` conflicts under rebase even though no upstream commit in the
range touches it. Under merge-forward the merge-base advances to upstream's tip at every
release, so each sync only ever resolves genuinely-new overlap — that is the mechanism that
keeps steady state cheap.

## Branch layout

### `robert-johansson/mlx` — rebase

| branch | contents |
|---|---|
| `mirror/ml-explore` | mirror of `ml-explore/main`. Zero own commits, ever. |
| `mirror/nax` | mirror of `mlx-node/mlx` `perf/qwen-d256-sdpa`. Force-updated, **not** FF-only — mlx-node rebases this branch. |
| `nax-on-ml-explore` | mlx-node's 4 commits replayed onto `mirror/ml-explore`. Rebased replay; rewriting expected. |
| `thor/stack` | our 15 patches on `mirror/nax` — conservative, identical base to what mlx-node tests against. |
| `thor/stack-mlx-latest` | our 15 patches on `nax-on-ml-explore` — current with true upstream. **Pin candidate.** |
| `pin/mlx/<date>` | annotated tag. **The gitlink points here, never at a branch tip.** |
| `pr/*` | one topic per branch off `mirror/ml-explore`. Rewriting allowed; never pinned. |
| `archive/pre-resync-<date>` | frozen former tip. Never deleted. |

### `robert-johansson/mlx-node` — merge-forward

| branch | contents |
|---|---|
| `mirror/upstream` | FF-only mirror of `mlx-node/main`. |
| `genmlx/integration` | **never rewound.** One `merge --no-ff` per upstream release. The pinned line. |
| `sync/up-vNEXT` | per-release scratch. Merge, resolve, and build here; land `--ff-only` when green. Abortable without touching the pinned line. |
| `pin/mlx-node/<date>` | annotated tag; genmlx's gitlink points here. |
| `pr/*` | topic branches for upstreaming. |
| `archive/pre-resync-<date>` | frozen former tip. |

### `genmlx` — consumer

`docs/fork/` (this directory) and `sync/mlx-node-vNEXT` branches for gitlink bumps.

## Two laws, both derived from measured failures

**Push depth-first, pin breadth-last.** mlx branch+tag pushed → mlx-node pin commit →
mlx-node pushed+tagged → genmlx gitlink. Never pin an unpushed SHA; never pin a branch tip.

The failure this prevents is precise, and narrower than it first appears. GitHub serves
submodule commits **by SHA** even when no branch reaches them — 6 of upstream mlx-node's 14
historical mlx pins are reachable from no branch on `mlx-node/mlx`, yet
`git fetch mlxnode <sha>` retrieves all six, so their recursive clones still work. Branch
reachability is therefore *not* the invariant.

**The invariant is that the SHA exists on a remote at all.** A commit that has only ever lived
in a local clone is unfetchable, and that is exactly the state this repo was in on 2026-07-25:
the pinned mlx SHA `49787ee19` existed nowhere but one working copy. Gate on fetchability:

```bash
sha=$(git -C <parent> rev-parse HEAD:<submodule-path>)
git -C <submodule> fetch origin "$sha" || echo "FAIL: pinned commit is not on the remote"
```

Cutting an immortal `pin/*` tag is still worth doing — it makes the pin *findable* by a human,
survives a `pr/*` branch rewrite, and does not depend on the host allowing fetch-by-SHA.

**Never `git checkout --ours/--theirs` a gitlink.** On a `160000` tree entry that silently runs
`git rm`, deleting the submodule *and* its `.gitmodules` stanza. In the rebase experiment it
fabricated 10 phantom downstream conflicts and turned 9 stops into 19. The only correct form is:

```bash
git update-index --cacheinfo "160000,$MLXSHA,crates/mlx-sys/mlx"
```

## What is *not* the problem

**`.gitmodules` is latent, not active.** Upstream has touched it exactly twice in the repo's
history and **zero times** since our merge-base (`git log 5602f12d..up/main -- .gitmodules` is
empty). It did not appear in the 14-file conflict set. The **gitlink** is the real recurring
item — it conflicted once in the merge and twice in the rebase, and no merge driver, no `-X`
flag, and no rerere can ever resolve a `160000` entry.

A `merge=forkurl` driver for `.gitmodules` is a nice-to-have, not a priority. The endgame is
better: land the MLX patches upstream (see below), at which point the URL reverts to upstream's
and the divergence is deleted rather than managed.

## Relationship to the earlier fork-to-zero docs

`docs/fork-to-zero-plan.md`, `docs/mlx-node-fork-minimization-plan.md`,
`docs/fork-to-zero-h1-milestone.md` and `docs/mlx-node-integration-roadmap.md` (all June 2026)
set the *direction*: shrink the mlx-node fork, keep the PPL-specific native surface in a
GenMLX-owned crate. That direction stands.

**What this document supersedes is their cost model.** Those plans rank shrink targets by
footprint inside the mlx-node tree. The 2026-07-25 measurement shows footprint and merge cost
are almost uncorrelated:

- **`crates/genmlx-core` + `packages/genmlx-core`** (2 219 lines, the most "liftable-looking"
  thing in the tree, and Horizon 1's headline target) contributes **+0 both-touched lines** and
  produced **0 conflicts** in both experiments. Upstream has never touched it. Relocating it
  buys *zero* merge-cost reduction. It is also currently blocked: `crates/mlx-sys/Cargo.toml`
  has no `links` key, so an out-of-tree crate cannot receive its include/lib metadata without
  re-deriving MLX's paths or double-building MLX.
- **`packages/agent/src/provider/genmlx/**`** — barely mentioned in the June plans, because it
  did not exist yet — carries **all four severe conflicts** and ~9 both-touched files.

So: the shrink program is still right, but the priority order inverts. Extract the pi provider
first; defer the addon relocation.

## The shrink program, ranked by both-touched lines removed

| move | effect | verdict |
|---|---|---|
| Extract the pi provider to a genmlx-side package via pi's public `registerProvider` seam | removes ~9 both-touched files, all 4 severe conflicts | **do it** (10–16 h) |
| Delete our 119 hand-edited lines in `packages/core/index.d.cts` | removes the worst silent-loss landmine; it is a **generated** file upstream regenerates wholesale at `preversion` | **do it** (free) |
| Upstream 8 MLX + 7 mlx-node patch series | shrinks the fork permanently | **do it** (rolling) |
| Relocate `crates/genmlx-core` out of tree | +0 both-touched lines, and blocked on the missing `links` key | **defer** |
| Amputate the native branchable-KV surface | **rejected** — load-bearing for the 80B (`src/genmlx/llm/branched.cljs`, `smc.cljs`, `token_smc_real_test.cljs`); qwen3_next has no owned forward | **keep** |
| Park the flat-VLM cores | **rejected** — `vlmPrefillFlat` itself is test-only, but the flat vision *cores* are what make `.chatSessionStart` image turns work on CUDA at all (`src/genmlx/llm/vision.cljs:119`) | **keep** |

Add `"private": true` to `packages/genmlx-core/package.json` regardless — one line, zero risk.
Upstream's publish job runs `yarn workspaces foreach -Rt --no-private npm publish`, and the
package is currently `version: 0.0.0` with no `private` flag, i.e. not excluded.

## Effort

| | first sync | steady state |
|---|---|---|
| mlx | 2–3 h (measured clean) | ~0.25 h attention, 3–6 h wall |
| mlx-node | 39–64 h total | 4–10 h/release *with* the shrink program; 12–20 h *without* |

Roughly 20–30 h of the first-sync figure is human attention; the rest is Thor build and test
wall-clock under the one-GPU-process rule (see `docs/thor-gpu-discipline.md`).

Upstream's cadence is bimodal and measured: 8 tags in 10 months — six within five days in
March 2026, then a **95-day gap** before v0.0.8. That gap is what produced today's bill.
**Adopt a ceiling: never let `up/main` run more than ~60 days or ~150 commits ahead unsynced.**

## Standing audit — drop what upstream has taken

Every sync, in both forks:

```bash
git cherry -v <upstream-mirror> <our-stack>   # a leading '-' means upstream took that patch
```

A `-` line means the patch is now redundant: delete it from the stack rather than carrying it.
As of 2026-07-25 this returns nothing in either repo — nothing of ours has landed upstream yet.
