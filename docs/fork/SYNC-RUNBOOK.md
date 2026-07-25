# Sync runbook — "upstream cut a new release"

> **Bean:** `genmlx-rjfm`. **Date:** 2026-07-25.
> **Read first:** [`README.md`](README.md) (why the two forks use opposite strategies) and
> [`CONFLICT-LEDGER.md`](CONFLICT-LEDGER.md) (what to do with each conflicted file).
> Every test path and command below was verified to exist against the live tree on 2026-07-25.

Paths used throughout:

```bash
GENMLX=/home/robert/code/mlx/genmlx
MN=$GENMLX/mlx-node
MLX=$MN/crates/mlx-sys/mlx
```

**The two laws** (see README for the measured failures behind them):
1. **Push depth-first, pin breadth-last.** Never pin an unpushed SHA. Never pin a branch tip.
2. **Never `git checkout --ours/--theirs` a gitlink.** Use `update-index --cacheinfo`.

---

## 0. Preflight — read-only, ~30 min, no build

```bash
git -C $MN  fetch up --tags
git -C $MLX fetch up mlxnode origin --tags

# What are we taking on?
git -C $MN diff --shortstat HEAD..up/main
git -C $MN log --oneline HEAD..up/main

# THE cost metric: files BOTH sides touch. Everything else is free.
MB=$(git -C $MN merge-base HEAD up/main)
comm -12 <(LC_ALL=C git -C $MN diff --name-only $MB..HEAD    | LC_ALL=C sort) \
         <(LC_ALL=C git -C $MN diff --name-only $MB..up/main | LC_ALL=C sort)

# Predicted conflicts, without touching a working tree:
git -C $MN merge-tree --write-tree HEAD up/main | grep '^CONFLICT'

# The .gitmodules assertion — must be exactly our one-line URL rewrite, nothing else:
git -C $MN diff up/main...HEAD -- .gitmodules

# Standing audit: has upstream taken any of our patches? A leading '-' means DROP it.
git -C $MLX cherry -v mirror/ml-explore thor/stack-mlx-latest | grep '^-' || echo "none taken yet"
git -C $MN  cherry -v mirror/upstream   genmlx/integration    | grep '^-' || echo "none taken yet"
```

**Gate:** if the both-touched list is empty, the sync is mechanical — skip straight to §3.
If it contains `packages/agent/**` or `qwen3_5*/model.rs`, budget the full battery in §4.

Optionally enable `rerere` before a sync you expect to retry:

```bash
git -C $MN config rerere.enabled true
git -C $MN config rerere.autoupdate false   # keep FALSE on pass 1 — see "rerere" below
```

---

## 1. MLX first — deepest repo first, always

Measured 2026-07-25: our 15 patches replay **clean on every candidate base**; upstream touched
**0 of our 11 files** across 37 new ml-explore commits. Expect this step to be uneventful.

```bash
# Refresh the mirrors. mirror/nax is force-updated, NOT fast-forward — mlx-node rebases it.
git -C $MLX branch -f mirror/ml-explore up/main
git -C $MLX branch -f mirror/nax        mlxnode/perf/qwen-d256-sdpa

# Replay mlx-node's own MLX commits onto current ml-explore.
#   NOTE THE CARET. Their base commit de7e34290 is mlx-node-authored, not an ml-explore
#   commit; using it as the rebase base SILENTLY DROPS IT. <their-base>^ keeps all of theirs.
git -C $MLX rebase --onto mirror/ml-explore <their-base>^ mirror/nax
git -C $MLX branch -f nax-on-ml-explore HEAD

# Our stack on top.
git -C $MLX rebase --update-refs --onto nax-on-ml-explore <prev-base> thor/stack-mlx-latest
```

Do the rebases in a **detached worktree under a scratch directory**, never in the live
submodule checkout — changing the submodule working tree invalidates the built
`packages/genmlx-core/index.node`:

```bash
git -C $MLX worktree add --detach /tmp/wt-mlx <our-tip>
# ... rebase in /tmp/wt-mlx, then: git -C $MLX branch -f <name> $(git -C /tmp/wt-mlx rev-parse HEAD)
git -C $MLX worktree remove --force /tmp/wt-mlx && git -C $MLX worktree prune
```

**Verify the replay was faithful** — both of these, not just the first:

```bash
git -C $MLX range-diff <old-base>..<old-tip> <new-base>..<new-tip>   # want every line marked '='
diff <(git -C $MLX diff <old-base>..<old-tip>) \
     <(git -C $MLX diff <new-base>..<new-tip>) && echo "patch content byte-identical"
```

**Check the API surface mlx-sys binds** before accepting a jump past what mlx-node tests
against. Only non-`backend/` headers matter; anything defaulted or Metal-only is safe:

```bash
git -C $MLX diff --name-only <old-base>..up/main -- 'mlx/*.h' 'mlx/**/*.h' | grep -v backend/
git -C $MLX log  --oneline   <old-base>..up/main -- mlx/backend/cuda/
```

Then push **before anything pins it**, and tag:

```bash
MLXSHA=$(git -C $MLX rev-parse thor/stack-mlx-latest)
git -C $MLX push origin mirror/ml-explore mirror/nax nax-on-ml-explore thor/stack thor/stack-mlx-latest
git -C $MLX tag -a "pin/mlx/$(date +%F)" -m "pinned by mlx-node sync" $MLXSHA
git -C $MLX push origin "pin/mlx/$(date +%F)"
```

> Cut the `pin/*` tag only once a build has validated the SHA. Keep `thor/stack` (same patches,
> mlx-node's exact base) as the instant fallback if the newer base misbehaves.

---

## 2. mlx-node merge — the expensive step

```bash
git -C $MN branch -f mirror/upstream up/main
git -C $MN checkout -b sync/up-vNEXT genmlx/integration
git -C $MN merge --no-ff vNEXT -m "merge mlx-node vNEXT into genmlx/integration"

# The gitlink. NEVER checkout --ours/--theirs here.
git -C $MN update-index --cacheinfo "160000,$MLXSHA,crates/mlx-sys/mlx"
```

Resolve each conflicted file against [`CONFLICT-LEDGER.md`](CONFLICT-LEDGER.md), then:

```bash
git -C $MN commit          # the merge commit is PURE RESOLUTION — nothing else
```

Every repair beyond resolution lands in **separate follow-up commits** on `sync/up-vNEXT`
(precedent: `b1252958`), so the merge stays reviewable and a later bisect has something to
bisect.

Then run the markerless-hazard checks in §4 of the ledger **before** spending a build.

**On rerere:** keep `rerere.autoupdate=false` on the first pass. rerere keys resolutions on
hunk content, not meaning, and `autoupdate` stages a replayed resolution silently — on a
superficially-similar future hunk that is a defect with no symptom. It earns its keep on
*retries* after a `merge --abort`, where the resolutions replay for free. Keep the cache local
and unversioned; a cross-machine cache multiplies the stale-replay risk.

---

## 3. Build

One GPU process at a time; route anything 35B/80B-class through `~/genmlx-guarded-run.sh`
(see `docs/thor-gpu-discipline.md`).

```bash
cd $MN
yarn install
yarn build:ts
node packages/genmlx-core/build.mjs      # REQUIRED and easy to miss
```

`yarn build` = `build:native` (produces `@mlx-node/core`) + `build:ts`. **GenMLX loads
`@genmlx/core`**, which only `build.mjs` produces. Splitting the command keeps upstream's
darwin-flavoured native packaging off the critical path. Add `yarn build:native` only if the
agent TS path needs it.

On Linux, `build.mjs` also colocates the CUDA JIT headers into `packages/include/` — skipping it
produces cold-cache NVRTC "cannot open cute/..." errors that surface, confusingly, as
`item: array must have size 1`.

**Assert the addon is fresh before trusting any test result.** A stale symlink has previously
caused four days of runs against a five-day-old addon (`genmlx-s8ij`):

```bash
realpath $GENMLX/node_modules/@genmlx/core          # must point into mlx-node/packages/genmlx-core
ls -l $MN/packages/genmlx-core/index.node           # mtime must post-date the build
node -e "console.log(Object.keys(require('$MN/packages/genmlx-core')).length)"   # expect 227
```

Do **not** run `git clean` inside the submodule — it deletes the 139 MB untracked `index.node`.

---

## 4. Battery — never skipped

All four of upstream's build/test/lint jobs are `runs-on: macos-26`; the only `ubuntu-latest`
job is `publish`. So there is no Linux build or test in their CI at all. Their `model-test` e2e
matrix covers qwen3-0.6B, qwen3.5-0.8B dense and LFM2.5 — **no MoE** — and it is label-gated on
`model-e2e`, so it does not run by default. **We are the only integration test for both our
platform and our headline model.**

In gating order — each gate unlocks the next:

```bash
cd $GENMLX
bun run --bun nbb test/genmlx/membrane_coverage_test.cljs      # 227 exports / 49 omissions

# the CLAUDE.md native/membrane contract guard
for f in exact_test gradient_fd_test score_gradient_test clip_contract_test; do
  bun run --bun nbb "test/genmlx/${f}.cljs"; done

bun run --bun nbb test/genmlx/level0_certification_test.cljs   # 68/68
bun run --bun nbb test/genmlx/genjax_compat_test.cljs          # 73/73

# fork-specific guards the CLAUDE.md five do NOT cover
for f in gdn_scan_contract_test native_guard_test gather_qmm_oracle_test \
         qmm_determinism_test conv_scatter_test l3_5_multivariate_test; do
  bun run --bun nbb "test/genmlx/${f}.cljs"; done

TEST_TIME_SCALE=8 test/run.sh all
```

Then **by hand** — the excluded/slow tiers, which are the only things that exercise the native
surface the merge actually disturbed:

```bash
bun run --bun nbb test/genmlx/world_train_test.cljs             # GRPO + GrpoEngineConfig.seed
bun run --bun nbb test/genmlx/world_train_reward_test.cljs
bun run --bun nbb test/genmlx/llm/qwen3_next_native_test.cljs   # 80B native forward
bun run --bun nbb test/genmlx/llm/owned_branch_test.cljs
bun run --bun nbb test/genmlx/llm/branched_test.cljs            # native branchable KV
bun run --bun nbb test/genmlx/llm/vlm_flat_branch_test.cljs     # exclude-tagged (tiers.txt:295)
bun run --bun nbb test/genmlx/llm/qwen35_vlm_e2e_test.cljs      # exclude-tagged (tiers.txt:285)
bun run --bun nbb test/genmlx/llm/vlm_batched_smoke_test.cljs   # exclude-tagged (tiers.txt:294)

# MANDATORY, not optional — the ONLY detector for the markerless-auto-merge hazard class:
python scripts/llm_forward_xval_mlxlm.py

# one live agent turn
~/genmlx-guarded-run.sh mlx agent ...
```

Known baseline flake — do **not** misattribute to the sync: `fused_mcmc` `:300`/`:417`
posterior bands on Thor/CUDA with fixed seeds.

---

## 5. Land and pin — breadth-last

```bash
git -C $MN checkout genmlx/integration
git -C $MN merge --ff-only sync/up-vNEXT
git -C $MN tag -a "pin/mlx-node/$(date +%F)" -m "genmlx pin for mlx-node vNEXT"
git -C $MN push origin genmlx/integration mirror/upstream "pin/mlx-node/$(date +%F)"

cd $GENMLX && git checkout -b sync/mlx-node-vNEXT

# HARD GATE: every pinned commit must EXIST ON A REMOTE (fetchable by SHA).
# Branch reachability is not the invariant — GitHub serves submodule commits by SHA.
# All four of genmlx's submodules are checked; three are forks and instaparse pins a
# non-default branch, so the same rot mechanism applies to each.
for sm in mlx-node malli test.check instaparse; do
  sha=$(git rev-parse HEAD:$sm 2>/dev/null) || continue
  git -C $sm fetch -q origin "$sha" 2>/dev/null \
    && echo "OK   $sm $sha" || echo "FAIL $sm $sha NOT on the remote"
done
sha=$(git -C $MN rev-parse HEAD:crates/mlx-sys/mlx)
git -C $MLX fetch -q origin "$sha" && echo "OK   mlx $sha" || echo "FAIL mlx $sha NOT on the remote"

git add mlx-node && git commit -m "chore(submodule): mlx-node -> vNEXT merged; mlx pin $MLXSHA"
# + a docs/REPRODUCIBILITY.md pin-pair row, + a CONFLICT-LEDGER.md sync entry
git checkout main && git merge --no-ff sync/mlx-node-vNEXT
```

---

## Scheduling constraint

`docs/REPRODUCIBILITY.md` ties published results to a `(genmlx, mlx-node)` pin pair and leans on
the native binary being identical across runs. A sync moves the mlx-node half **and** the MLX
numerics floor. Before starting §2:

```bash
git -C $GENMLX tag -a results/pre-sync-$(date +%F) -m "result set frozen before mlx-node vNEXT"
```

Then either freeze the E1–E6 pins or re-run and add a new pin-pair row. **Do not start a sync
mid-campaign.**
