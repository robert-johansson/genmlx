/**
 * GenMLX ops as pi agent tools (genmlx-wdx0, L5-A): Bayesian model
 * evaluation and evidence scoring as tool calls, over a SIGTRAP-safe
 * subprocess bridge.
 *
 * Run it:
 *
 *   mlx agent --model genmlx/<name> --no-builtin-tools \
 *     --extension /home/robert/code/mlx/genmlx/extensions/genmlx-tools.ts \
 *     -p 'call genmlx_score_model with code "(fn [trace] ...)" and observations {"y0": 2.0}'
 *
 * PROCESS RULE (hard, the reason this file spawns a child): `@mlx-node/*`
 * and `@genmlx/core` in one process = SIGTRAP. The agent process holds
 * one of them (whichever provider is serving), so every tool call runs
 * `bun run --bun nbb scripts/genmlx_tool_worker.cljs` in its OWN process
 * from the genmlx repo root — JSON on stdin, one `GENMLX_RESULT:` line
 * out. GPU-light by construction: SCI eval + tiny scalar graphs; no LLM
 * checkpoint ever loads in the child.
 *
 * The model-code contract is msa_score's: `code` is a ClojureScript
 * `(fn [trace] ...)` form over `dist/*` constructors, e.g.
 *   (fn [trace]
 *     (let [mu (trace :mu (dist/gaussian 0 3))]
 *       (trace :y0 (dist/gaussian mu 1))))
 * Observations map trace addresses to numbers: {"y0": 2.0}. Conjugate
 * models score by EXACT analytical evidence; everything else falls back
 * to importance sampling (`method` reports which).
 *
 * `runGenmlxOp` is deliberately pi-import-free so it can be driven
 * directly by tests; only the default-export factory touches the pi API
 * (loaded via jiti inside the agent — no node_modules needed here).
 */
import { spawn } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { Type } from '@earendil-works/pi-ai';
import { defineTool, type ExtensionAPI } from '@earendil-works/pi-coding-agent';

const MARKER = 'GENMLX_RESULT:';

/** The genmlx repo root: GENMLX_HOME wins, else this file's parent dir. */
function genmlxHome(): string {
  const env = process.env.GENMLX_HOME;
  if (env) return env;
  return dirname(dirname(fileURLToPath(import.meta.url)));
}

export interface GenmlxOpResult {
  ok: boolean;
  error?: string;
  [key: string]: unknown;
}

/**
 * Run one worker op in a child process. Never throws — every failure
 * (spawn, timeout, protocol) becomes `{ ok: false, error }`.
 */
export function runGenmlxOp(
  request: Record<string, unknown>,
  opts: { timeoutMs?: number } = {},
): Promise<GenmlxOpResult> {
  const timeoutMs = opts.timeoutMs ?? 120_000;
  const home = genmlxHome();
  // Thor CUDA env defaults — set only when absent so an already-configured
  // environment wins (docs/thor-gpu-discipline.md incantation).
  const env = {
    ...process.env,
    LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH ?? '/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu',
    CUDA_HOME: process.env.CUDA_HOME ?? '/usr/local/cuda',
    CUDA_PATH: process.env.CUDA_PATH ?? '/usr/local/cuda',
    GLIBC_TUNABLES: process.env.GLIBC_TUNABLES ?? 'glibc.rtld.optional_static_tls=8192',
  };
  return new Promise((resolve) => {
    let settled = false;
    const finish = (result: GenmlxOpResult): void => {
      if (!settled) {
        settled = true;
        resolve(result);
      }
    };
    let child;
    try {
      child = spawn('bun', ['run', '--bun', 'nbb', join('scripts', 'genmlx_tool_worker.cljs')], {
        cwd: home,
        env,
        stdio: ['pipe', 'pipe', 'pipe'],
      });
    } catch (err) {
      finish({ ok: false, error: `spawn failed: ${String(err)}` });
      return;
    }
    let stdout = '';
    let stderr = '';
    const timer = setTimeout(() => {
      try {
        child.kill('SIGKILL');
      } catch {
        // already gone
      }
      finish({ ok: false, error: `genmlx worker timed out after ${timeoutMs}ms` });
    }, timeoutMs);
    child.stdout.on('data', (d: Buffer) => {
      stdout += d.toString();
    });
    child.stderr.on('data', (d: Buffer) => {
      stderr += d.toString();
    });
    child.on('error', (err) => {
      clearTimeout(timer);
      finish({ ok: false, error: `worker error: ${String(err)}` });
    });
    child.on('close', (code) => {
      clearTimeout(timer);
      const line = stdout
        .split('\n')
        .filter((l) => l.startsWith(MARKER))
        .pop();
      if (!line) {
        finish({
          ok: false,
          error: `worker exited (code ${code}) without a ${MARKER} line; stderr tail: ${stderr.slice(-400)}`,
        });
        return;
      }
      try {
        finish(JSON.parse(line.slice(MARKER.length)) as GenmlxOpResult);
      } catch (err) {
        finish({ ok: false, error: `malformed worker result: ${String(err)}` });
      }
    });
    child.stdin.end(JSON.stringify(request));
  });
}

function summarize(result: GenmlxOpResult): string {
  if (!result.ok) return `genmlx op failed: ${result.error ?? 'unknown error'}`;
  if ('ranking' in result) {
    const rows = (result.ranking as Array<Record<string, unknown>>).map(
      (r) =>
        `#${String(r.index)}: ${r.valid ? (r.finite ? `log-ML ${(r.logMl as number).toFixed(3)} (${String(r.method)})` : 'log-ML -Inf') : 'INVALID'}`,
    );
    return `ranked ${rows.length} candidate model(s), best first:\n${rows.join('\n')}`;
  }
  if ('logMl' in result) {
    return result.finite
      ? `log-ML ${(result.logMl as number).toFixed(3)} (method ${String(result.method)})`
      : 'log-ML -Inf (model invalid or scoring failed)';
  }
  if ('valid' in result) {
    if (!result.valid) return `model INVALID: ${String(result.error ?? 'did not evaluate')}`;
    const schema = result.schema as { traceSites: Array<{ addr: string; distType: string | null }>; static: boolean; conjugate: boolean };
    const sites = schema.traceSites.map((s) => `${s.addr}~${s.distType ?? '?'}`).join(', ');
    return `model VALID: sites [${sites}], static=${schema.static}, conjugate=${schema.conjugate}`;
  }
  return JSON.stringify(result);
}

const CODE_PARAM = Type.String({
  description:
    'ClojureScript model code: a (fn [trace] ...) form over dist/* constructors, e.g. (fn [trace] (let [mu (trace :mu (dist/gaussian 0 3))] (trace :y0 (dist/gaussian mu 1))))',
});
const OBSERVATIONS_PARAM = Type.Record(Type.String(), Type.Number(), {
  description: 'Observed values keyed by trace address, e.g. {"y0": 2.0, "y1": 1.7}',
});

export default function genmlxTools(pi: ExtensionAPI) {
  pi.registerTool(
    defineTool({
      name: 'genmlx_eval_model',
      label: 'GenMLX eval model',
      description:
        'Evaluate GenMLX probabilistic model code (SCI-sandboxed). Returns validity and the model schema (trace sites, staticness, conjugacy). Use before scoring to check a model parses.',
      parameters: Type.Object({ code: CODE_PARAM }),
      async execute(_toolCallId, params) {
        const result = await runGenmlxOp({ op: 'eval-model', code: params.code });
        return { content: [{ type: 'text', text: summarize(result) }], details: result };
      },
    }),
  );

  pi.registerTool(
    defineTool({
      name: 'genmlx_score_model',
      label: 'GenMLX score model',
      description:
        'Score a GenMLX model against observed data by Bayesian model evidence (log marginal likelihood). Conjugate models score exactly; others by importance sampling. Higher log-ML = better model of the data.',
      parameters: Type.Object({
        code: CODE_PARAM,
        observations: OBSERVATIONS_PARAM,
        nParticles: Type.Optional(Type.Number({ description: 'Importance samples for non-conjugate models (default 50)' })),
      }),
      async execute(_toolCallId, params) {
        const result = await runGenmlxOp({
          op: 'score-model',
          code: params.code,
          observations: params.observations,
          nParticles: params.nParticles,
        });
        return { content: [{ type: 'text', text: summarize(result) }], details: result };
      },
    }),
  );

  pi.registerTool(
    defineTool({
      name: 'genmlx_rank_models',
      label: 'GenMLX rank models',
      description:
        'Rank several candidate GenMLX models against the same observed data by log marginal likelihood, best first. Invalid or unscoreable candidates rank last.',
      parameters: Type.Object({
        candidates: Type.Array(CODE_PARAM, { description: 'Candidate model code strings' }),
        observations: OBSERVATIONS_PARAM,
        nParticles: Type.Optional(Type.Number({ description: 'Importance samples for non-conjugate models (default 50)' })),
      }),
      async execute(_toolCallId, params) {
        const result = await runGenmlxOp({
          op: 'rank-models',
          candidates: params.candidates,
          observations: params.observations,
          nParticles: params.nParticles,
        });
        return { content: [{ type: 'text', text: summarize(result) }], details: result };
      },
    }),
  );
}
