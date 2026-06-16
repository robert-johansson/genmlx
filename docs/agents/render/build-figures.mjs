// Stage 2 of the figure pipeline: read the manifest + per-figure JSON emitted by
// figures.cljs (stage 1) and render each to a PNG/GIF in docs/agents/src/figures/.
// Deterministic: identical input data -> identical bytes.
//
// Run via bin/agents-book-figures (which runs stage 1 first).

import { readFileSync, writeFileSync, mkdirSync, statSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import * as cap from './capture.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const dataDir = join(here, 'data');
const outDir = join(here, '..', 'src', 'figures');
mkdirSync(outDir, { recursive: true });

const manifest = JSON.parse(readFileSync(join(dataDir, 'manifest.json'), 'utf8'));

const RENDER = {
  frame:      (data, opts) => ['png', cap.pngFromFrame(data, opts)],
  trajectory: (data, opts) => ['gif', cap.gifFromTrajectory(data, opts)],
  bars:       (data, opts) => ['png', cap.pngFromBars(data, opts)],
  lines:      (data, opts) => ['png', cap.pngFromLines(data, opts)],
};

let n = 0;
for (const fig of manifest.figures) {
  const render = RENDER[fig.kind];
  if (!render) throw new Error(`unknown figure kind: ${fig.kind} (${fig.id})`);
  const data = JSON.parse(readFileSync(join(dataDir, `${fig.id}.json`), 'utf8'));
  const [ext, buf] = render(data, fig.opts || {});
  const out = join(outDir, `${fig.id}.${ext}`);
  writeFileSync(out, buf);
  console.log(`  ${fig.id}.${ext}  (${fig.kind}, ${buf.length} bytes)`);
  n++;
}
console.log(`\nrendered ${n} figures -> docs/agents/src/figures/`);
