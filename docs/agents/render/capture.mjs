// Headless capture: render the genmlx.agents.presentation data shapes to PNG/GIF
// buffers using @napi-rs/canvas + gifenc. The drawing itself lives in
// pacman-render.mjs (ctx-agnostic); this module only owns the headless surface.

import { createCanvas } from '@napi-rs/canvas';
import gifenc from 'gifenc';
import * as R from './pacman-render.mjs';

const { GIFEncoder, quantize, applyPalette } = gifenc;

export function pngFromFrame(frame, opts = {}) {
  const cell = opts.cell ?? 36;
  const { width, height } = R.frameSize(frame, cell);
  const c = createCanvas(width, height);
  R.renderFrame(c.getContext('2d'), frame, { cell });
  return c.toBuffer('image/png');
}

export function gifFromTrajectory(frames, opts = {}) {
  const cell = opts.cell ?? 36, delay = opts.delay ?? 360;
  const { width, height } = R.frameSize(frames[0], cell);
  const c = createCanvas(width, height);
  const ctx = c.getContext('2d');
  const enc = GIFEncoder();
  frames.forEach((fr, i) => {
    R.renderFrame(ctx, fr, { cell, frameIndex: i });
    const { data } = ctx.getImageData(0, 0, width, height);
    const pal = quantize(data, 256);
    // hold the final frame longer so the GIF reads as "arrived"
    const d = i === frames.length - 1 ? delay * 3 : delay;
    enc.writeFrame(applyPalette(data, pal), width, height, { palette: pal, delay: d });
  });
  enc.finish();
  return Buffer.from(enc.bytes());
}

export function pngFromBars(p, opts = {}) {
  const { width, height } = R.barsSize(p, opts);
  const c = createCanvas(width, height);
  R.renderBars(c.getContext('2d'), p, opts);
  return c.toBuffer('image/png');
}

export function pngFromLines(chart, opts = {}) {
  const { width, height } = R.linesSize(chart, opts);
  const c = createCanvas(width, height);
  R.renderLines(c.getContext('2d'), chart, opts);
  return c.toBuffer('image/png');
}
