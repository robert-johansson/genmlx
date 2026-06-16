// Headless capture: render the genmlx.agents.presentation data shapes to PNG/GIF
// buffers using @napi-rs/canvas + gifenc. The drawing itself lives in
// pacman-render.mjs (ctx-agnostic); this module only owns the headless surface.

import { createCanvas } from '@napi-rs/canvas';
import gifenc from 'gifenc';
import * as R from './pacman-render.mjs';

const { GIFEncoder, quantize, applyPalette } = gifenc;

// Exact RGBA-buffer equality, for dropping duplicate consecutive frames.
function sameFrame(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// Encode RGBA frame buffers into a LOOPING GIF. Consecutive duplicate frames are
// dropped so the loop never lingers on a static state (e.g. a snapped belief held
// for several rollout steps), and the end-hold is short — the animation reads as a
// continuous cycle, not a still image. repeat:0 on the first written frame writes
// the NETSCAPE2.0 loop extension (loop forever).
function encodeLooping(framesData, width, height, delay) {
  const distinct = [];
  for (const d of framesData)
    if (distinct.length === 0 || !sameFrame(distinct[distinct.length - 1], d)) distinct.push(d);
  const enc = GIFEncoder();
  distinct.forEach((data, i) => {
    const pal = quantize(data, 256);
    const ms = i === distinct.length - 1 ? Math.round(delay * 1.6) : delay;  // a short beat before looping
    enc.writeFrame(applyPalette(data, pal), width, height,
      i === 0 ? { palette: pal, delay: ms, repeat: 0 } : { palette: pal, delay: ms });
  });
  enc.finish();
  return Buffer.from(enc.bytes());
}

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
  const ctx = createCanvas(width, height).getContext('2d');
  const data = frames.map((fr, i) => {
    R.renderFrame(ctx, fr, { cell, frameIndex: i });
    return ctx.getImageData(0, 0, width, height).data.slice();
  });
  return encodeLooping(data, width, height, delay);
}

export function pngFromBars(p, opts = {}) {
  const { width, height } = R.barsSize(p, opts);
  const c = createCanvas(width, height);
  R.renderBars(c.getContext('2d'), p, opts);
  return c.toBuffer('image/png');
}

export function gifFromBars(barsList, opts = {}) {
  const delay = opts.delay ?? 500;
  // Stable canvas across frames: size to the max bar count.
  const maxN = Math.max(...barsList.map(p => p.bars.length));
  const { width, height } = R.barsSize({ bars: new Array(maxN) }, opts);
  const ctx = createCanvas(width, height).getContext('2d');
  const data = barsList.map(p => {
    R.renderBars(ctx, p, opts);
    return ctx.getImageData(0, 0, width, height).data.slice();
  });
  return encodeLooping(data, width, height, delay);
}

export function pngFromLines(chart, opts = {}) {
  const { width, height } = R.linesSize(chart, opts);
  const c = createCanvas(width, height);
  R.renderLines(c.getContext('2d'), chart, opts);
  return c.toBuffer('image/png');
}
