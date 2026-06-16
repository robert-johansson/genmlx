// Ctx-agnostic Canvas2D renderer for the GenMLX Agents book.
//
// Consumes the genmlx.agents.presentation DATA shapes produced in ClojureScript
// (Frame / Trajectory / PosteriorBars), and draws them to ANY Canvas2D context —
// a headless @napi-rs/canvas (the figure pipeline) or a browser <canvas> (the
// index.html demo). Pure drawing; no I/O, no cs188-cljs dependency.
//
// Frame         {W, H, cells: [{glyph, role, value?}], meta:{step, action}}
//   role ∈ wall | empty | pacman | pellet | power | fruit | ghost | path | goal
// PosteriorBars {title, bars: [{label, weight, highlight?}]}
// LineChart     {title, xlabel?, ylabel?, series: [{label, points:[y...]}]}

export const THEME = {
  bg: '#0b1026', wall: '#2b3aa0', pacman: '#ffd23f',
  pellet: '#ffe0a3', power: '#7fe7ff', fruit: '#ff5e6c', ghost: '#ff79c6',
  path: '#3a4480', heatLo: '#101733', heatHi: '#ffb454',
  text: '#e6e9ff', dim: '#8b93c4', track: '#1b2347',
  series: ['#ffd23f', '#7fe7ff', '#ff79c6', '#9bff9b', '#ff5e6c'],
};

const hex = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
function lerpColor(a, b, t) {
  const [ar, ag, ab] = hex(a), [br, bg, bb] = hex(b);
  const m = (x, y) => Math.round(x + (y - x) * Math.max(0, Math.min(1, t)));
  return `rgb(${m(ar, br)},${m(ag, bg)},${m(ab, bb)})`;
}
function roundRect(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawCache(ctx, cx, cy, cell, role) {
  const col = role === 'pellet' ? THEME.pellet : role === 'power' ? THEME.power
            : role === 'fruit' ? THEME.fruit : THEME.pellet;
  const r = role === 'pellet' ? cell * 0.15 : cell * 0.27;
  ctx.save();
  if (role !== 'pellet') { ctx.shadowColor = col; ctx.shadowBlur = cell * 0.35; }
  ctx.fillStyle = col;
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2 * Math.PI); ctx.fill();
  ctx.restore();
}
function drawPacman(ctx, cx, cy, cell, action, frameIndex) {
  const r = cell * 0.4;
  const open = 0.06 + 0.17 * Math.abs(Math.sin(frameIndex * 0.9));
  const base = { right: 0, left: Math.PI, up: -Math.PI / 2, down: Math.PI / 2 }[action] ?? 0;
  ctx.fillStyle = THEME.pacman;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.arc(cx, cy, r, base + open * Math.PI, base + (2 - open) * Math.PI);
  ctx.closePath(); ctx.fill();
}
function drawGhost(ctx, cx, cy, cell) {
  const r = cell * 0.36, top = cy - r * 0.2, bot = cy + r * 0.8;
  ctx.fillStyle = THEME.ghost;
  ctx.beginPath();
  ctx.arc(cx, top, r, Math.PI, 0);
  ctx.lineTo(cx + r, bot);
  for (let i = 0; i < 3; i++) { const x = cx + r - (2 * r) * (i + 1) / 3; ctx.lineTo(x + r / 3, bot - r * 0.22); ctx.lineTo(x, bot); }
  ctx.closePath(); ctx.fill();
  ctx.fillStyle = '#fff';
  ctx.beginPath(); ctx.arc(cx - r * 0.4, top, r * 0.24, 0, 2 * Math.PI); ctx.arc(cx + r * 0.4, top, r * 0.24, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = THEME.bg;
  ctx.beginPath(); ctx.arc(cx - r * 0.34, top, r * 0.11, 0, 2 * Math.PI); ctx.arc(cx + r * 0.46, top, r * 0.11, 0, 2 * Math.PI); ctx.fill();
}

export function frameSize(frame, cell) { return { width: frame.W * cell, height: frame.H * cell }; }

export function renderFrame(ctx, frame, opts = {}) {
  const cell = opts.cell ?? 36;
  const frameIndex = opts.frameIndex ?? (frame.meta && frame.meta.step) ?? 0;
  const { W, cells } = frame;
  ctx.fillStyle = THEME.bg; ctx.fillRect(0, 0, frame.W * cell, frame.H * cell);
  for (let idx = 0; idx < cells.length; idx++) {
    const x = idx % W, y = (idx / W) | 0, px = x * cell, py = y * cell, cx = px + cell / 2, cy = py + cell / 2;
    const c = cells[idx], role = c.role;
    if (role === 'wall') { ctx.fillStyle = THEME.wall; roundRect(ctx, px + 1.5, py + 1.5, cell - 3, cell - 3, 6); ctx.fill(); continue; }
    if (c.value != null) { ctx.fillStyle = lerpColor(THEME.heatLo, THEME.heatHi, c.value); ctx.fillRect(px, py, cell, cell); }
    if (role === 'pacman') drawPacman(ctx, cx, cy, cell, frame.meta && frame.meta.action, frameIndex);
    else if (role === 'ghost') drawGhost(ctx, cx, cy, cell);
    else if (role === 'pellet' || role === 'power' || role === 'fruit' || role === 'goal') drawCache(ctx, cx, cy, cell, role);
    else if (role === 'path') { ctx.fillStyle = THEME.path; ctx.beginPath(); ctx.arc(cx, cy, cell * 0.1, 0, 2 * Math.PI); ctx.fill(); }
  }
}

export function barsSize(p, opts = {}) {
  const rowH = opts.rowH ?? 30, pad = 18, titleH = 28;
  return { width: opts.width ?? 460, height: pad * 2 + titleH + p.bars.length * rowH };
}
export function renderBars(ctx, p, opts = {}) {
  const rowH = opts.rowH ?? 30, pad = 18, titleH = 28;
  const { width, height } = barsSize(p, opts);
  ctx.fillStyle = THEME.bg; ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = THEME.text; ctx.font = '600 15px sans-serif'; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
  ctx.fillText(p.title ?? '', pad, pad);
  const labelW = 116, barX = pad + labelW, barW = width - barX - 56;
  ctx.textBaseline = 'middle';
  p.bars.forEach((b, i) => {
    const y = pad + titleH + i * rowH;
    ctx.fillStyle = THEME.dim; ctx.font = '13px sans-serif'; ctx.textAlign = 'right';
    ctx.fillText(b.label, barX - 8, y + rowH / 2);
    ctx.fillStyle = THEME.track; roundRect(ctx, barX, y + 5, barW, rowH - 12, 4); ctx.fill();
    ctx.fillStyle = b.highlight ? THEME.fruit : THEME.pacman;
    roundRect(ctx, barX, y + 5, Math.max(2, barW * b.weight), rowH - 12, 4); ctx.fill();
    ctx.fillStyle = THEME.text; ctx.textAlign = 'left';
    ctx.fillText(b.weight.toFixed(3), barX + barW + 8, y + rowH / 2);
  });
}

export function linesSize(chart, opts = {}) { return { width: opts.width ?? 480, height: opts.height ?? 300 }; }
export function renderLines(ctx, chart, opts = {}) {
  const { width, height } = linesSize(chart, opts);
  const L = 52, R = 16, T = 36, B = 36;
  ctx.fillStyle = THEME.bg; ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = THEME.text; ctx.font = '600 15px sans-serif'; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
  ctx.fillText(chart.title ?? '', L, 10);
  const all = chart.series.flatMap(s => s.points);
  const lo = Math.min(...all), hi = Math.max(...all), span = (hi - lo) || 1;
  const n = Math.max(...chart.series.map(s => s.points.length));
  const xOf = i => L + (width - L - R) * (n > 1 ? i / (n - 1) : 0);
  const yOf = v => (height - B) - (height - T - B) * (v - lo) / span;
  ctx.strokeStyle = THEME.track; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(L, T); ctx.lineTo(L, height - B); ctx.lineTo(width - R, height - B); ctx.stroke();
  ctx.fillStyle = THEME.dim; ctx.font = '11px sans-serif'; ctx.textBaseline = 'middle'; ctx.textAlign = 'right';
  ctx.fillText(hi.toFixed(1), L - 6, T); ctx.fillText(lo.toFixed(1), L - 6, height - B);
  chart.series.forEach((s, si) => {
    ctx.strokeStyle = THEME.series[si % THEME.series.length]; ctx.lineWidth = 2;
    ctx.beginPath();
    s.points.forEach((v, i) => { const X = xOf(i), Y = yOf(v); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
    ctx.stroke();
    s.points.forEach((v, i) => { ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.arc(xOf(i), yOf(v), 2.4, 0, 2 * Math.PI); ctx.fill(); });
  });
}
