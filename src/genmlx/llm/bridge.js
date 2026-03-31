/**
 * Bridge between nbb (ClojureScript) and @mlx-node/core MxArray.
 *
 * Key discovery: NAPI-RS shape parameters require BigInt64Array, not
 * regular JS arrays. This module handles the conversion transparently.
 */
const { MxArray } = require('@mlx-node/core');

/** Convert a JS array of numbers to BigInt64Array (required by NAPI-RS for shapes). */
function toBigShape(shape) {
  return new BigInt64Array(shape.map(BigInt));
}

/** Create an MxArray from a JS array of integers + shape array. */
exports.fromInt32 = function(data, shape) {
  return MxArray.fromInt32(new Int32Array(data), toBigShape(shape));
};

/** Create an MxArray from a JS array of unsigned integers + shape array. */
exports.fromUint32 = function(data, shape) {
  return MxArray.fromUint32(new Uint32Array(data), toBigShape(shape));
};

/** Create an MxArray from a JS array of floats + shape array. */
exports.fromFloat32 = function(data, shape) {
  return MxArray.fromFloat32(new Float32Array(data), toBigShape(shape));
};

/** Create a zeros MxArray of given shape. */
exports.zeros = function(shape) {
  return MxArray.zeros(toBigShape(shape));
};

/** Create a ones MxArray of given shape. */
exports.ones = function(shape) {
  return MxArray.ones(toBigShape(shape));
};

/** MxArray → Float32Array */
exports.toFloat32 = function(arr) { return arr.toFloat32(); };

/** MxArray → Int32Array */
exports.toInt32 = function(arr) { return arr.toInt32(); };

/** MxArray → Uint32Array */
exports.toUint32 = function(arr) { return arr.toUint32(); };

/** Get shape as regular JS array of numbers. */
exports.shape = function(arr) {
  return Array.from(arr.shape(), Number);
};

/** Reshape MxArray. */
exports.reshape = function(arr, shape) {
  return arr.reshape(toBigShape(shape));
};

/** Slice MxArray. start/stop are JS arrays of numbers. */
exports.slice = function(arr, start, stop) {
  return arr.slice(start, stop);
};

/** Log-softmax along last axis. */
exports.logSoftmax = function(arr) { return arr.logSoftmax(); };

/** Argmax. */
exports.argmax = function(arr) { return arr.argmax(); };

/**
 * Run a forward pass and return last-position logits as Float32Array.
 * Handles all MxArray creation/slicing on the JS side to avoid
 * TypedArray interop issues with nbb/SCI.
 *
 * @param {Qwen3Model} model - loaded model
 * @param {Uint32Array|number[]} tokenIds - token IDs
 * @returns {Float32Array} last-position logits [vocab_size]
 */
exports.forwardLastLogits = function(model, tokenIds) {
  const plain = Array.from(tokenIds);
  const n = plain.length;
  // Create MxArray from plain numbers
  const input = MxArray.fromUint32(new Uint32Array(plain), toBigShape([1, n]));
  // Forward pass
  const logits = model.forward(input);
  const shape = Array.from(logits.shape(), Number);
  const vocabSize = shape[2];
  // Get all logits as Float32Array, then slice last position on JS side
  const allF32 = logits.toFloat32();
  const offset = (n - 1) * vocabSize;
  return allF32.slice(offset, offset + vocabSize);
};
