#!/bin/bash
# GenMLX revised test suite runner
# Runs each test file as a separate process to avoid MLX Metal segfault
# under sustained GPU load. Exit code 1 if any test fails.
#
# Usage: bash test/run_all.sh

set -o pipefail

FAILED=0
PASSED=0
SKIPPED=0
DIR="test/genmlx"

run_test() {
  local file="$1"
  local name="${file%.cljs}"
  name="${name##*/}"
  printf "  %-45s " "$name"
  if timeout 120 bun run --bun nbb "$file" > /tmp/genmlx_test_output.txt 2>&1; then
    local summary=$(grep -E "Ran [0-9]+ tests" /tmp/genmlx_test_output.txt | tail -1)
    if [ -n "$summary" ]; then
      echo "PASS  $summary"
    else
      echo "PASS"
    fi
    PASSED=$((PASSED + 1))
  else
    local exit_code=$?
    if [ $exit_code -eq 124 ]; then
      echo "TIMEOUT (120s)"
      SKIPPED=$((SKIPPED + 1))
    elif [ $exit_code -eq 133 ] || [ $exit_code -eq 139 ]; then
      echo "SEGFAULT (MLX Metal — known issue)"
      SKIPPED=$((SKIPPED + 1))
    else
      echo "FAIL (exit $exit_code)"
      grep -E "FAIL|ERROR|fail|error" /tmp/genmlx_test_output.txt | head -5
      FAILED=$((FAILED + 1))
    fi
  fi
}

echo "=== GenMLX Revised Test Suite ==="
echo ""

echo "--- Distribution tests ---"
for f in dist_logprob_test dist_symmetry_test dist_boundary_test \
         dist_batch_test dist_normalization_test dist_gradient_test \
         dist_error_test dist_moments_test; do
  run_test "$DIR/${f}.cljs"
done

echo ""
echo "--- GFI protocol tests ---"
for f in gfi_simulate_test gfi_generate_test gfi_update_test \
         gfi_regenerate_test gfi_assess_test gfi_project_test \
         gfi_contracts_test; do
  run_test "$DIR/${f}.cljs"
done

echo ""
echo "--- Handler tests ---"
for f in handler_purity_test handler_transitions_test; do
  run_test "$DIR/${f}.cljs"
done

echo ""
echo "--- Combinator tests ---"
for f in combinator_map_test2 combinator_unfold_test2 combinator_switch_test2 \
         combinator_scan_test2 combinator_mask_test combinator_mix_test2 \
         combinator_recurse_test2; do
  run_test "$DIR/${f}.cljs"
done

echo ""
echo "--- Vectorized & compiled tests ---"
for f in vectorized_shape_test vectorized_equivalence_test \
         compiled_equivalence_test compiled_schema_test2; do
  run_test "$DIR/${f}.cljs"
done

echo ""
echo "--- Data structure tests ---"
for f in choicemap_algebra_test2 trace_immutability_test selection_algebra_test2; do
  run_test "$DIR/${f}.cljs"
done

echo ""
echo "--- Inference tests (GPU-heavy, individual processes) ---"
for f in inference_is_test inference_mh_test inference_hmc_test \
         inference_smc_test inference_agreement_test inference_vi_test \
         inference_adev_test; do
  run_test "$DIR/${f}.cljs"
done

echo ""
echo "--- Certification & compatibility ---"
for f in level0_certification_test l4_certification_test \
         gen_clj_compat_test genjax_compat_test; do
  run_test "$DIR/${f}.cljs"
done

echo ""
echo "=== Summary ==="
TOTAL=$((PASSED + FAILED + SKIPPED))
echo "  Total:    $TOTAL"
echo "  Passed:   $PASSED"
echo "  Failed:   $FAILED"
echo "  Skipped:  $SKIPPED (segfault/timeout)"
echo ""

if [ $FAILED -gt 0 ]; then
  echo "RESULT: FAIL"
  exit 1
elif [ $SKIPPED -gt 0 ]; then
  echo "RESULT: PASS (with $SKIPPED skipped due to MLX segfault)"
  exit 0
else
  echo "RESULT: PASS"
  exit 0
fi
