#!/bin/bash
# Sync generated figures/tables from results/ to the Overleaf repo and push.
# Usage: ./scripts/sync_to_overleaf.sh [commit message]

set -e

PAPER_DIR="paper/TOPML_system"
RESULTS_DIR="results"
MSG="${1:-Update figures and tables from latest results}"

# Copy all paper figures
for f in fig0_architecture fig1_particle_scaling fig2_method_speedup \
         fig3_ffi_scaling fig5_linreg_posteriors fig7_hmm_logml \
         fig8_gmm_logml fig9_system_bars fig12_contract_heatmap; do
    if [ -f "$RESULTS_DIR/${f}.pdf" ]; then
        cp "$RESULTS_DIR/${f}.pdf" "$PAPER_DIR/${f}.pdf"
        echo "  Copied ${f}.pdf"
    fi
done

# Copy table .tex files if they exist in results
for f in fig4_speedup_table fig6_linreg_table fig10_system_table \
         fig11_verification_ladder compilation_table; do
    if [ -f "$RESULTS_DIR/${f}.tex" ]; then
        cp "$RESULTS_DIR/${f}.tex" "$PAPER_DIR/${f}.tex"
        echo "  Copied ${f}.tex"
    fi
done

# Commit and push in the Overleaf subrepo
cd "$PAPER_DIR"
if git diff --quiet && git diff --cached --quiet; then
    echo "No changes to push."
    exit 0
fi

git add -A
git commit -m "$MSG

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
git push origin master
echo "Pushed to Overleaf."
