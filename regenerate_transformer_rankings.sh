#!/bin/bash

# Script to regenerate rankings to include Transformer model
# This ensures Transformer appears in both single-stage and two-stage comparisons

set -e

PYTHON="/Users/fredxu/miniconda3/envs/graph/bin/python"

echo "=========================================================================="
echo "REGENERATING RANKINGS TO INCLUDE TRANSFORMER MODEL"
echo "=========================================================================="
echo ""

echo "Step 1: Regenerating single-stage rankings..."
echo "----------------------------------------------------------------------"
$PYTHON rank_all_variants_out_of_sample.py --metric combined --force
echo ""

echo "Step 2: Regenerating two-stage rankings..."
echo "----------------------------------------------------------------------"
$PYTHON rank_all_variants_out_of_sample.py --two_stage --metric combined --force
echo ""

echo "Step 3: Generating comparison plots..."
echo "----------------------------------------------------------------------"
$PYTHON compare_single_vs_two_stage.py --metric combined
echo ""

echo "=========================================================================="
echo "✓ COMPLETE!"
echo "=========================================================================="
echo ""
echo "Transformer model should now appear in:"
echo "  - results_out_of_sample/rankings/all_variants_ranked_combined_out_of_sample.csv"
echo "  - results_out_of_sample/rankings/all_variants_ranked_combined_out_of_sample_two_stage.csv"
echo "  - results_out_of_sample/rankings/comparison_single_vs_two_stage_summary.csv"
echo ""

