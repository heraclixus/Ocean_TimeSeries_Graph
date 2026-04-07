#!/bin/bash
#
# Generate All Paper Plots
#
# This script runs all plot generation scripts according to plots.md requirements.
#
# Usage:
#   ./run_generate_plots.sh              # Generate all plots
#   ./run_generate_plots.sh quick        # Generate only deterministic plots (1-6)
#   ./run_generate_plots.sh ensemble     # Generate only ensemble plots (7-9)
#   ./run_generate_plots.sh special      # Generate special case 3x2 plots (section 10)
#   ./run_generate_plots.sh section 1    # Generate only section 1
#

set -e

echo "========================================"
echo "PAPER PLOT GENERATION"
echo "========================================"
echo ""

MODE="${1:-all}"
SECTION="${2:-}"

case "$MODE" in
    "quick")
        echo "Mode: Quick (Sections 1-6 only)"
        echo ""
        python generate_paper_plots.py --sections 1,2,3,4,5,6
        ;;
    
    "ensemble")
        echo "Mode: Ensemble plots (Sections 7-9)"
        echo ""
        echo "Generating XRO baseline ensemble plots..."
        python generate_ensemble_plumes.py --section 7
        echo ""
        echo "Generating Top 5 ORAS5 ensemble plots..."
        python generate_ensemble_plumes.py --section 8 --combined
        echo ""
        echo "Generating Top 5 Two-Stage ensemble plots..."
        python generate_ensemble_plumes.py --section 9 --combined
        ;;
    
    "special")
        echo "Mode: Special case 3x2 plots (Section 10)"
        echo ""
        echo "Key ENSO events and Spring Predictability Barrier cases"
        python generate_paper_plots.py --sections 10
        ;;
    
    "section")
        if [ -z "$SECTION" ]; then
            echo "Error: Please specify section number"
            echo "Usage: ./run_generate_plots.sh section <1-10>"
            exit 1
        fi
        
        echo "Mode: Single section ($SECTION)"
        echo ""
        
        if [ "$SECTION" -le 6 ]; then
            python generate_paper_plots.py --sections $SECTION
        else
            python generate_ensemble_plumes.py --section $SECTION
        fi
        ;;
    
    "all"|*)
        echo "Mode: All plots"
        echo ""
        
        echo "========================================" 
        echo "STEP 1: Deterministic & Uncertainty Plots"
        echo "========================================"
        python generate_paper_plots.py --sections 1,2,3,4,5,6
        
        echo ""
        echo "========================================"
        echo "STEP 2: Ensemble Plume Plots"
        echo "========================================"
        
        # Note: This can take a long time as it generates plots for every month/year
        # For a quick test, you can limit to a specific year:
        # python generate_ensemble_plumes.py --year 1997
        
        python generate_ensemble_plumes.py --section 7
        python generate_ensemble_plumes.py --section 8 --combined
        python generate_ensemble_plumes.py --section 9 --combined
        ;;
esac

echo ""
echo "========================================"
echo "COMPLETE"
echo "========================================"
echo ""
echo "Output directory: plots/"
echo ""
echo "Directory structure:"
echo "  plots/"
echo "  ├── 1_deterministic_oras5/       # ORAS5-only model rankings"
echo "  ├── 2_deterministic_two_stage/   # Two-stage model rankings"
echo "  ├── 3_deterministic_combined/    # Combined model rankings"
echo "  ├── 4_uncertainty_oras5/         # CRPS rankings (ORAS5)"
echo "  ├── 5_uncertainty_two_stage/     # CRPS rankings (Two-stage)"
echo "  ├── 6_generalization_gaps/       # Train/Test gap analysis"
echo "  ├── 7_ensemble_xro/              # XRO ensemble forecasts"
echo "  ├── 8_ensemble_top5_oras5/       # Top 5 ORAS5 ensemble forecasts"
echo "  └── 9_ensemble_top5_two_stage/   # Top 5 Two-stage ensemble forecasts"
echo ""
