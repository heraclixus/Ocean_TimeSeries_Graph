#!/bin/bash
# Generate forecast plume plots for specific initialization dates
# Similar to XRO_example.py plot_forecast_plume

echo "==================================================================="
echo "Generating Stochastic Forecast Plume Plots"
echo "==================================================================="

# Default dates (similar to XRO_example.py)
python visualize_stochastic_comparison.py \
    --results_dir results_out_of_sample \
    --plot_plumes \
    --plume_dates 1988-04