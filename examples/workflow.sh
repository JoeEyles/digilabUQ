#!/usr/bin/env bash

# ============================================================
# Automated Workflow
# ============================================================

set -euo pipefail

# Activate the designated runtime environment
conda activate digilab
echo "Runtime environment activated"

# Execute python script
uq --coverage-percentile 50 --input-data-path "data.csv" --output-data-path "out.csv"
if [ $? -eq 0 ]; then
    echo "Uncertainty quantification completed"
    echo "Plotting results..."
else
    echo "Uncertainty quantification failed"
    echo "Recovering from failure..."
fi
