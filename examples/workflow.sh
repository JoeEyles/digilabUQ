#!/usr/bin/env bash

# ============================================================
# Automated Workflow
# ============================================================

set -euo pipefail

# Activate the designated runtime environment
conda activate digilab
echo "Runtime environment activated"

# Execute python script
uq --coverage-percentile 50 --data-path "TODO"
echo "Uncertainty quantification completed"
