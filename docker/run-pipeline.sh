#!/bin/bash

# run this script by typing:
#   nohup ./run-pipeline.sh  > pipeline.log 2>&1 &

# Stop the pipeline if any step fails (returns error code)
set -e

# Make sure the child scripts are executable
chmod +x ./3dgs-colmap/run-container.sh
chmod +x ./3dgsworkspace/run-container.sh

echo "=========================================="
echo "Running COLMAP (Feature Matching)"
echo "=========================================="

./3dgs-colmap/run-container.sh

echo "=========================================="
echo "Running 3DGS Workspace (Training)"
echo "=========================================="

./3dgsworkspace/run-container.sh

echo "=========================================="
echo "Pipeline Finished Successfully"
echo "=========================================="
