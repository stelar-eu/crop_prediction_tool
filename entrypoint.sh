#!/bin/bash
set -e

# Define or load the Conda environment name
CONDA_ENV=${CONDA_ENV:-stcon5}

# Initialize Conda and activate the environment
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda activate "$CONDA_ENV"

# Move to working directory (defined in Dockerfile as /app)
cd /app

# Execute the command passed to the container
exec "$@"