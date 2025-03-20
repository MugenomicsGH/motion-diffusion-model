#!/bin/bash

# Enable conda command
eval "$(conda shell.bash hook)"

# Check if mdm environment exists
if ! conda env list | grep -q "^mdm "; then
    echo "Creating mdm conda environment..."
    conda env create -f environment.yml
fi

# Activate conda environment
conda activate mdm

# Ensure required packages are installed
pip install moviepy trimesh

# Start the API server
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload 