#!/bin/bash
# Script to activate virtual environment and start Jupyter

echo "=== ACTIVATING VIRTUAL ENVIRONMENT ==="
source venv/bin/activate

echo "Virtual environment activated!"
echo "Python path: $(which python)"

echo ""
echo "=== STARTING JUPYTER NOTEBOOK ==="
echo "Jupyter will start on http://localhost:8888"
echo "Press Ctrl+C to stop Jupyter"
echo ""

jupyter notebook --no-browser --port=8888
