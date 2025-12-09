#!/bin/bash
# Start Jupyter Lab server for remote access via Tailscale/SSH
# Access via: main-pc:8888 in your browser

# Change to LCAS project directory
cd "$(dirname "$0")"

# Activate virtual environment
source .venv/bin/activate

echo "========================================"
echo "Starting Jupyter Lab Server"
echo "========================================"
echo ""
echo "Access the notebook server at:"
echo "  http://main-pc:8888"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Jupyter Lab
# --ip=0.0.0.0: Bind to all network interfaces (allows Tailscale/SSH access)
# --port=8888: Use port 8888 (change if needed)
# --no-browser: Don't try to open a browser on the server
# --NotebookApp.token='': Disable token authentication (use with caution!)
#                         Remove this line if you want password protection
jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --NotebookApp.token='' \
    --NotebookApp.password=''
