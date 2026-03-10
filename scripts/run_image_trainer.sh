#!/bin/bash
set -e
# Start Redis server for state management
redis-server --daemonize yes
sleep 2  # Give Redis a moment to start up
echo "[run_image_trainer.sh] Starting trainer with args: $@"
python3 /workspace/scripts/image_trainer.py "$@"