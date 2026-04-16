#!/bin/bash
#SBATCH --job-name=dreamerv3_train
#SBATCH --account=eecs545w26_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

echo "=== Job info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

REPO_DIR="/scratch/eecs545w26_class_root/eecs545w26_class/jtyzhang/dreamerv3_carla"
CARLA_DIR="/scratch/eecs545w26_class_root/eecs545w26_class/jtyzhang/CARLA_0.9.15"  
PORT=2000

cd "$REPO_DIR"
mkdir -p logs

# Conda in batch shell - activate environment
eval "$(conda shell.bash hook)"
conda activate eecs545proj

# Device check
echo "=== Python / CUDA checks ==="
which python
python -V
nvidia-smi || true

python - << 'PY'
import torch
print("torch:", torch.__version__)
print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

# # Start CARLA headless in background
# cd "$CARLA_DIR"
# ./CarlaUE4.sh -RenderOffScreen -quality-level=Low -nosound -carla-port=${PORT} > "$REPO_DIR/logs/carla-${SLURM_JOB_ID}.log" 2>&1 &
# CARLA_PID=$!
# echo "Started CARLA PID=$CARLA_PID"

# # initial buffer for CARLA to start
# sleep 300

# # Wait for CARLA port to be ready
# echo "Waiting for CARLA on 127.0.0.1:${PORT}..."
# for i in {1..300}; do
#   python - <<'PY'
# import socket, sys
# s = socket.socket()
# s.settimeout(1.0)
# ok = (s.connect_ex(("127.0.0.1", 2000)) == 0)
# s.close()
# sys.exit(0 if ok else 1)
# PY
#   if [ $? -eq 0 ]; then
#     echo "CARLA is ready on 127.0.0.1:2000"
#     break
#   fi
#   echo "Waiting for CARLA... attempt $i"
#   sleep 5
# done

# Run training
cd "$REPO_DIR"
python -m train > "$REPO_DIR/logs/train-${SLURM_JOB_ID}.log" 2>&1

# Cleanup
kill $CARLA_PID || true
wait $CARLA_PID || true

echo "Finished: $(date)"