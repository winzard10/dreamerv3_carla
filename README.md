/rl_car
├── /models
│   ├── encoder.py       # Visual (Depth/Seg) + Vector fusion
│   ├── rssm.py          # Recurrent State-Space Model (The World Model)
│   ├── actor_critic.py  # Policy and Value networks
│   └── decoder.py       # Reconstruction for training images
├── /env
│   ├── carla_wrapper.py # Gym-like interface for CARLA 0.9.x
│   └── data_collect.py  # Script for gathering initial PID driving data
├── /utils
│   ├── buffer.py        # Replay buffer for latent states
│   └── logger.py        # TensorBoard/WandB integration
├── train.py             # Main training loop (Imagination + Real)
└── evaluate.py          # OOD testing (Town 02, Rain, etc.)

# Activate venv
source venv_rl/bin/activate

# Runs in low-quality mode to save VRAM and power
./carla_sim/CarlaUE4.sh -quality-level=Low
# Runs off-screen
./carla_sim/CarlaUE4.sh -RenderOffScreen

./CarlaUE4.sh -quality-level=Low -benchmark -fps=20

# Check GPU status
watch -n 1 nvidia-smi

# Run data collect
python3 -m env.data_collect

# Train
python3 -m train

# Test
python3 -m test

# Tensorboard
tensorboard --logdir=runs