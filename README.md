# DreamerV3 Autonomous Driving in CARLA

A world model-based reinforcement learning agent for autonomous driving, built on [DreamerV3](https://arxiv.org/abs/2301.04104) and trained in the [CARLA](https://carla.org/) simulator.

The agent learns a latent world model from multimodal observations (depth + semantic segmentation), then trains a driving policy entirely in imagination — minimizing real environment interaction.

---

## Requirements

- Ubuntu 22.04 LTS
- Python 3.10+
- CARLA 0.9.16
- NVIDIA GPU with 8GB+ VRAM (tested on RTX 4070 Laptop)
- CUDA 11.8+

Install CARLA 0.9.16 + Python dependencies:
```bash
pip install -r requirements.txt
```

For CARLA 0.9.15 + Python compatibility:
```bash
pip install -r requirements_0-9-15.txt
```

---

## Project Structure

```
rl_car/
├── models/
│   ├── encoder.py          # Multimodal encoder (depth + semantic + vector + goal)
│   ├── rssm.py             # Recurrent State-Space Model (world model dynamics)
│   ├── decoder.py          # Observation reconstruction heads
│   ├── actor_critic.py     # Policy (actor) and value (critic) networks
│   ├── rewardhead.py       # Distributional reward prediction
│   └── continuehead.py     # Episode continuation prediction
├── env/
│   ├── carla_wrapper.py    # Gym-compatible CARLA interface with reward function
│   └── data_collect.py     # Expert data collection using pure pursuit controller
├── utils/
│   ├── buffer.py           # Sequence replay buffer
│   ├── train_utils.py      # World model + actor-critic training steps
│   ├── test_utils.py       # Evaluation loop and model loading
│   ├── lambda_returns.py   # Dreamer-style λ-return computation
│   ├── twohot.py           # TwoHot distributional encoding
│   └── plotter.py          # Training curve generation
├── params.py               # All hyperparameters (single source of truth)
├── train.py                # Main training loop (Phase A + Phase B)
└── test.py                 # Closed-loop evaluation
```

---

## Quickstart

### 1. Start CARLA

```bash
# Standard (with rendering)
./carla_sim/CarlaUE4.sh -quality-level=Low

# Headless (recommended for training)
./carla_sim/CarlaUE4.sh -quality-level=Low -benchmark -fps=20
./carla_sim/CarlaUE4.sh -RenderOffScreen -benchmark -fps=20
```

### 2. Activate environment

```bash
source venv_rl/bin/activate
```

### 3. Collect expert data

```bash
python3 -m env.data_collect
```

Collects driving sequences using a pure pursuit expert controller. Data is saved to `./data/expert_sequences/`. Configure collection parameters in `params.py` (`COLLECT_TARGET_STEPS`, `COLLECT_LOOKAHEAD`, etc.).

### 4. Train

```bash
python3 -m train
```

Training runs in two phases:
- **Phase A** — World model pretraining on expert data (offline, ~20k steps)
- **Phase B** — Online RL with CARLA interaction (~2000 episodes)

Checkpoints are saved to `./checkpoints/`. Monitor training with TensorBoard:

```bash
tensorboard --logdir=runs
```

### 5. Evaluate

```bash
python3 -m test
```

Runs closed-loop evaluation across multiple towns. Configure `TEST_TOWN` and `TEST_NUM_EPISODES` in `params.py`. Reports average speed, travel distance, center distance, and episode reward.

---

## Key Hyperparameters

All hyperparameters are centralized in `params.py`. Key values:

| Parameter | Value |
|-----------|-------|
| Deterministic dim | 512 |
| Stochastic dim | 32 × 32 |
| Sequence length | 10 |
| Imagination horizon | 8 |
| Discount γ | 0.97 |
| World model LR | 8e-5 |
| Actor/Critic LR | 3e-5 |
| Target speed | 15 km/h |

---

## Monitoring

```bash
# GPU utilization
watch -n 1 nvidia-smi

# Training curves
tensorboard --logdir=runs

# Generate summary plots
python3 -m utils.plotter
```

Key metrics to watch in TensorBoard:
- `Train/Episode_Reward` — policy performance
- `Train/imag_return_mean` — imagination vs reality gap
- `Train/prior_entropy` — latent space health (target: 2.0–2.5)
- `Train/kl_loss` — world model quality

---