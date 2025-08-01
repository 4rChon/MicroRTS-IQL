# IQL MicroRTS - Implicit Q-Learning for Real-Time Strategy Games

A PyTorch implementation of Implicit Q-Learning (IQL) for training reinforcement learning agents in the [MicroRTS environment](https://github.com/Farama-Foundation/MicroRTS-Py). This project is based on the [CORL IQL implementation](https://github.com/corl-team/CORL/blob/main/algorithms/offline/iql.py).

## Overview

This repository implements Implicit Q-Learning, an offline reinforcement learning algorithm that learns from pre-collected data without requiring online interaction with the environment. The implementation is specifically designed for the MicroRTS real-time strategy game environment.

## Features

- **MicroRTS Integration**: Custom gym environment for real-time strategy gameplay
- **IQL Algorithm**: State-of-the-art offline reinforcement learning
- **Spatial Networks**: CNN-based state and action encoders for grid-based observations
- **Experiment Tracking**: Built-in Wandb logging and experiment management
- **Docker Support**: Containerized training with GPU acceleration
- **Replay Buffer**: Efficient LMDB-based data storage and loading
- **Configurable**: YAML-based hyperparameter configuration

## Installation

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended)
- Docker (optional, for containerized training)

### Local Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd iql
```

2. **Install Poetry (if not already installed):**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. **Install dependencies:**
```bash
poetry install
```

4. **Build the MicroRTS environment:**
```bash
cd lib/gym_microrts
chmod +x build.sh
./build.sh
```

### Docker Setup

1. **Build the Docker image:**
```bash
docker build -t iql-microrts .
```

2. **Run with Docker Compose:**
```bash
# Create .env file with required environment variables
echo "WANDB_API_KEY=your_wandb_api_key_here" > .env
echo "HTTPS_PROXY=your_https_proxy_url" >> .env
```


# Run training
docker-compose up
```

## Usage

### Training

The main training script accepts the following arguments:

```bash
python src/main.py --note "experiment_description" --config config/hyperparams_iql.yaml
```

**Arguments:**
- `--note`: Description of the experiment (required)
- `--config`: Path to hyperparameter configuration file (default: `../config/hyperparams_iql.yaml`)

### Configuration

Create a YAML configuration file with the following structure:

```yaml
seed: 42
debug: False
device: cuda
render: True

eval: {
  eval: True,
  rollouts_num: 100,
  seed_count: 4,
}

environment: {
  save_dir: _exp,
  project: PhD,
  group: iql,
  episode_steps_max: 2000,
}

data: {
  buffer_path: "../data/8x8/1v1/transitions/train",
  save_interval: 100000,
  log_interval: 10,
  num_workers: 12,
}

iql: {
  model: {
    dropout: 0.1,
    hidden_dim: 256,
    hidden_layers: 2,
    load: True,
    load_path: "./iql_model.pt",
  },
  training: {
    train: False,
    iql_tau: 0.1,
    tau: 0.005,
    beta: 4.0,
    discount: 0.99,
    max_timesteps: 800000,
    vf_lr: 3e-4,
    qf_lr: 3e-4,
    actor_lr: 3e-4,
    batch_size: 32,
    reward_scale: 1.0,
    warmup_pct: 0.45,
  },
  eval: {
    eval_freq: 10000,
    episodes_num: 10,
    longer_episodes_num: 100,
  }
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- CORL IQL implementation: [CORL](https://github.com/corl-team/CORL/blob/main/algorithms/offline/iql.py)
- MicroRTS environment: [gym-microrts](https://github.com/Farama-Foundation/MicroRTS-Py)
- IQL paper: [Implicit Q-Learning](https://arxiv.org/pdf/2110.06169.pdf)