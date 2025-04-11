# PPO Atari Reinforcement Learning

This repository contains an implementation of Proximal Policy Optimization (PPO) for training agents to play Atari games. The implementation is based on PyTorch and uses Gymnasium (the successor to OpenAI Gym) for the Atari environment.

## Project Structure

- `atari_wrappers.py`: Contains wrapper classes for Atari environments to preprocess observations
- `cnn_policy.py`: Defines the CNN policy network architecture
- `ppo.py`: Core PPO algorithm implementation
- `train_torch.py`: Main training script
- `run.sh`: Convenience script to start training

## Installation

python 3.10 

Install dependencies:

```bash
pip install torch
pip install "gymnasium[atari,accept-rom-license]==0.29.1" # 版本太高不支持环境PongNoFrameskip-v4
pip install opencv-python
pip install moviepy==1.0.3 #版本太高报错
```
