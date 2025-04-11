#!/usr/bin/env python3

import torch
import argparse
import os
import numpy as np
import random
from atari_wrappers import make_atari, wrap_deepmind
from cnn_policy import CnnPolicy
from ppo import PPO


def set_seed(seed):
    """Set all seeds for reproducibility"""
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(env_id, num_timesteps, seed, log_dir=None, checkpoint_path=None):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Create environment
    env = make_atari(env_id)
    env = wrap_deepmind(env, frame_stack=True)
    obs, _ = env.reset(seed=seed)  # Use new reset API
    print(f"Environment {env_id} created with observation space: {env.observation_space}, action space: {env.action_space}")

    # Create the log directory if it doesn't exist
    if log_dir is None:
        log_dir = f"./logs/{env_id.replace('/', '_')}"
  
    # Print TensorBoard launch instructions
    print(f"\nTraining with TensorBoard logging enabled.")
    print(f"To view training metrics, run:")
    print(f"  tensorboard --logdir={log_dir}/tensorboard")
    print(f"Then open http://localhost:6006 in your browser.\n")
    
    # Create policy
    policy = CnnPolicy(name="ppo_policy", ob_space=env.observation_space, ac_space=env.action_space)
    policy = policy.to(device)
    
    batch_size = 2048
    # Create PPO agent with visualization
    ppo_agent = PPO(
        policy=policy,
        env=env,
        device=device,
        clip_param=0.1,
        ppo_epochs=4,
        batch_size=batch_size,
        gamma=0.99,
        gae_lambda=0.95,
        value_coef=0.5,
        entropy_coef=0.01,
        lr=3e-4,
        max_grad_norm=0.5,
        log_dir=log_dir
    )
    
    # Load checkpoint if provided
    start_timestep = 0
    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            ppo_agent.policy.load_state_dict(checkpoint['policy_state_dict'])
            ppo_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Get the timestep from the checkpoint
            if 'timestep' in checkpoint:
                start_timestep = checkpoint['timestep']
                print(f"Resuming training from timestep {start_timestep}")
                
            print("Checkpoint loaded successfully")
        else:
            print(f"Warning: Checkpoint file {checkpoint_path} not found, starting from scratch")
    
    # Train the agent
    rewards = ppo_agent.train(
        total_timesteps=num_timesteps, 
        log_interval=batch_size,  
        evaluation_interval=100000,  #Evaluate & save video every 100k steps
        start_timestep=start_timestep  # Pass the start timestep to learn method
    )
    
    env.close()
    return rewards


def main():
    parser = argparse.ArgumentParser(description='PPO Atari')
    parser.add_argument('--env', type=str, default='ALE/Pong-v5', help='environment ID')
    parser.add_argument('--num_timesteps', type=int, default=10000000, help='number of timesteps to train')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--log_dir', type=str, default=None, help='directory to save model and logs')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to checkpoint file to continue training')
    args = parser.parse_args()
    
    train(args.env, args.num_timesteps, args.seed, args.log_dir, args.checkpoint)


if __name__ == '__main__':
    main()