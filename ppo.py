import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
from collections import deque
import matplotlib.pyplot as plt
import gymnasium as gym
import os
from moviepy.editor import ImageSequenceClip
from torch.utils.tensorboard import SummaryWriter
from atari_wrappers import make_atari, wrap_deepmind


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        
    def store(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return np.array(self.states), np.array(self.actions), np.array(self.probs), \
               np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches


class TrainingVisualizer:
    def __init__(self, env_id, log_dir='./logs'):
        self.log_dir = log_dir
        self.env_id = env_id
        
        # Create logging directory
        os.makedirs(log_dir, exist_ok=True)
        self.video_dir = os.path.join(log_dir, 'videos')
        os.makedirs(self.video_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))
        
        # Initialize data containers
        self.timesteps = []
        self.rewards = []
        self.avg_rewards = []
        self.fps_history = []
        self.episode_lengths = []
        
        # For rendering and recording videos
        self.render_frames = []
        self.evaluation_counter = 0

        # Add counters for termination and truncation
        self.terminate_count = 0
        self.truncated_count = 0
    
    def update_plots(self, episode, reward, avg_reward, timestep, ep_length):
        # Update data containers (keep these for reference/history)
        self.timesteps.append(timestep)
        self.rewards.append(reward)
        self.avg_rewards.append(avg_reward)
        self.episode_lengths.append(ep_length)
        
        # Log to TensorBoard (this is the only visualization we'll keep)
        self.writer.add_scalar('Training/Episode Reward', reward, episode)
        self.writer.add_scalar('Training/Average Reward (100 ep)', avg_reward, episode)
        self.writer.add_scalar('Training/Episode Length', ep_length, episode)
    
    def log_losses(self, actor_loss, critic_loss, entropy, timestep):
        """Log training losses to TensorBoard."""
        self.writer.add_scalar('Loss/Actor', actor_loss, timestep)
        self.writer.add_scalar('Loss/Critic', critic_loss, timestep)
        self.writer.add_scalar('Loss/Entropy', entropy, timestep)
    
    def log_model_gradients(self, model, timestep):
        """Log model parameter gradients to TensorBoard."""
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.writer.add_histogram(f'Gradients/{name}', param.grad.data.cpu().numpy(), timestep)
    
    def evaluate_and_render(self, policy, num_episodes=1, device='cpu', render_mode=None):
        """Evaluate the policy and optionally render."""
        self.evaluation_counter += 1
        
        # Create a test environment with rendering enabled
        test_env = make_atari(self.env_id, render_mode=render_mode)
        test_env = wrap_deepmind(test_env, frame_stack=True)
        
        # Add recording wrapper after all preprocessing
        test_env = gym.wrappers.RecordVideo(
            test_env, 
            os.path.join(self.video_dir, f'evaluation_{self.evaluation_counter}'),
            episode_trigger=lambda x: True
        )
        
        eval_rewards = []
        
        for _ in range(num_episodes):
            obs, _ = test_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Use LazyFrames to_torch method
                state = obs.to_torch(device)
                
                # Get action - add shape check before passing to policy
                
                with torch.no_grad():
                    logits, _ = policy(state.unsqueeze(0))
                    action = torch.argmax(logits, dim=1).cpu().numpy()[0]
                
                # Take action
                obs, reward, term, trunc, _ = test_env.step(action)
                done = term or trunc
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
            print(f"Evaluation episode reward: {episode_reward}")
        
        # Log evaluation metrics to TensorBoard
        avg_eval_reward = np.mean(eval_rewards)
        self.writer.add_scalar('Evaluation/Average Reward', avg_eval_reward, self.evaluation_counter)
        
        test_env.close()

    
    def close(self):
        """Close the TensorBoard writer when done."""
        self.writer.close()


class PPO:
    def __init__(self, policy, env, device, clip_param=0.2, ppo_epochs=4, 
                 batch_size=64, gamma=0.99, gae_lambda=0.95, value_coef=0.5, 
                 entropy_coef=0.01, lr=3e-4, max_grad_norm=0.5,
                 log_dir='./logs'):
        self.policy = policy
        self.env = env
        self.device = device
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy.to(self.device)
        
        self.memory = PPOMemory(batch_size)
        
        # Initialize visualizer
        env_id = env.unwrapped.spec.id if hasattr(env.unwrapped, 'spec') else "Unknown"
        self.visualizer = TrainingVisualizer(env_id, log_dir)
        
        # Global step counter for TensorBoard logging
        self.global_step = 0


    def train(self, total_timesteps, log_interval=100, evaluation_interval=10000, start_timestep=0):
        # Initialize environment and metrics
        obs, _ = self.env.reset()
        episode_rewards = []
        episode_lengths = []
        episode_reward = 0
        episode_length = 0
        start_time = time.time()
        
        # Set global step for resuming training
        self.global_step = start_timestep // self.batch_size if start_timestep > 0 else 0
        
        # Main training loop
        for timestep in range(start_timestep + 1, total_timesteps + 1):
            # Get action from policy
            state = obs.to_torch(self.device)
            with torch.no_grad():
                logits, value = self.policy(state.unsqueeze(0))
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                action_np = action.cpu().numpy()[0]
                log_prob_np = log_prob.cpu().numpy()
                value_np = value.cpu().numpy()
                
            # Execute action in environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            
            # Store experience
            self.memory.store(obs, action_np, log_prob_np, value_np, reward, done)
            
            # Update tracking metrics
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            # Periodic policy update
            if timestep % self.batch_size == 0:
                # --- BEGIN UPDATE LOGIC ---
                states, actions, old_probs, values, rewards, dones, batches = self.memory.generate_batches()
                
                # Convert dones to masks (1 - done)
                masks = 1 - dones
                
                # Calculate returns and advantages
                # Convert values list to numeric array and append next_value
                values_array = np.array(values, dtype=np.float32)
                values_array = np.append(values_array, 0.0)  # Next value is 0 at the end of a batch
                
                gae = 0
                returns = np.zeros(len(rewards), dtype=np.float32)
                advantages = np.zeros(len(rewards), dtype=np.float32)
                for step in reversed(range(len(rewards))):
                    delta = rewards[step] + self.gamma * values_array[step + 1] * masks[step] - values_array[step]
                    gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
                    returns[step] = gae + values_array[step]
                    advantages[step] = gae
                
                # Prepare tensors for processing
                returns_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)
                advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)
                advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)  # Normalize
                
                # Convert states to tensor with proper shape for CNN input
                states_tensor = torch.FloatTensor(np.transpose(np.array(states), (0, 3, 1, 2))).to(self.device)
                
                # Training metrics
                epoch_actor_loss = 0
                epoch_critic_loss = 0
                epoch_entropy = 0
                num_updates = len(batches) * self.ppo_epochs
                
                # PPO update loop
                for _ in range(self.ppo_epochs):
                    for batch in batches:
                        # --- BEGIN OPTIMIZATION STEP LOGIC ---
                        # Process batch data
                        batch_states = states_tensor[batch]
                        batch_actions = torch.tensor(actions[batch], dtype=torch.int64).to(self.device)
                        batch_old_probs = torch.tensor(old_probs[batch], dtype=torch.float32).to(self.device).reshape(-1)
                        batch_advantages = advantages_tensor[batch]
                        batch_returns = returns_tensor[batch]
                        
                        # Forward pass through neural network
                        logits, critic_value = self.policy(batch_states)
                        critic_value = critic_value.squeeze()
                        
                        # Create probability distribution
                        dist = torch.distributions.Categorical(logits=logits)
                        
                        # Get log probabilities of actions
                        new_probs = dist.log_prob(batch_actions)
                        
                        # Calculate policy ratio and surrogate objectives
                        ratio = torch.exp(new_probs - batch_old_probs)
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                        
                        # Calculate losses
                        entropy = dist.entropy().mean()
                        actor_loss = -torch.min(surr1, surr2).mean()
                        critic_loss = nn.MSELoss()(critic_value, batch_returns)
                        
                        # Combined loss
                        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                        
                        # Gradient step
                        self.optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        # --- END OPTIMIZATION STEP LOGIC ---
                        
                        # Track losses
                        epoch_actor_loss += actor_loss.item()
                        epoch_critic_loss += critic_loss.item()
                        epoch_entropy += entropy.item()
                
                # Log metrics
                self.global_step += 1
                self.visualizer.log_losses(
                    epoch_actor_loss / num_updates,
                    epoch_critic_loss / num_updates,
                    epoch_entropy / num_updates,
                    self.global_step
                )
                
                # Log gradients periodically
                if self.global_step % 10 == 0:
                    self.visualizer.log_model_gradients(self.policy, self.global_step)
                
                # Clear memory for next update
                self.memory.clear()
                # --- END UPDATE LOGIC ---
            
            # Handle episode completion
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Update visualization with rolling average
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                self.visualizer.update_plots(
                    len(episode_rewards), 
                    episode_reward, 
                    avg_reward, 
                    timestep, 
                    episode_length
                )
                
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            
            # Periodic evaluation and checkpoint saving
            if timestep % evaluation_interval == 0:
                print(f"\nEvaluating policy at timestep {timestep}...")
                self.visualizer.evaluate_and_render(
                    self.policy, 
                    num_episodes=1, 
                    device=self.device, 
                    render_mode='rgb_array'
                )
                
                # Save model checkpoint
                torch.save({
                    'policy_state_dict': self.policy.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'timestep': timestep,
                }, os.path.join(self.visualizer.log_dir, f'checkpoint_{timestep}.pt'))
            
            # Periodic logging
            if timestep % log_interval == 0:
                elapsed_time = time.time() - start_time
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                
                print(f"Timestep: {timestep}/{total_timesteps} | "
                      f"Episodes: {len(episode_rewards)} | "
                      f"Mean reward: {avg_reward:.2f} | "
                      f"Elapsed time: {elapsed_time:.2f}s")
        
        # Finalize training
        final_avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
        print(f"Training completed. Final performance: {final_avg_reward:.2f}")
        
        # Final evaluation
        self.visualizer.evaluate_and_render(
            self.policy, 
            num_episodes=3, 
            device=self.device, 
            render_mode='rgb_array'
        )
        
        # Save final model
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'timestep': total_timesteps,
            'final_reward': final_avg_reward
        }, os.path.join(self.visualizer.log_dir, 'final_model.pt'))
        
        # Close TensorBoard writer
        self.visualizer.close()
        
        return episode_rewards