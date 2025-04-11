import torch
import os
from cnn_policy import CnnPolicy
from atari_wrappers import make_atari, wrap_deepmind
import gymnasium as gym

model_path = './logs/PongNoFrameskip-v4/checkpoint_12100000.pt'
env_id = 'PongNoFrameskip-v4'
video_dir = './logs/PongNoFrameskip-v4/videos_eval'

# Load the trained model
checkpoint = torch.load(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = make_atari(env_id, render_mode='rgb_array')
env = wrap_deepmind(env, frame_stack=True)

# Add recording wrapper after all preprocessing
env = gym.wrappers.RecordVideo(
    env, 
    video_dir,
    episode_trigger=lambda x: True
)

# Create and load policy
policy = CnnPolicy("ppo_policy", env.observation_space, env.action_space)
policy.load_state_dict(checkpoint['policy_state_dict'])
policy = policy.to(device)

# Run evaluation
obs, _ = env.reset()
done = False
total_reward = 0
action_list = []

while not done:
    state = obs.to_torch(device)
            
    with torch.no_grad():
        logits, _ = policy(state.unsqueeze(0))
        action = torch.argmax(logits, dim=1).to('cpu').numpy()[0]
        action_list.append(action)
        
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward
    

print(f"Actions taken: {action_list}")
print(f"Evaluation reward: {total_reward}")

env.close()