import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, 0.01)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight, 1.0)
        m.bias.data.fill_(0.0)


class CnnPolicy(nn.Module):
    recurrent = False
    
    def __init__(self, name, ob_space, ac_space, kind='large'):
        super(CnnPolicy, self).__init__()
        self.name = name
        self._init(ob_space, ac_space, kind)
        
    def _init(self, ob_space, ac_space, kind):
        assert isinstance(ob_space, gym.spaces.Box)
        
        num_actions = ac_space.n
        
        # CNN layers
        if kind == 'small':  # from A3C paper
            self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
            self.fc = nn.Linear(32 * 9 * 9, 256)
            self.logits = nn.Linear(256, num_actions)
            self.value = nn.Linear(256, 1)
        elif kind == 'large':  # Nature DQN
            self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
            self.fc = nn.Linear(64 * 7 * 7, 512)
            self.logits = nn.Linear(512, num_actions)
            self.value = nn.Linear(512, 1)
        else:
            raise NotImplementedError
        
        # Apply weight initialization
        self.apply(init_weights)
        
    def forward(self, x):
        # Input shape: [batch_size, 4, 84, 84]
        x = x / 255.0
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        if hasattr(self, 'conv3'):
            x = F.relu(self.conv3(x))
            
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))
        
        logits = self.logits(x)
        value = self.value(x).squeeze(-1)
        
        return logits, value
    
    def act(self, obs, deterministic=False):
        # Convert numpy array to torch tensor
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
            
        # Add batch dimension if needed
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
            
        # Move to the correct device
        device = next(self.parameters()).device
        obs = obs.to(device)
        
        # Forward pass
        with torch.no_grad():
            logits, value = self.forward(obs)
            
            # Get action
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                dist = Categorical(logits=logits)
                action = dist.sample()
                
        return action.cpu().numpy()[0], value.cpu().numpy()[0]
    
    def get_value(self, obs):
        # Convert numpy array to torch tensor
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
            
        # Add batch dimension if needed
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
            
        # Move to the correct device
        device = next(self.parameters()).device
        obs = obs.to(device)
        
        # Forward pass
        with torch.no_grad():
            _, value = self.forward(obs)
                
        return value.cpu().numpy()[0]
    
    def get_initial_state(self):
        return []