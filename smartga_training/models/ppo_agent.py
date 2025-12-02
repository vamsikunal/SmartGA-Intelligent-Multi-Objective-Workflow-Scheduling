"""
PPO Policy and Value Networks for SmartGA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PPOPolicy(nn.Module):
    """
    PPO Policy Network (Actor).
    
    Takes state and outputs action probabilities.
    """
    
    def __init__(self, state_dim=75, action_dim=12, hidden_dim=128):
        super(PPOPolicy, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state):
        """
        Forward pass.
        
        Args:
            state: (batch_size, state_dim) tensor
            
        Returns:
            action_probs: (batch_size, action_dim) probabilities
        """
        features = self.feature_net(state)
        logits = self.policy_head(features)
        action_probs = F.softmax(logits, dim=-1)
        
        return action_probs
    
    def get_action(self, state, deterministic=False):
        """
        Sample action from policy.
        
        Args:
            state: numpy array or tensor
            deterministic: If True, return argmax. If False, sample.
            
        Returns:
            action: int
            log_prob: log probability of action
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0)
        
        # Move to same device as model
        device = next(self.parameters()).device
        state = state.to(device)
        
        with torch.no_grad():
            action_probs = self.forward(state)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1).item()
            log_prob = torch.log(action_probs[0, action])
        else:
            dist = torch.distributions.Categorical(action_probs)
            action_tensor = dist.sample()
            action = action_tensor.item()
            log_prob = dist.log_prob(action_tensor)
        
        return action, log_prob.item()


class PPOValue(nn.Module):
    """
    PPO Value Network (Critic).
    
    Estimates state value V(s).
    """
    
    def __init__(self, state_dim=75, hidden_dim=128):
        super(PPOValue, self).__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        """
        Forward pass.
        
        Args:
            state: (batch_size, state_dim) tensor
            
        Returns:
            value: (batch_size, 1) state value
        """
        value = self.value_net(state)
        return value


class PPOAgent:
    """
    Complete PPO agent with policy and value networks.
    """
    
    def __init__(self, state_dim=75, action_dim=12, lr=3e-4, device='cuda'):
        self.device = device
        
        # Networks
        self.policy = PPOPolicy(state_dim, action_dim).to(device)
        self.value = PPOValue(state_dim).to(device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)
        
        # Hyperparameters
        self.clip_ratio = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.entropy_coef = 0.01
        self.value_coef = 0.5
    
    def select_action(self, state):
        """Select action using current policy."""
        return self.policy.get_action(state, deterministic=False)
    
    def compute_gae(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: List of rewards
            values: List of state values
            dones: List of done flags
            
        Returns:
            advantages: GAE advantages
            returns: TD(λ) returns
        """
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, advantages, returns, epochs=10):
        """
        PPO update step.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            old_log_probs: Old log probabilities
            advantages: Advantages
            returns: Returns
            epochs: Number of update epochs
        """
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(epochs):
            # Policy update
            action_probs = self.policy(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            policy_loss = policy_loss - self.entropy_coef * entropy
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()
            
            # Value update
            values = self.value(states).squeeze()
            value_loss = F.mse_loss(values, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
            self.value_optimizer.step()
    
    def save(self, path):
        """Save model weights."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model weights."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
