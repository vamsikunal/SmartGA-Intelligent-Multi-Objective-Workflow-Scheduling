"""
Training utilities for SmartGA
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List


def plot_training_progress(episode_rewards: List[float], save_path='training_progress.png'):
    """
    Plot training progress.
    
    Args:
        episode_rewards: List of episode rewards
        save_path: Where to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.3, label='Raw')
    
    # Moving average
    window = 100
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_rewards)), moving_avg, 
                label=f'Moving Avg ({window} eps)', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Distribution
    plt.subplot(1, 2, 2)
    plt.hist(episode_rewards, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training plot saved to {save_path}")


def compute_statistics(rewards: List[float]):
    """Compute and print training statistics."""
    rewards = np.array(rewards)
    
    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)
    print(f"Total Episodes:     {len(rewards)}")
    print(f"Mean Reward:        {rewards.mean():.4f}")
    print(f"Std Reward:         {rewards.std():.4f}")
    print(f"Min Reward:         {rewards.min():.4f}")
    print(f"Max Reward:         {rewards.max():.4f}")
    
    # Last 100 episodes
    if len(rewards) >= 100:
        last_100 = rewards[-100:]
        print(f"\nLast 100 Episodes:")
        print(f"  Mean:             {last_100.mean():.4f}")
        print(f"  Std:              {last_100.std():.4f}")
    
    print("="*60 + "\n")


class Logger:
    """Simple logger for training."""
    
    def __init__(self, log_file='training.log'):
        self.log_file = log_file
        self.episode_rewards = []
    
    def log(self, message):
        """Log message to console and file."""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def log_episode(self, episode, reward, avg_reward=None):
        """Log episode results."""
        msg = f"Episode {episode:5d} | Reward: {reward:8.4f}"
        if avg_reward is not None:
            msg += f" | Avg(100): {avg_reward:8.4f}"
        
        self.log(msg)
        self.episode_rewards.append(reward)
    
    def get_rewards(self):
        """Get all episode rewards."""
        return self.episode_rewards
