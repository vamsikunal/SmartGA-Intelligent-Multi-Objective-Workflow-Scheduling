"""
SmartGA RL Training Script - Google Colab Compatible
=====================================================

This script trains a PPO agent to learn workflow scheduling policies.
It's optimized to run on Google Colab (free GPU).

Usage in Colab:
1. Upload this entire smartga_training/ folder to Colab
2. Install requirements: !pip install -r requirements.txt
3. Run: !python train_rl_colab.py

The script will:
- Train for 2000 episodes (~2-3 hours on Colab GPU)
- Save model to smartga_pretrained.pth
- Generate training plots
- Download model when done
"""

import sys
import os
import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.workflow_env import WorkflowSchedulingEnv
from models.ppo_agent import PPOAgent
from data.workflow_generator import create_workflow_generator_function
from utils.training_utils import Logger, plot_training_progress, compute_statistics


# Training Configuration
CONFIG = {
    'num_episodes': 2000,        # Reduced for Colab (was 10000 in plan)
    'num_nodes': 12,              # 2 Cloud + 10 Edge
    'state_dim': 75,
    'action_dim': 12,
    'learning_rate': 3e-4,
    'ppo_epochs': 10,
    'save_interval': 100,         # Save checkpoint every 100 episodes
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def train():
    """Main training loop."""
    
    print("="*70)
    print("SmartGA RL Training - PPO Agent for Workflow Scheduling")
    print("="*70)
    print(f"Device: {CONFIG['device']}")
    print(f"Episodes: {CONFIG['num_episodes']}")
    print(f"Learning Rate: {CONFIG['learning_rate']}")
    print("="*70 + "\n")
    
    # Create environment
    workflow_gen = create_workflow_generator_function()
    env = WorkflowSchedulingEnv(workflow_gen, num_nodes=CONFIG['num_nodes'])
    
    # Create agent
    agent = PPOAgent(
        state_dim=CONFIG['state_dim'],
        action_dim=CONFIG['action_dim'],
        lr=CONFIG['learning_rate'],
        device=CONFIG['device']
    )
    
    # Logger
    logger = Logger('smartga_training.log')
    
    # Training loop
    episode_rewards = []
    
    for episode in tqdm(range(CONFIG['num_episodes']), desc="Training"):
        # Collect trajectory
        states, actions, rewards, log_probs, dones, values = [], [], [], [], [], []
        
        state = env.reset()
        episode_reward = 0
        step = 0
        done = False
        
        while not done:
            # Select action
            action, log_prob = agent.select_action(state)
            
            # Get value
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(CONFIG['device'])
                value = agent.value(state_tensor).item()
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(done)
            values.append(value)
            
            state = next_state
            episode_reward += reward
            step += 1
        
        # Compute advantages and returns
        advantages, returns = agent.compute_gae(rewards, values, dones)
        
        # Update agent
        agent.update(states, actions, log_probs, advantages, returns, 
                    epochs=CONFIG['ppo_epochs'])
        
        # Log progress
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            logger.log_episode(episode + 1, episode_reward, avg_reward)
        
        # Save checkpoint
        if (episode + 1) % CONFIG['save_interval'] == 0:
            checkpoint_path = f'checkpoint_ep{episode+1}.pth'
            agent.save(checkpoint_path)
            print(f"\n[Checkpoint saved: {checkpoint_path}]\n")
    
    # Final save
    final_model_path = 'smartga_pretrained.pth'
    agent.save(final_model_path)
    
    print("\n" + "="*70)
    print(f"✅ Training Complete! Model saved to: {final_model_path}")
    print("="*70 + "\n")
    
    # Generate plots
    plot_training_progress(episode_rewards, 'training_progress.png')
    
    # Print statistics
    compute_statistics(episode_rewards)
    
    # Colab-specific: Trigger download
    try:
        from google.colab import files
        print("📥 Downloading model file to your computer...")
        files.download(final_model_path)
        files.download('training_progress.png')
        print("✅ Download complete!")
    except ImportError:
        print("💾 Model saved locally (not running in Colab)")
    
    return agent


def test_pretrained_model(model_path='smartga_pretrained.pth', num_test_episodes=10):
    """
    Test the pretrained model.
    
    Args:
        model_path: Path to model file
        num_test_episodes: Number of test episodes
    """
    print("\n" + "="*70)
    print("Testing Pretrained Model")
    print("="*70 + "\n")
    
    # Create environment
    workflow_gen = create_workflow_generator_function()
    env = WorkflowSchedulingEnv(workflow_gen, num_nodes=CONFIG['num_nodes'])
    
    # Create and load agent
    agent = PPOAgent(
        state_dim=CONFIG['state_dim'],
        action_dim=CONFIG['action_dim'],
        device=CONFIG['device']
    )
    agent.load(model_path)
    
    test_rewards = []
    
    for episode in range(num_test_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Use policy deterministically (argmax)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(CONFIG['device'])
                action_probs = agent.policy(state_tensor)
                action = torch.argmax(action_probs, dim=-1).item()
            
            state, reward, done, info = env.step(action)
            episode_reward += reward
        
        test_rewards.append(episode_reward)
        
        if 'metrics' in info:
            print(f"Episode {episode+1}:")
            print(f"  Reward: {episode_reward:.4f}")
            print(f"  Metrics: {info['metrics']}")
            print()
    
    print(f"\nTest Results (n={num_test_episodes}):")
    print(f"  Mean Reward: {np.mean(test_rewards):.4f} ± {np.std(test_rewards):.4f}")
    print()


if __name__ == "__main__":
    # Train the agent
    agent = train()
    
    # Test the trained model
    test_pretrained_model('smartga_pretrained.pth', num_test_episodes=5)
    
    print("\n🎉 All done! You can now use 'smartga_pretrained.pth' in your main project.")
