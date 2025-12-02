# SmartGA Training - Google Colab Instructions

## 📦 Package Contents

This standalone package trains an **RL agent (PPO)** to learn workflow scheduling policies for SmartGA.

**Files**:
- `train_rl_colab.py` - Main training script (Colab-optimized)
- `envs/workflow_env.py` - RL environment (Gym interface)
- `models/ppo_agent.py` - PPO policy and value networks
- `data/workflow_generator.py` - Synthetic workflow generator
- `utils/training_utils.py` - Logging and plotting
- `requirements.txt` - Dependencies

---

## 🚀 Quick Start (Google Colab)

### Step 1: Upload to Colab

1. Zip this entire `smartga_training/` folder
   ```bash
   # On your local machine:
   cd "New Folder 1"
   zip -r smartga_training.zip smartga_training/
   ```

2. Open Google Colab: https://colab.research.google.com

3. Upload the zip file:
   ```python
   from google.colab import files
   uploaded = files.upload()  # Select smartga_training.zip
   ```

4. Unzip:
   ```bash
   !unzip smartga_training.zip
   %cd smartga_training
   ```

### Step 2: Install Dependencies

```bash
!pip install -q -r requirements.txt
```

### Step 3: Enable GPU

- In Colab menu: **Runtime → Change runtime type → Hardware accelerator → GPU (T4)**
- Click **Save**

### Step 4: Run Training

```bash
!python train_rl_colab.py
```

**Training Time**: ~2-3 hours on Colab T4 GPU (2000 episodes)

### Step 5: Download Model

The script automatically downloads:
- `smartga_pretrained.pth` - Trained model (use this in your main project!)
- `training_progress.png` - Training curves

---

## 📊 Training Configuration

**Default Settings** (optimized for Colab free tier):
- Episodes: 2000 (reduced from 10K for faster training)
- Workflow sizes: 50-200 tasks
- Workflow types: Montage, CyberShake, SIPHT
- Learning rate: 3e-4
- PPO epochs: 10 per update
- Device: Auto-detect (GPU if available)

**To modify**, edit `CONFIG` in `train_rl_colab.py`:
```python
CONFIG = {
    'num_episodes': 2000,    # Increase for better quality
    'learning_rate': 3e-4,   # Adjust if training unstable
    ...
}
```

---

## 📈 Expected Results

**Training Progress**:
```
Episode     0 | Reward:  -2.4500
Episode   100 | Reward:  -1.8200 | Avg(100): -2.0100
Episode   500 | Reward:  -0.9500 | Avg(100): -1.2300
Episode  1000 | Reward:  -0.6200 | Avg(100): -0.8100
Episode  2000 | Reward:  -0.4800 | Avg(100): -0.5500
```

**Final Metrics**:
- Mean Reward: -0.55 (typical)
- Higher is better (less negative = better schedules)
- Reward improves significantly over training

---

## 🔧 Troubleshooting

### Out of Memory Error
**Solution**: Reduce batch size
```python
# In train_rl_colab.py, add to CONFIG:
'ppo_epochs': 5,  # Reduce from 10
```

### Training Too Slow
**Solution**: 
1. Confirm GPU is enabled (check `Device: cuda` in output)
2. Reduce num_episodes to 1000 for quick test

### Import Errors
**Solution**: Ensure you're in the `smartga_training/` directory:
```bash
%cd smartga_training
!pwd  # Should show .../smartga_training
```

---

## 🎯 Using the Trained Model

After training, copy `smartga_pretrained.pth` to your main project:

```bash
# In your main project directory:
cp smartga_pretrained.pth ../algorithms/smartga_pretrained.pth
```

Then in your SmartGA algorithm:
```python
class SmartGA:
    def __init__(self, tasks, graph, nodes):
        # Load pretrained RL agent
        self.rl_agent = PPOAgent(...)
        self.rl_agent.load('algorithms/smartga_pretrained.pth')
    
    def initialize_population(self):
        # Use RL agent to generate high-quality population
        for i in range(70):
            schedule = self.rl_agent.generate_schedule(...)
            population.append(schedule)
        ...
```

---

## 📝 Advanced: Local Training (if you have GPU)

If you have a local GPU (RTX 3060+):

```bash
cd smartga_training/
pip install -r requirements.txt
python train_rl_colab.py
```

**For longer training** (better quality):
- Edit `CONFIG['num_episodes'] = 5000` in `train_rl_colab.py`
- Training time: 6-8 hours on RTX 3090

---

## 📚 Architecture Details

**RL Environment**:
- State: 75 dims (task features + node features + graph context)
- Action: Select node ID (0-11)
- Reward: Weighted multi-objective (makespan, cost, energy, reliability, load)

**PPO Agent**:
- Policy Network: 75 → 128 → 64 → 12 (softmax)
- Value Network: 75 → 128 → 64 → 1
- Optimizer: Adam (lr=3e-4)
- Clipping: ε=0.2

**Workflow Generation**:
- Types: Montage (pipeline), CyberShake (embarassingly parallel), SIPHT (fork-join)
- Sizes: 50-200 tasks (random)
- Diversity: Different patterns for generalization

---

## 🆘 Support

**Common Questions**:

**Q: Can I reduce training time?**  
A: Yes, set `num_episodes=1000` for quick test. Quality will be lower but still better than random.

**Q: Will this work on Colab free tier?**  
A: Yes! Optimized for T4 GPU (free tier). Training takes ~2-3 hours.

**Q: Can I resume training from checkpoint?**  
A: Yes, checkpoints saved every 100 episodes (`checkpoint_ep100.pth`, etc.). Load with:
```python
agent.load('checkpoint_ep500.pth')
```

**Q: How do I know if training is working?**  
A: Reward should increase (less negative) over time. Check `training_progress.png`.

---

## 📄 Citation

If you use SmartGA in research, please cite:

```bibtex
@misc{smartga2024,
  title={SmartGA: RL-Pretrained Multi-Objective Genetic Algorithm},
  author={Your Team},
  year={2024}
}
```

---

## ✅ Checklist

Before running:
- [ ] Uploaded `smartga_training/` to Colab
- [ ] Installed dependencies (`!pip install -r requirements.txt`)
- [ ] Enabled GPU (Runtime → Change runtime type → GPU)
- [ ] Confirmed in correct directory (`%cd smartga_training`)

After training:
- [ ] Downloaded `smartga_pretrained.pth`
- [ ] Downloaded `training_progress.png`
- [ ] Copied model to main project

---

**Good luck! 🚀**
