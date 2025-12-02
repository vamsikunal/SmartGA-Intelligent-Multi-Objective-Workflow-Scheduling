import torch
import torch.nn as nn
import numpy as np
import os

class PolicyNetwork(nn.Module):
    """
    Policy Network matching the architecture in checkpoint_ep1700.pth
    Input: 75 features
    Hidden: 128 -> 128
    Output: 12 logits (one per node slot)
    """
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        
        # Feature extraction layer (75 -> 128)
        self.feature_net = nn.Sequential(
            nn.Linear(75, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Policy head (128 -> 12)
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12)
        )
        
    def forward(self, x):
        features = self.feature_net(x)
        logits = self.policy_head(features)
        return logits

class RLAgent:
    def __init__(self, checkpoint_path, device='cpu'):
        self.device = device
        self.model = PolicyNetwork().to(device)
        
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if 'policy_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['policy_state_dict'])
                else:
                    # Fallback if state dict structure is different
                    self.model.load_state_dict(checkpoint)
                print(f"✅ RL Agent loaded from {checkpoint_path}")
            except Exception as e:
                print(f"⚠️ Failed to load RL checkpoint: {e}")
                print("Using initialized weights (random behavior)")
        else:
            print(f"⚠️ Checkpoint not found at {checkpoint_path}")
            
        self.model.eval()
        
    def get_state(self, task, nodes, env_makespan=0):
        """
        Construct 75-dim state vector.
        Task (3) + 12 * Node (6) = 75
        """
        # Task Features (3)
        # 1. Instruction Count (normalized approx)
        inst_norm = task.instruction_count / 1e9 
        # 2. Data Size (approx from parents or input)
        data_size = getattr(task, 'data_size_in', 0) / 1e6 # MB
        # 3. Number of parents
        num_parents = len(task.parents)
        
        task_features = [inst_norm, data_size, num_parents]
        
        # Node Features (12 * 6 = 72)
        node_features = []
        
        # We need exactly 12 node slots. 
        # If actual nodes < 12, pad with zeros.
        # If actual nodes > 12, take first 12 (limitation of fixed input size)
        
        max_nodes = 12
        
        for i in range(max_nodes):
            if i < len(nodes):
                node = nodes[i]
                # 1. MIPS (normalized)
                mips_norm = node.mips / 10000.0
                # 2. Cost (normalized)
                cost_norm = node.cost_per_sec * 3600 # per hour
                # 3. Power (normalized)
                power_norm = node.power_active / 200.0
                # 4. Failure Rate
                fail_norm = node.failure_rate * 100
                # 5. Available Time (relative to current makespan/env time)
                # Since we generate schedule offline, we use estimated finish time
                avail_time = node.available_time
                # 6. Is Cloud (1 or 0)
                is_cloud = 1.0 if "Cloud" in node.__class__.__name__ else 0.0
                
                node_features.extend([mips_norm, cost_norm, power_norm, fail_norm, avail_time, is_cloud])
            else:
                # Padding for missing nodes
                node_features.extend([0.0] * 6)
                
        state = np.array(task_features + node_features, dtype=np.float32)
        return torch.FloatTensor(state).to(self.device)

    def select_action(self, state, valid_mask):
        """
        Select node index based on policy.
        valid_mask: boolean array of valid node indices (length 12)
        """
        with torch.no_grad():
            logits = self.model(state.unsqueeze(0)) # Add batch dim
            
            # Mask invalid actions (nodes that don't exist)
            # Set logits of invalid actions to -inf
            if valid_mask is not None:
                # Ensure mask is tensor
                mask_tensor = torch.BoolTensor(valid_mask).to(self.device)
                logits[0][~mask_tensor] = -float('inf')
            
            # Greedy selection (argmax)
            action = torch.argmax(logits, dim=1).item()
            return action

    def generate_schedule(self, tasks, graph, nodes):
        """
        Generate a complete schedule using the RL policy.
        """
        import networkx as nx
        
        # Reset node states for simulation
        # We need a temporary copy or just reset their available_time tracking
        # Since we are just generating a schedule list, we can track avail_time locally
        
        node_avail_times = [0.0] * len(nodes)
        
        topo_order = list(nx.topological_sort(graph))
        schedule = []
        
        # Valid mask for 12 output slots
        # First len(nodes) are True, rest False
        valid_mask = [False] * 12
        for i in range(len(nodes)):
            valid_mask[i] = True
            
        for task_id in topo_order:
            task = tasks[task_id]
            
            # Construct state
            # We need to pass node objects with updated available_time
            # Let's temporarily update node objects (and revert later if needed, or just use copies)
            # Actually, RLAgent.get_state reads node.available_time. 
            # We should update it as we go to simulate the greedy construction.
            
            # Save original times to restore later
            original_times = [n.available_time for n in nodes]
            
            # Update nodes with local tracking
            for i, n in enumerate(nodes):
                n.available_time = node_avail_times[i]
                
            state = self.get_state(task, nodes)
            
            # Restore original times (to not mess up external state)
            for i, n in enumerate(nodes):
                n.available_time = original_times[i]
            
            # Select action
            node_idx = self.select_action(state, valid_mask)
            
            # If model selects invalid node (shouldn't happen with masking), fallback
            if node_idx >= len(nodes):
                node_idx = 0 # Fallback to first node
            
            node_id = nodes[node_idx].node_id
            schedule.append((task, node_id))
            
            # Update local tracking
            # Estimate exec time
            exec_time = task.instruction_count / nodes[node_idx].mips
            # Simple availability update (ignoring comms for state estimation to keep it fast)
            node_avail_times[node_idx] += exec_time
            
        return schedule
