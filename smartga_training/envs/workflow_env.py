"""
Workflow Scheduling RL Environment

Gymnasium-compatible environment for training RL agents to schedule workflow tasks.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple


class WorkflowSchedulingEnv(gym.Env):
    """
    RL Environment for Workflow Scheduling.
    
    State: Task features + Node features + Graph features
    Action: Select node ID for current task
    Reward: Negative weighted sum of objectives (makespan, cost, energy, etc.)
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, workflow_generator, num_nodes=12):
        """
        Args:
            workflow_generator: Function that generates (tasks, graph) tuples
            num_nodes: Number of available compute nodes
        """
        super(WorkflowSchedulingEnv, self).__init__()
        
        self.workflow_generator = workflow_generator
        self.num_nodes = num_nodes
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(num_nodes)
        
        # State space: task(6) + nodes(5*12=60) + graph(4) + schedule(5) = 75 dims
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(75,), 
            dtype=np.float32
        )
        
        # Node characteristics (Cloud vs Edge)
        self.nodes = self._create_nodes()
        
        # Episode state
        self.tasks = None
        self.graph = None
        self.tasks_to_schedule = None
        self.current_task_idx = 0
        self.schedule = []
        self.scheduled_tasks = set()
        
    def _create_nodes(self):
        """Create heterogeneous nodes (2 Cloud + 10 Edge)."""
        nodes = []
        
        # Cloud nodes (powerful, expensive)
        for i in range(2):
            nodes.append({
                'id': i,
                'type': 'cloud',
                'mips': 10000,
                'cost': 0.5,
                'reliability': 0.9999,
                'energy_rate': 300,
                'current_load': 0.0
            })
        
        # Edge nodes (weaker, cheaper)
        for i in range(2, 12):
            nodes.append({
                'id': i,
                'type': 'edge',
                'mips': 2000 + np.random.randint(-500, 500),
                'cost': 0.05 + np.random.rand() * 0.05,
                'reliability': 0.99 + np.random.rand() * 0.009,
                'energy_rate': 50 + np.random.randint(-10, 10),
                'current_load': 0.0
            })
        
        return nodes
    
    def reset(self):
        """Start a new episode with a new workflow."""
        # Generate new workflow
        self.tasks, self.graph = self.workflow_generator()
        
        # Topological sort for scheduling order
        self.tasks_to_schedule = list(nx.topological_sort(self.graph))
        
        # Reset episode state
        self.current_task_idx = 0
        self.schedule = []
        self.scheduled_tasks = set()
        
        # Reset node loads
        for node in self.nodes:
            node['current_load'] = 0.0
        
        return self._get_state()
    
    def step(self, action):
        """
        Execute one scheduling decision.
        
        Args:
            action: Node ID to assign current task
            
        Returns:
            next_state, reward, done, info
        """
        # Get current task
        task_id = self.tasks_to_schedule[self.current_task_idx]
        task = self.tasks[task_id]
        
        # Assign task to selected node
        node = self.nodes[action]
        self.schedule.append((task_id, node['id']))
        self.scheduled_tasks.add(task_id)
        
        # Update node load (simple approximation)
        exec_time = task['instruction_count'] / node['mips']
        node['current_load'] += exec_time
        
        # Move to next task
        self.current_task_idx += 1
        
        # Check if episode is done
        done = (self.current_task_idx >= len(self.tasks_to_schedule))
        
        # Compute reward
        if done:
            metrics = self._evaluate_schedule()
            reward = self._compute_reward(metrics)
        else:
            reward = 0.0  # Sparse reward (only at end)
        
        next_state = self._get_state()
        
        info = {'metrics': metrics} if done else {}
        
        return next_state, reward, done, info
    
    def _get_state(self):
        """Encode current state as numpy array."""
        if self.current_task_idx >= len(self.tasks_to_schedule):
            # Terminal state
            return np.zeros(75, dtype=np.float32)
        
        task_id = self.tasks_to_schedule[self.current_task_idx]
        task = self.tasks[task_id]
        
        # Task features (6)
        task_features = np.array([
            task['instruction_count'] / 100000,  # Normalize
            task['data_size'] / 10000,
            len(list(self.graph.predecessors(task_id))),
            len(list(self.graph.successors(task_id))),
            self._get_task_depth(task_id) / 10.0,
            self._get_upward_rank(task_id) / 1000.0
        ], dtype=np.float32)
        
        # Node features (5 * 12 = 60)
        node_features = []
        for node in self.nodes:
            node_features.extend([
                node['mips'] / 10000,
                node['cost'],
                node['reliability'],
                node['current_load'] / 100.0,
                node['energy_rate'] / 300
            ])
        node_features = np.array(node_features, dtype=np.float32)
        
        # Graph features (4)
        graph_features = np.array([
            len(self.scheduled_tasks) / len(self.tasks),
            (len(self.tasks) - len(self.scheduled_tasks)) / len(self.tasks),
            self._get_critical_path_length() / 1000.0,
            self._get_avg_parallelism() / 10.0
        ], dtype=np.float32)
        
        # Schedule state (5)
        if len(self.schedule) > 0:
            partial_metrics = self._evaluate_schedule()
            schedule_features = np.array([
                partial_metrics['makespan'] / 1000.0,
                partial_metrics['cost'] / 100.0,
                partial_metrics['energy'] / 10000.0,
                partial_metrics['reliability'],
                partial_metrics['load_variance']
            ], dtype=np.float32)
        else:
            schedule_features = np.zeros(5, dtype=np.float32)
        
        # Concatenate all features
        state = np.concatenate([task_features, node_features, graph_features, schedule_features])
        
        return state
    
    def _evaluate_schedule(self):
        """Evaluate current (possibly partial) schedule."""
        if len(self.schedule) == 0:
            return {
                'makespan': 0.0,
                'cost': 0.0,
                'energy': 0.0,
                'reliability': 1.0,
                'load_variance': 0.0
            }
        
        # Simple evaluation (approximation)
        node_finish_times = {i: 0.0 for i in range(self.num_nodes)}
        task_finish_times = {}
        total_cost = 0.0
        total_energy = 0.0
        reliability = 1.0
        
        for task_id, node_id in self.schedule:
            task = self.tasks[task_id]
            node = self.nodes[node_id]
            
            # Compute start time (after dependencies)
            start_time = node_finish_times[node_id]
            for pred_id in self.graph.predecessors(task_id):
                if pred_id in task_finish_times:
                    start_time = max(start_time, task_finish_times[pred_id])
            
            # Compute finish time
            exec_time = task['instruction_count'] / node['mips']
            finish_time = start_time + exec_time
            
            # Update
            node_finish_times[node_id] = finish_time
            task_finish_times[task_id] = finish_time
            
            # Accumulate metrics
            total_cost += node['cost'] * exec_time
            total_energy += node['energy_rate'] * exec_time
            reliability *= node['reliability']
        
        makespan = max(node_finish_times.values()) if node_finish_times else 0.0
        load_variance = np.var(list(node_finish_times.values()))
        
        return {
            'makespan': makespan,
            'cost': total_cost,
            'energy': total_energy,
            'reliability': reliability,
            'load_variance': load_variance
        }
    
    def _compute_reward(self, metrics):
        """
        Compute reward from metrics.
        
        Multi-objective reward (scalarized with weights).
        """
        # Normalize metrics
        norm_makespan = metrics['makespan'] / 100.0  # Typical max ~100
        norm_cost = metrics['cost'] / 50.0            # Typical max ~50
        norm_energy = metrics['energy'] / 5000.0      # Typical max ~5000
        norm_reliability = 1.0 - metrics['reliability']  # Penalty for low reliability
        norm_load_var = metrics['load_variance'] / 100.0
        
        # Weighted sum (negative = minimize)
        reward = -(
            0.30 * norm_makespan +
            0.25 * norm_cost +
            0.20 * norm_energy +
            0.15 * norm_reliability +
            0.10 * norm_load_var
        )
        
        return reward
    
    # Helper methods
    def _get_task_depth(self, task_id):
        """Get depth of task in DAG."""
        try:
            return nx.shortest_path_length(self.graph, 
                                          source=list(nx.topological_sort(self.graph))[0],
                                          target=task_id)
        except:
            return 0
    
    def _get_upward_rank(self, task_id):
        """Approximate upward rank (HEFT-style)."""
        task = self.tasks[task_id]
        avg_exec_time = task['instruction_count'] / 5000  # Average MIPS
        
        max_successor_rank = 0
        for succ_id in self.graph.successors(task_id):
            max_successor_rank = max(max_successor_rank, self._get_upward_rank(succ_id))
        
        return avg_exec_time + max_successor_rank
    
    def _get_critical_path_length(self):
        """Approximate critical path length."""
        try:
            return nx.dag_longest_path_length(self.graph)
        except:
            return len(self.tasks)
    
    def _get_avg_parallelism(self):
        """Average parallelism in workflow."""
        if len(self.tasks) == 0:
            return 0.0
        
        # Count tasks at each depth level
        depths = {}
        for task_id in self.tasks:
            depth = self._get_task_depth(task_id)
            depths[depth] = depths.get(depth, 0) + 1
        
        return np.mean(list(depths.values())) if depths else 1.0
    
    def render(self, mode='human'):
        """Render environment state."""
        if mode == 'human':
            print(f"Task {self.current_task_idx}/{len(self.tasks_to_schedule)}")
            print(f"Scheduled: {len(self.schedule)} tasks")
