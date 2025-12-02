"""
WOA - Whale Optimization Algorithm for Workflow Scheduling

Implements the Whale Optimization Algorithm (Mirjalili & Lewis, 2016)
adapted for discrete workflow scheduling problems.

Key Features:
- Encircling prey (exploitation)
- Bubble-net attacking (exploitation)
- Search for prey (exploration)
- Single-objective optimization (makespan)
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple

class WOA:
    """
    Whale Optimization Algorithm for workflow scheduling.
    
    Optimizes task-to-processor assignments to minimize makespan.
    """
    
    def __init__(self, tasks, graph, nodes, pop_size=30, iterations=50):
        """
        Initialize WOA scheduler.
        
        Args:
            tasks: Dictionary or list of task objects
            graph: NetworkX DAG
            nodes: List of ResourceNode objects
            pop_size: Population size (number of whales)
            iterations: Maximum iterations
        """
        # Handle task input
        if isinstance(tasks, dict):
            self.task_list = list(tasks.values())
        else:
            self.task_list = tasks
            
        self.graph = graph
        self.nodes = nodes
        self.pop_size = pop_size
        self.iterations = iterations
        
        # Get topological order
        self.task_order = list(nx.topological_sort(graph))
        self.n_tasks = len(self.task_order)
        self.n_nodes = len(nodes)
        
        # Initialize population (positions)
        # Each whale position is a continuous vector [0, n_nodes-1]
        self.positions = np.random.uniform(0, self.n_nodes - 1, (pop_size, self.n_tasks))
        self.fitness = np.full(pop_size, np.inf)
        
        # Best solution found
        self.best_position = None
        self.best_fitness = np.inf
        
    def _evaluate(self, position: np.ndarray) -> float:
        """
        Evaluate a whale's position (schedule) by calculating makespan.
        
        Args:
            position: Continuous position vector
            
        Returns:
            Makespan (fitness to minimize)
        """
        from simulator.environment import CloudEdgeEnvironment
        
        # Convert continuous position to discrete processor assignments
        assignments = np.round(position).astype(int)
        assignments = np.clip(assignments, 0, self.n_nodes - 1)
        
        # Build schedule
        schedule = []
        for i, task_id in enumerate(self.task_order):
            # Find task object
            task = next(t for t in self.task_list if t.task_id == task_id)
            node_id = self.nodes[assignments[i]].node_id
            schedule.append((task, node_id))
        
        # Evaluate using environment
        cloud_nodes = [n for n in self.nodes if "Cloud" in n.__class__.__name__]
        edge_nodes = [n for n in self.nodes if "Edge" in n.__class__.__name__]
        
        env = CloudEdgeEnvironment(cloud_nodes, edge_nodes)
        
        try:
            metrics = env.run(schedule)
            return metrics['makespan']
        except:
            return 1e6
    
    def run(self) -> List[List[Tuple[Any, int]]]:
        """
        Execute WOA algorithm.
        
        Returns:
            List containing the best schedule found
        """
        # Evaluate initial population
        for i in range(self.pop_size):
            self.fitness[i] = self._evaluate(self.positions[i])
            
            # Update best
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_position = self.positions[i].copy()
        
        # Main WOA loop
        for t in range(self.iterations):
            # Linearly decrease a from 2 to 0
            a = 2 - t * (2.0 / self.iterations)
            
            # Update each whale
            for i in range(self.pop_size):
                # Random parameters
                r = np.random.random()
                A = 2 * a * r - a
                C = 2 * r
                
                l = np.random.uniform(-1, 1)
                p = np.random.random()
                
                # Update position
                if p < 0.5:
                    if abs(A) < 1:
                        # Encircling prey (exploitation)
                        D = abs(C * self.best_position - self.positions[i])
                        self.positions[i] = self.best_position - A * D
                    else:
                        # Search for prey (exploration)
                        rand_whale_idx = np.random.randint(0, self.pop_size)
                        X_rand = self.positions[rand_whale_idx]
                        D = abs(C * X_rand - self.positions[i])
                        self.positions[i] = X_rand - A * D
                else:
                    # Bubble-net attacking (exploitation)
                    D_prime = abs(self.best_position - self.positions[i])
                    self.positions[i] = D_prime * np.exp(l) * np.cos(2 * np.pi * l) + self.best_position
                
                # Clip to bounds
                self.positions[i] = np.clip(self.positions[i], 0, self.n_nodes - 1)
                
                # Evaluate new position
                self.fitness[i] = self._evaluate(self.positions[i])
                
                # Update best
                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_position = self.positions[i].copy()
        
        # Decode best solution to schedule
        assignments = np.round(self.best_position).astype(int)
        assignments = np.clip(assignments, 0, self.n_nodes - 1)
        
        schedule = []
        for i, task_id in enumerate(self.task_order):
            task = next(t for t in self.task_list if t.task_id == task_id)
            node_id = self.nodes[assignments[i]].node_id
            schedule.append((task, node_id))
        
        # Return as list of schedules (single solution)
        return [schedule]
