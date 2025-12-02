"""
Workflow Generator for Training Data

Generates synthetic workflows for RL agent training.
"""

import networkx as nx
import numpy as np
from typing import Dict, Tuple, List


class WorkflowGenerator:
    """
    Generates synthetic workflows for training.
    """
    
    def __init__(self, workflow_types=['montage', 'cybershake', 'sipht'],
                 size_range=(50, 200)):
        """
        Args:
            workflow_types: List of workflow types to generate
            size_range: (min_size, max_size) for number of tasks
        """
        self.workflow_types = workflow_types
        self.size_range = size_range
    
    def generate(self):
        """
        Generate random workflow.
        
        Returns:
            tasks: Dict[task_id] -> {instruction_count, data_size}
            graph: NetworkX DiGraph
        """
        # Random workflow type
        workflow_type = np.random.choice(self.workflow_types)
        
        # Random size
        size = np.random.randint(self.size_range[0], self.size_range[1])
        
        # Generate based on type
        if workflow_type == 'montage':
            return self._generate_montage(size)
        elif workflow_type == 'cybershake':
            return self._generate_cybershake(size)
        elif workflow_type == 'sipht':
            return self._generate_sipht(size)
        else:
            return self._generate_random_dag(size)
    
    def _generate_montage(self, size):
        """Generate Montage-like workflow (pipeline with parallelism)."""
        graph = nx.DiGraph()
        tasks = {}
        
        # Create stages: projection, background, overlap, fit
        stages = 4
        tasks_per_stage = size // stages
        
        task_id = 0
        prev_stage = []
        
        for stage in range(stages):
            current_stage = []
            
            for i in range(tasks_per_stage):
                # Task properties
                tasks[task_id] = {
                    'instruction_count': np.random.randint(10000, 100000),
                    'data_size': np.random.randint(1000, 10000)
                }
                
                graph.add_node(task_id)
                
                # Add dependencies to previous stage
                if prev_stage:
                    parent = np.random.choice(prev_stage)
                    graph.add_edge(parent, task_id)
                
                current_stage.append(task_id)
                task_id += 1
            
            prev_stage = current_stage
        
        return tasks, graph
    
    def _generate_cybershake(self, size):
        """Generate CyberShake-like workflow (embarrassingly parallel with merge)."""
        graph = nx.DiGraph()
        tasks = {}
        
        # Entry task
        tasks[0] = {
            'instruction_count': np.random.randint(5000, 20000),
            'data_size': np.random.randint(500, 2000)
        }
        graph.add_node(0)
        
        # Parallel tasks
        parallel_tasks = []
        for i in range(1, size - 1):
            tasks[i] = {
                'instruction_count': np.random.randint(50000, 150000),
                'data_size': np.random.randint(5000, 15000)
            }
            graph.add_node(i)
            graph.add_edge(0, i)
            parallel_tasks.append(i)
        
        # Exit task (merge)
        exit_id = size - 1
        tasks[exit_id] = {
            'instruction_count': np.random.randint(20000, 50000),
            'data_size': np.random.randint(10000, 30000)
        }
        graph.add_node(exit_id)
        
        for task in parallel_tasks:
            graph.add_edge(task, exit_id)
        
        return tasks, graph
    
    def _generate_sipht(self, size):
        """Generate SIPHT-like workflow (fork-join pattern)."""
        graph = nx.DiGraph()
        tasks = {}
        
        # Create fork-join structure
        num_forks = max(2, size // 20)
        tasks_per_fork = size // num_forks
        
        task_id = 0
        
        # Entry
        tasks[task_id] = {
            'instruction_count': np.random.randint(5000, 15000),
            'data_size': np.random.randint(500, 1500)
        }
        graph.add_node(task_id)
        entry = task_id
        task_id += 1
        
        join_tasks = []
        
        for fork in range(num_forks):
            fork_tasks = []
            
            for i in range(tasks_per_fork):
                tasks[task_id] = {
                    'instruction_count': np.random.randint(20000, 80000),
                    'data_size': np.random.randint(2000, 8000)
                }
                graph.add_node(task_id)
                graph.add_edge(entry, task_id)
                fork_tasks.append(task_id)
                task_id += 1
            
            # Join task for this fork
            tasks[task_id] = {
                'instruction_count': np.random.randint(10000, 30000),
                'data_size': np.random.randint(1000, 5000)
            }
            graph.add_node(task_id)
            
            for ft in fork_tasks:
                graph.add_edge(ft, task_id)
            
            join_tasks.append(task_id)
            task_id += 1
        
        # Final merge
        if task_id < size:
            tasks[task_id] = {
                'instruction_count': np.random.randint(15000, 40000),
                'data_size': np.random.randint(3000, 10000)
            }
            graph.add_node(task_id)
            
            for jt in join_tasks:
                graph.add_edge(jt, task_id)
        
        return tasks, graph
    
    def _generate_random_dag(self, size):
        """Generate random DAG."""
        graph = nx.DiGraph()
        tasks = {}
        
        # Create nodes
        for i in range(size):
            tasks[i] = {
                'instruction_count': np.random.randint(10000, 100000),
                'data_size': np.random.randint(1000, 10000)
            }
            graph.add_node(i)
        
        # Create edges (ensure DAG property)
        for i in range(size):
            # Each task can have 0-3 children
            num_children = np.random.randint(0, 4)
            
            for _ in range(num_children):
                # Child must have higher ID (ensures DAG)
                if i < size - 1:
                    child = np.random.randint(i + 1, size)
                    graph.add_edge(i, child)
        
        return tasks, graph


def create_workflow_generator_function():
    """
    Create a workflow generator function for the environment.
    
    Returns:
        Function that generates (tasks, graph) tuples
    """
    generator = WorkflowGenerator()
    return generator.generate
