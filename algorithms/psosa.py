import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple

from .pso_sa.pso_sa import PSOSAScheduler, ResourceConfig

class PSOSA:
    """
    Adapter for PSO-SA to fit into the existing experiment framework.
    """
    def __init__(self, tasks, graph, nodes, pop_size=20, iterations=100):
        self.tasks_input = tasks
        self.graph = graph
        self.nodes = nodes
        self.pop_size = pop_size
        self.iterations = iterations
        
        # Handle task input (dict or list)
        if isinstance(tasks, dict):
            self.task_list = list(tasks.values())
        else:
            self.task_list = tasks
            
        # Map task IDs to integers
        self.id_map = {t.task_id: i for i, t in enumerate(self.task_list)}
        self.reverse_map = {i: t.task_id for i, t in enumerate(self.task_list)}
        
        # Prepare data for PSO-SA
        self.n = len(self.task_list)
        self.p = len(nodes)
        
        # 1. Computation Cost Matrix W (n x p)
        self.W = np.zeros((self.n, self.p))
        for i, task in enumerate(self.task_list):
            for j, node in enumerate(nodes):
                # Calculate execution time: instructions / mips
                self.W[i][j] = task.instruction_count / node.mips
                
        # 2. Communication Cost Matrix C (n x n)
        # C[i][j] should be communication time.
        # We use average bandwidth for simplification as PSO-SA model is static
        avg_bandwidth = np.mean([n.bandwidth for n in nodes]) if nodes else 1000.0
        
        self.C = np.zeros((self.n, self.n))
        for u, v, data in graph.edges(data=True):
            if u in self.id_map and v in self.id_map:
                u_idx = self.id_map[u]
                v_idx = self.id_map[v]
                data_size = data.get('weight', 0)
                self.C[u_idx][v_idx] = data_size / avg_bandwidth
                
        # 3. DAG and Predecessors
        self.dag = {i: [] for i in range(self.n)}
        self.predecessors = {i: [] for i in range(self.n)}
        
        for u, v in graph.edges():
            if u in self.id_map and v in self.id_map:
                u_idx = self.id_map[u]
                v_idx = self.id_map[v]
                self.dag[u_idx].append(v_idx)
                self.predecessors[v_idx].append(u_idx)
                
        # Initialize Scheduler
        self.scheduler = PSOSAScheduler(
            pop_size=pop_size,
            iterations=iterations,
            verbose=False
        )
        
    def run(self):
        """
        Run the algorithm and return the schedule.
        
        Returns:
            List of dicts (archive) or single schedule.
            The framework expects a list of schedules (even if just one).
        """
        workflow_tuple = (self.n, self.W, self.C, self.dag, self.predecessors)
        resources = ResourceConfig(num_processors=self.p)
        
        result = self.scheduler.schedule(workflow_tuple, resources)
        
        # Convert result to expected format: [(task, node_id), ...]
        schedule = []
        for task_idx, node_idx in result.task_assignments.items():
            task_id = self.reverse_map[task_idx]
            # Find task object
            task = next(t for t in self.task_list if t.task_id == task_id)
            # Find node ID
            node_id = self.nodes[node_idx].node_id
            
            schedule.append((task, node_id))
            
        # Return as list of schedules (PSO-SA returns single best)
        # But to be compatible with multi-objective return format (list of schedules), wrap it.
        # Wait, PAES returned an archive. PSO-SA is single objective (makespan) in this implementation?
        # Looking at pso_sa.py: _fitness_function returns makespan.
        # So it optimizes makespan.
        # run_full_experiment expects a list of schedules.
        
        return [schedule]
