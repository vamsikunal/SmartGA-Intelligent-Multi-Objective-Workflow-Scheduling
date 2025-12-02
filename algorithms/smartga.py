import networkx as nx
import numpy as np
from typing import List, Dict, Any, Tuple

# Import V2 components
from .smartga_v2.smart_ga import SmartGA as SmartGAV2
from .smartga_v2.utils import Workflow, ResourceConfig

class SmartGA:
    """
    Adapter for SmartGA V2 to fit into the existing experiment framework.
    """
    def __init__(self, tasks, graph, nodes, pop_size=50, generations=50):
        self.tasks = tasks
        self.graph = graph
        self.nodes = nodes
        self.pop_size = pop_size
        self.generations = generations
        
        # Convert to V2 data structures
        # If tasks is a dict (id -> task), convert to list of tasks
        if isinstance(tasks, dict):
            task_list = list(tasks.values())
        else:
            task_list = tasks
            
        self.workflow_v2 = Workflow(task_list, graph)
        self.resources_v2 = ResourceConfig.from_nodes(nodes)
        
        # Initialize V2 Algorithm
        self.algo = SmartGAV2(
            workflow=self.workflow_v2,
            resources=self.resources_v2,
            pop_size=pop_size,
            use_heuristic_seeding=True,
            use_adaptive_params=True
        )
        
    def schedule(self) -> List[List[Tuple[Any, int]]]:
        """
        Run the algorithm and return a list of schedules (Pareto front).
        
        Returns:
            List of schedules, where each schedule is a list of (task, node_id) tuples.
        """
        # Run optimization
        result = self.algo.run(num_generations=self.generations, verbose=False)
        
        # Extract Pareto front
        # result['pareto_front'] is list of (chromosome, objectives)
        pareto_front = result['pareto_front']
        
        schedules = []
        self.pareto_front_objectives = []
        
        for chrom, objectives in pareto_front:
            # Store objectives
            self.pareto_front_objectives.append(objectives)
            
            # Decode chromosome to V2 schedule dict: task_id -> (vm_id, start, end)
            schedule_dict, _ = chrom.decode(self.workflow_v2, self.resources_v2)
            
            # Convert to expected format: [(task_obj, node_id), ...]
            formatted_schedule = []
            for task_id, (vm_idx, _, _) in schedule_dict.items():
                # Find the original task object
                task = self.workflow_v2.tasks[task_id]
                
                # Map vm_idx back to node_id
                # ResourceConfig.from_nodes created processor_speeds from nodes list
                # So vm_idx 0 corresponds to nodes[0]
                node_id = self.nodes[vm_idx].node_id
                
                formatted_schedule.append((task, node_id))
            
            schedules.append(formatted_schedule)
            
        return schedules
