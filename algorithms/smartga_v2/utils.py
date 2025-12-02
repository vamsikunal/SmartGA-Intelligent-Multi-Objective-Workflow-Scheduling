from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

@dataclass
class ResourceConfig:
    """Configuration for computational resources."""
    num_processors: int
    processor_speeds: List[float]
    network_bandwidth: List[float]
    
    @classmethod
    def from_nodes(cls, nodes: List[Any]):
        """Create from list of ResourceNode objects."""
        speeds = [node.mips for node in nodes]
        # Simplified bandwidth model (assuming uniform or taking from first link)
        bandwidth = [1000.0] # Default
        return cls(
            num_processors=len(nodes),
            processor_speeds=speeds,
            network_bandwidth=bandwidth
        )

class Workflow:
    """Workflow representation compatible with SmartGA V2."""
    def __init__(self, tasks: List[Any], graph: Any):
        # Map string IDs to integer indices
        self.id_map = {t.task_id: i for i, t in enumerate(tasks)}
        self.reverse_map = {i: t.task_id for i, t in enumerate(tasks)}
        
        # Store tasks by integer ID
        self.tasks = {i: t for i, t in enumerate(tasks)}
        self.graph = graph
        
        # Ensure tasks have necessary attributes for V2
        for i, t in self.tasks.items():
            # Update predecessors/successors to use integer IDs
            if not hasattr(t, 'predecessors'):
                preds = list(graph.predecessors(t.task_id))
                t.predecessors = [self.id_map[p] for p in preds]
            else:
                # If they exist but are strings, map them
                t.predecessors = [self.id_map[p] if p in self.id_map else p for p in t.predecessors]
                
            if not hasattr(t, 'successors'):
                succs = list(graph.successors(t.task_id))
                t.successors = [self.id_map[s] for s in succs]
            else:
                t.successors = [self.id_map[s] if s in self.id_map else s for s in t.successors]

            if not hasattr(t, 'avg_cost'):
                t.avg_cost = t.instruction_count # Simplified if not present
                
        # Communication costs matrix (Task x Task)
        num_tasks = len(tasks)
        self.communication_costs = np.zeros((num_tasks, num_tasks))
        
        # Populate communication costs from graph edges
        for u, v, data in graph.edges(data=True):
            if u in self.id_map and v in self.id_map:
                u_idx = self.id_map[u]
                v_idx = self.id_map[v]
                weight = data.get('weight', 0)
                self.communication_costs[u_idx, v_idx] = weight
