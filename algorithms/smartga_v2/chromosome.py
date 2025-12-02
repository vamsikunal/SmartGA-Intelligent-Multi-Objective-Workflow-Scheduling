"""
Chromosome representation for workflow scheduling GA.

A chromosome encodes a complete schedule as a sequence of VM assignments,
where chromosome.genes[i] = j means task i is assigned to VM j.
"""

from typing import List, Tuple, Optional
import random
from dataclasses import dataclass


@dataclass
class Chromosome:
    """
    Represents a solution (schedule) as a chromosome.
    
    Attributes:
        genes: List of VM assignments, where genes[task_id] = vm_id
        fitness: Fitness value (lower is better for makespan)
        is_valid: Whether this chromosome represents a valid schedule
    """
    genes: List[int]
    fitness: float = float('inf')
    is_valid: bool = True
    
    def __post_init__(self):
        """Validate chromosome after initialization."""
        if not self.genes:
            raise ValueError("Chromosome genes cannot be empty")
    
    @classmethod
    def from_schedule(cls, schedule: dict, num_tasks: int) -> 'Chromosome':
        """
        Create a chromosome from a schedule dictionary.
        
        Args:
            schedule: Dict mapping task_id -> (vm_id, start_time, end_time)
            num_tasks: Total number of tasks in workflow
            
        Returns:
            Chromosome with genes encoding the VM assignments
        """
        genes = [0] * num_tasks
        for task_id, (vm_id, _, _) in schedule.items():
            genes[task_id] = vm_id
        return cls(genes=genes)
    
    @classmethod
    def random(cls, num_tasks: int, num_vms: int) -> 'Chromosome':
        """
        Create a random valid chromosome.
        
        Args:
            num_tasks: Number of tasks in workflow
            num_vms: Number of VMs available
            
        Returns:
            Random chromosome with valid VM assignments
        """
        genes = [random.randint(0, num_vms - 1) for _ in range(num_tasks)]
        return cls(genes=genes)
    
    def decode(self, workflow, resource_config) -> Tuple[dict, float]:
        """
        Decode chromosome into a schedule using list scheduling.
        
        This uses a simple list scheduling approach:
        1. Topologically sort tasks by dependencies
        2. For each task, assign to the VM specified by chromosome
        3. Schedule at earliest available time respecting dependencies
        
        Args:
            workflow: Workflow object with tasks and dependencies
            resource_config: Resource configuration with VM speeds
            
        Returns:
            Tuple of (schedule_dict, makespan) where:
            - schedule_dict maps task_id -> (vm_id, start_time, end_time)
            - makespan is the total completion time
        """
        import numpy as np
        
        # Get topological order
        topo_order = self._topological_sort(workflow)
        
        # Track when each VM becomes available
        vm_available = [0.0] * len(resource_config.processor_speeds)
        
        # Track when each task finishes (for dependency checking)
        task_finish = {}
        
        # Build schedule
        schedule = {}
        
        for task_id in topo_order:
            vm_id = self.genes[task_id]
            
            # Calculate execution time on assigned VM
            task = workflow.tasks[task_id]
            # Use computation_costs if available, otherwise use average cost
            if hasattr(task, 'computation_costs') and len(task.computation_costs) > vm_id:
                execution_time = task.computation_costs[vm_id]
            else:
                # Fallback: estimate from average cost and processor speed
                avg_cost = getattr(task, 'avg_cost', 10.0)
                execution_time = avg_cost / resource_config.processor_speeds[vm_id]
            
            # Find earliest start time (after dependencies and VM available)
            earliest_start = vm_available[vm_id]
            
            # Check dependencies
            predecessors = task.predecessors if hasattr(task, 'predecessors') else []
            for parent_id in predecessors:
                if parent_id in task_finish:
                    # Add communication cost if on different VM
                    parent_vm = schedule[parent_id][0]
                    comm_cost = 0.0
                    if parent_vm != vm_id:
                        # Get communication cost from matrix
                        if hasattr(workflow, 'communication_costs') and isinstance(workflow.communication_costs, np.ndarray):
                            comm_data = workflow.communication_costs[parent_id, task_id]
                            # Use network bandwidth (assuming uniform for simplicity)
                            bandwidth = resource_config.network_bandwidth[0] if len(resource_config.network_bandwidth) > 0 else 100.0
                            comm_cost = comm_data / bandwidth
                    earliest_start = max(earliest_start, task_finish[parent_id] + comm_cost)
            
            # Schedule task
            start_time = earliest_start
            end_time = start_time + execution_time
            schedule[task_id] = (vm_id, start_time, end_time)
            
            # Update tracking
            vm_available[vm_id] = end_time
            task_finish[task_id] = end_time
        
        # Calculate makespan
        if schedule:
            makespan = max(end_time for _, _, end_time in schedule.values())
        else:
            makespan = 0.0
        
        return schedule, makespan
    
    def validate(self, workflow) -> bool:
        """
        Check if chromosome represents a valid schedule.
        
        A chromosome is valid if:
        1. All genes are within valid VM range
        2. All task dependencies can be satisfied
        
        Args:
            workflow: Workflow object with tasks and dependencies
            
        Returns:
            True if chromosome is valid, False otherwise
        """
        # Check all genes are non-negative integers
        if any(gene < 0 for gene in self.genes):
            return False
        
        # Check chromosome length matches workflow
        if len(self.genes) != len(workflow.tasks):
            return False
        
        # Dependencies will be satisfied by decode() which uses topological sort
        # Just check we have all tasks
        return True
    
    def repair(self, num_vms: int, workflow) -> 'Chromosome':
        """
        Repair an invalid chromosome to make it valid.
        
        Fixes:
        1. Out-of-range VM assignments -> clip to [0, num_vms-1]
        2. Wrong length -> pad or trim to match workflow
        
        Args:
            num_vms: Number of available VMs
            workflow: Workflow object
            
        Returns:
            New repaired chromosome
        """
        repaired_genes = self.genes.copy()
        
        # Fix length
        num_tasks = len(workflow.tasks)
        if len(repaired_genes) < num_tasks:
            # Pad with random VMs
            repaired_genes.extend([
                random.randint(0, num_vms - 1) 
                for _ in range(num_tasks - len(repaired_genes))
            ])
        elif len(repaired_genes) > num_tasks:
            # Trim excess
            repaired_genes = repaired_genes[:num_tasks]
        
        # Fix out-of-range VMs
        repaired_genes = [
            max(0, min(gene, num_vms - 1))
            for gene in repaired_genes
        ]
        
        return Chromosome(genes=repaired_genes, is_valid=True)
    
    def _topological_sort(self, workflow) -> List[int]:
        """
        Topologically sort tasks by dependencies.
        
        Args:
            workflow: Workflow object with tasks and dependencies
            
        Returns:
            List of task IDs in topological order
        """
        from collections import deque
        
        # Build adjacency list and in-degree count
        graph = {task_id: [] for task_id in workflow.tasks}
        in_degree = {task_id: 0 for task_id in workflow.tasks}
        
        # Build graph from task predecessors/successors
        for task_id, task in workflow.tasks.items():
            if hasattr(task, 'successors'):
                for succ_id in task.successors:
                    graph[task_id].append(succ_id)
            if hasattr(task, 'predecessors'):
                in_degree[task_id] = len(task.predecessors)
        
        # Find all tasks with no dependencies
        queue = deque([task_id for task_id, deg in in_degree.items() if deg == 0])
        
        # Process tasks in order
        topo_order = []
        while queue:
            task_id = queue.popleft()
            topo_order.append(task_id)
            
            # Reduce in-degree for children
            for child_id in graph[task_id]:
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)
        
        # Check for cycles
        if len(topo_order) != len(workflow.tasks):
            raise ValueError("Workflow contains cycles")
        
        return topo_order
    
    def copy(self) -> 'Chromosome':
        """Create a deep copy of this chromosome."""
        return Chromosome(
            genes=self.genes.copy(),
            fitness=self.fitness,
            is_valid=self.is_valid
        )
    
    def __len__(self) -> int:
        """Return number of genes (tasks)."""
        return len(self.genes)
    
    def __eq__(self, other) -> bool:
        """Check equality based on genes."""
        if not isinstance(other, Chromosome):
            return False
        return self.genes == other.genes
    
    def __hash__(self) -> int:
        """Hash based on genes for use in sets/dicts."""
        return hash(tuple(self.genes))
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Chromosome(genes={self.genes[:5]}{'...' if len(self.genes) > 5 else ''}, fitness={self.fitness:.2f})"
