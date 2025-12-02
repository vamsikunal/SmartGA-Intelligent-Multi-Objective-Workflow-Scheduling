"""
Multi-objective fitness evaluation for NSGA-II.

Implements 4 objectives for workflow scheduling:
1. Makespan - Total execution time (minimize)
2. Cost - Resource usage cost (minimize)
3. Energy - Energy consumption (minimize)
4. Load Balance - VM utilization variance (minimize)
"""

from typing import List, Dict, Tuple
import numpy as np
from .chromosome import Chromosome
from .utils import Workflow
from .utils import ResourceConfig


class MultiObjectiveFitness:
    """
    Multi-objective fitness evaluator for NSGA-II.
    
    Calculates 4 objectives from a chromosome's decoded schedule:
    - Makespan: Completion time of last task
    - Cost: Total resource usage cost
    - Energy: Total energy consumption
    - Load Balance: Variance in VM utilization (lower is better)
    
    Example:
        >>> evaluator = MultiObjectiveFitness(workflow, resources)
        >>> objectives = evaluator.evaluate(chromosome)
        >>> print(objectives)  # (makespan, cost, energy, balance)
        (125.5, 450.2, 1200.0, 0.15)
    """
    
    def __init__(self, workflow: Workflow, resources: ResourceConfig):
        """
        Initialize multi-objective evaluator.
        
        Args:
            workflow: Workflow to schedule
            resources: Available computing resources with cost/energy data
        """
        self.workflow = workflow
        self.resources = resources
        self._evaluation_count = 0
        self._cache = {}  # Cache: chromosome hash -> objectives
    
    def evaluate(self, chromosome: Chromosome) -> Tuple[float, float, float, float]:
        """
        Evaluate all 4 objectives for a chromosome.
        
        Args:
            chromosome: Chromosome to evaluate
            
        Returns:
            Tuple of (makespan, cost, energy, load_balance)
            All objectives are minimization (lower is better)
        """
        # Check cache
        chrom_hash = hash(chromosome)
        if chrom_hash in self._cache:
            return self._cache[chrom_hash]
        
        # Decode chromosome to schedule
        result = chromosome.decode(self.workflow, self.resources)
        
        # Handle tuple return from decode
        if isinstance(result, tuple):
            schedule, makespan = result
        else:
            schedule = result
            if schedule:
                makespan = max(end_time for _, _, end_time in schedule.values())
            else:
                makespan = float('inf')
        
        # Calculate all objectives
        if not schedule or makespan == 0:
            # Invalid schedule
            objectives = (float('inf'), float('inf'), float('inf'), float('inf'))
        else:
            cost = self._calculate_cost(schedule)
            energy = self._calculate_energy(schedule)
            balance = self._calculate_load_balance(schedule)
            objectives = (makespan, cost, energy, balance)
        
        # Update cache and counter
        self._cache[chrom_hash] = objectives
        self._evaluation_count += 1
        
        return objectives
    
    def evaluate_batch(
        self, 
        chromosomes: List[Chromosome]
    ) -> List[Tuple[float, float, float, float]]:
        """
        Evaluate multiple chromosomes efficiently.
        
        Args:
            chromosomes: List of chromosomes to evaluate
            
        Returns:
            List of objective tuples
        """
        return [self.evaluate(chrom) for chrom in chromosomes]
    
    def _calculate_cost(self, schedule: Dict) -> float:
        """
        Calculate total resource usage cost.
        
        Cost = sum over all tasks of (execution_time × processor_cost)
        
        Args:
            schedule: Schedule dict mapping task_id -> (vm_id, start_time, end_time)
            
        Returns:
            Total cost
        """
        if not hasattr(self.resources, 'processor_costs'):
            return 0.0
        
        total_cost = 0.0
        for task_id, (vm_id, start_time, end_time) in schedule.items():
            execution_time = end_time - start_time
            vm_cost = self.resources.processor_costs[vm_id]
            total_cost += execution_time * vm_cost
        
        return total_cost
    
    def _calculate_energy(self, schedule: Dict) -> float:
        """
        Calculate total energy consumption.
        
        Energy = sum over all tasks of (execution_time × energy_consumption)
        
        Args:
            schedule: Schedule dict mapping task_id -> (vm_id, start_time, end_time)
            
        Returns:
            Total energy consumption
        """
        if not hasattr(self.resources, 'energy_consumption'):
            return 0.0
        
        total_energy = 0.0
        for task_id, (vm_id, start_time, end_time) in schedule.items():
            execution_time = end_time - start_time
            vm_energy = self.resources.energy_consumption[vm_id]
            total_energy += execution_time * vm_energy
        
        return total_energy
    
    def _calculate_load_balance(self, schedule: Dict) -> float:
        """
        Calculate load balance as variance in VM utilization.
        
        Lower variance = better balance (all VMs equally loaded)
        
        Args:
            schedule: Schedule dict mapping task_id -> (vm_id, start_time, end_time)
            
        Returns:
            Variance of VM utilization (0 = perfectly balanced)
        """
        # Calculate utilization for each VM
        num_vms = self.resources.num_processors
        vm_utilization = np.zeros(num_vms)
        
        # Get makespan (total time)
        if not schedule:
            return 0.0
        
        makespan = max(end_time for _, _, end_time in schedule.values())
        if makespan == 0:
            return 0.0
        
        # Sum execution time per VM
        for task_id, (vm_id, start_time, end_time) in schedule.items():
            execution_time = end_time - start_time
            vm_utilization[vm_id] += execution_time
        
        # Normalize by makespan to get utilization ratio
        vm_utilization = vm_utilization / makespan
        
        # Calculate variance
        variance = np.var(vm_utilization)
        
        return float(variance)
    
    def get_evaluation_count(self) -> int:
        """Get total number of evaluations performed."""
        return self._evaluation_count
    
    def reset_cache(self):
        """Clear the evaluation cache."""
        self._cache.clear()


class ObjectiveNormalizer:
    """
    Normalize objectives to [0, 1] range for fair comparison.
    
    Useful for visualization and when objectives have different scales.
    """
    
    def __init__(self):
        """Initialize normalizer with no data."""
        self.mins = None
        self.maxs = None
    
    def fit(self, objectives: List[Tuple[float, float, float, float]]):
        """
        Learn normalization parameters from objective data.
        
        Args:
            objectives: List of objective tuples
        """
        if not objectives:
            return
        
        # Convert to numpy array
        obj_array = np.array(objectives)
        
        # Calculate min/max for each objective
        self.mins = np.min(obj_array, axis=0)
        self.maxs = np.max(obj_array, axis=0)
        
        # Avoid division by zero
        self.maxs = np.where(self.maxs == self.mins, self.mins + 1e-10, self.maxs)
    
    def normalize(
        self, 
        objectives: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Normalize objectives to [0, 1] range.
        
        Args:
            objectives: Raw objective values
            
        Returns:
            Normalized objectives in [0, 1]
        """
        if self.mins is None or self.maxs is None:
            return objectives
        
        obj_array = np.array(objectives)
        normalized = (obj_array - self.mins) / (self.maxs - self.mins)
        
        return tuple(normalized)
    
    def denormalize(
        self, 
        normalized: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Convert normalized objectives back to original scale.
        
        Args:
            normalized: Normalized objective values in [0, 1]
            
        Returns:
            Original scale objectives
        """
        if self.mins is None or self.maxs is None:
            return normalized
        
        norm_array = np.array(normalized)
        original = norm_array * (self.maxs - self.mins) + self.mins
        
        return tuple(original)
