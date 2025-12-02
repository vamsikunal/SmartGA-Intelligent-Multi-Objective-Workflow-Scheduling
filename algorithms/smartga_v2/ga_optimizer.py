"""
GA Optimizer Component for MADGA Integration

This module provides a wrapper around the SmartGA that makes it compatible
with MADGA's action-based interface. It handles:
- Time budget constraints
- Pareto front to single solution conversion
- Quick termination for RL environment
- Multi-objective to single-objective mapping

Author: Month 5 Week 4
Date: 2024
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from .utils import Workflow
from .utils import ResourceConfig
from .smart_ga import SmartGA


class GAOptimizer:
    """
    GA Optimizer wrapper for MADGA integration.
    
    This class wraps SmartGA to provide a simple interface that the RL
    environment can call as an action. It handles time budgets, generation
    limits, and converts multi-objective Pareto fronts into single solutions.
    
    Attributes:
        pop_size: Population size for GA
        max_generations: Maximum generations to run
        use_heuristic_seeding: Whether to use heuristic seeding
        use_adaptive_params: Whether to use adaptive parameters
        heuristic_ratio: Ratio of heuristic solutions (0.0-1.0)
        time_budget: Maximum time in seconds (None = no limit)
        primary_objective: Objective to optimize ('makespan', 'energy', 'cost')
        verbose: Whether to print progress
    """
    
    def __init__(
        self,
        pop_size: int = 50,
        max_generations: int = 50,
        use_heuristic_seeding: bool = True,
        use_adaptive_params: bool = True,
        heuristic_ratio: float = 0.3,
        time_budget: Optional[float] = None,
        primary_objective: str = 'makespan',
        verbose: bool = False
    ):
        """
        Initialize GA optimizer.
        
        Args:
            pop_size: Population size
            max_generations: Maximum generations (default reduced for RL)
            use_heuristic_seeding: Use heuristic initialization
            use_adaptive_params: Use adaptive parameter control
            heuristic_ratio: Ratio of heuristic solutions
            time_budget: Maximum time in seconds (None = no limit)
            primary_objective: Which objective to optimize
            verbose: Print progress messages
        """
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.use_heuristic_seeding = use_heuristic_seeding
        self.use_adaptive_params = use_adaptive_params
        self.heuristic_ratio = heuristic_ratio
        self.time_budget = time_budget
        self.primary_objective = primary_objective
        self.verbose = verbose
        
        # Statistics
        self.last_run_time = 0.0
        self.last_generations_run = 0
        self.last_pareto_size = 0
        self.early_stopped = False
    
    def optimize(
        self,
        workflow: Workflow,
        resources: ResourceConfig,
        time_budget: Optional[float] = None
    ) -> Dict:
        """
        Run GA optimization and return best schedule.
        
        This is the main interface called by MADGA environment. It runs
        SmartGA for limited generations and returns a single best solution.
        
        Args:
            workflow: Workflow to schedule
            resources: Resource configuration
            time_budget: Time limit in seconds (overrides instance setting)
            
        Returns:
            Dict with keys:
                - schedule: Task assignments (task_id -> processor_id)
                - makespan: Schedule makespan
                - energy: Energy consumption
                - cost: Cost
                - metadata: Additional info (generations, time, pareto_size)
        """
        # Use provided time budget or instance default
        budget = time_budget if time_budget is not None else self.time_budget
        
        # Create SmartGA instance (seeding/adaptive set in constructor)
        smart_ga = SmartGA(
            workflow=workflow,
            resources=resources,
            pop_size=self.pop_size,
            use_heuristic_seeding=self.use_heuristic_seeding,
            use_adaptive_params=self.use_adaptive_params,
            heuristic_ratio=self.heuristic_ratio
        )
        
        # Initialize population
        smart_ga.initialize()
        
        # Setup time tracking and early stopping callback
        start_time = time.time()
        generations_run = [0]  # Use list to allow modification in callback
        self.early_stopped = False
        
        def check_time_budget(gen, stats):
            """Callback to check time budget and stop early if exceeded."""
            generations_run[0] = gen + 1
            if budget is not None:
                elapsed = time.time() - start_time
                if elapsed >= budget:
                    raise StopIteration  # Signal to stop early
        
        # Run GA with time budget checking
        try:
            smart_ga.run(
                num_generations=self.max_generations,
                verbose=self.verbose,
                callback=check_time_budget
            )
        except StopIteration:
            # Early stop due to time budget
            self.early_stopped = True
            if self.verbose:
                print(f"GA stopped early at generation {generations_run[0]} (time budget)")
        
        # Record statistics
        self.last_run_time = time.time() - start_time
        self.last_generations_run = generations_run[0] if not self.early_stopped else generations_run[0]
        
        # Get Pareto front
        pareto_front = smart_ga.get_pareto_front()
        self.last_pareto_size = len(pareto_front)
        
        # Decode chromosomes to get schedules
        pareto_with_schedules = []
        for solution in pareto_front:
            chrom = solution['chromosome']
            result_tuple = chrom.decode(workflow, resources)
            
            # Handle tuple return: (schedule_dict, makespan)
            if isinstance(result_tuple, tuple):
                schedule, _ = result_tuple
            else:
                schedule = result_tuple
            
            pareto_with_schedules.append({
                'chromosome': chrom,
                'schedule': schedule,
                'objectives': solution['objectives']
            })
        
        # Select best solution based on primary objective
        best_solution = self._select_best_solution(pareto_with_schedules)
        
        # Unpack objectives tuple: (makespan, cost, energy, load_balance)
        makespan, cost, energy, load_balance = best_solution['objectives']
        
        # Build result
        result = {
            'schedule': best_solution['schedule'],
            'makespan': makespan,
            'energy': energy,
            'cost': cost,
            'load_balance': load_balance,
            'metadata': {
                'generations': generations_run[0],
                'time': self.last_run_time,
                'pareto_size': self.last_pareto_size,
                'early_stopped': self.early_stopped,
                'primary_objective': self.primary_objective
            }
        }
        
        return result
    
    def _select_best_solution(self, pareto_front: List[Dict]) -> Dict:
        """
        Select single best solution from Pareto front.
        
        Uses primary objective to pick the best solution. For makespan
        and energy, we want minimum. For cost, it depends on interpretation
        but we also minimize.
        
        Args:
            pareto_front: List of solution dictionaries with 'objectives' tuple
            
        Returns:
            Best solution dictionary
        """
        if not pareto_front:
            raise ValueError("Pareto front is empty")
        
        # Objective indices in tuple: (makespan, cost, energy, load_balance)
        objective_indices = {
            'makespan': 0,
            'cost': 1,
            'energy': 2,
            'load_balance': 3
        }
        
        # Get index for primary objective
        obj_idx = objective_indices.get(self.primary_objective, 0)
        
        # Find solution with best primary objective
        best_idx = 0
        best_value = pareto_front[0]['objectives'][obj_idx]
        
        for i, solution in enumerate(pareto_front[1:], 1):
            value = solution['objectives'][obj_idx]
            if value < best_value:
                best_value = value
                best_idx = i
        
        return pareto_front[best_idx]
    
    def get_statistics(self) -> Dict:
        """
        Get statistics from last run.
        
        Returns:
            Dict with runtime, generations, pareto size, early stop flag
        """
        return {
            'run_time': self.last_run_time,
            'generations': self.last_generations_run,
            'pareto_size': self.last_pareto_size,
            'early_stopped': self.early_stopped
        }
    
    def set_parameters(
        self,
        pop_size: Optional[int] = None,
        max_generations: Optional[int] = None,
        time_budget: Optional[float] = None,
        heuristic_ratio: Optional[float] = None
    ) -> None:
        """
        Update optimizer parameters.
        
        Useful for tuning during RL training or experiments.
        
        Args:
            pop_size: New population size
            max_generations: New max generations
            time_budget: New time budget
            heuristic_ratio: New heuristic ratio
        """
        if pop_size is not None:
            self.pop_size = pop_size
        if max_generations is not None:
            self.max_generations = max_generations
        if time_budget is not None:
            self.time_budget = time_budget
        if heuristic_ratio is not None:
            self.heuristic_ratio = heuristic_ratio


class AdaptiveGAOptimizer(GAOptimizer):
    """
    Adaptive GA Optimizer that adjusts parameters based on workflow size.
    
    This extends GAOptimizer to automatically tune parameters based on
    the workflow complexity. Larger workflows get more generations and
    larger populations.
    """
    
    def __init__(
        self,
        base_pop_size: int = 30,
        base_generations: int = 30,
        use_heuristic_seeding: bool = True,
        use_adaptive_params: bool = True,
        heuristic_ratio: float = 0.3,
        time_budget: Optional[float] = None,
        primary_objective: str = 'makespan',
        verbose: bool = False
    ):
        """
        Initialize adaptive GA optimizer.
        
        Base parameters are scaled based on workflow size.
        
        Args:
            base_pop_size: Base population size (scaled by workflow)
            base_generations: Base generations (scaled by workflow)
            use_heuristic_seeding: Use heuristic initialization
            use_adaptive_params: Use adaptive parameter control
            heuristic_ratio: Ratio of heuristic solutions
            time_budget: Maximum time in seconds
            primary_objective: Which objective to optimize
            verbose: Print progress messages
        """
        super().__init__(
            pop_size=base_pop_size,
            max_generations=base_generations,
            use_heuristic_seeding=use_heuristic_seeding,
            use_adaptive_params=use_adaptive_params,
            heuristic_ratio=heuristic_ratio,
            time_budget=time_budget,
            primary_objective=primary_objective,
            verbose=verbose
        )
        
        self.base_pop_size = base_pop_size
        self.base_generations = base_generations
    
    def optimize(
        self,
        workflow: Workflow,
        resources: ResourceConfig,
        time_budget: Optional[float] = None
    ) -> Dict:
        """
        Run adaptive GA optimization.
        
        Automatically adjusts pop_size and max_generations based on
        workflow size (number of tasks).
        
        Args:
            workflow: Workflow to schedule
            resources: Resource configuration
            time_budget: Time limit in seconds
            
        Returns:
            Best schedule and metadata
        """
        # Scale parameters by workflow size
        num_tasks = len(workflow.tasks)
        
        # Simple scaling: sqrt of task count
        scale_factor = np.sqrt(num_tasks / 20.0)  # 20 tasks = baseline
        scale_factor = max(0.5, min(2.0, scale_factor))  # Clamp [0.5, 2.0]
        
        # Apply scaling
        self.pop_size = int(self.base_pop_size * scale_factor)
        self.max_generations = int(self.base_generations * scale_factor)
        
        if self.verbose:
            print(f"Adaptive GA: {num_tasks} tasks -> pop_size={self.pop_size}, "
                  f"generations={self.max_generations}")
        
        # Call parent optimize
        return super().optimize(workflow, resources, time_budget)


def create_ga_optimizer(
    mode: str = 'standard',
    **kwargs
) -> GAOptimizer:
    """
    Factory function to create GA optimizer instances.
    
    Args:
        mode: 'standard' or 'adaptive'
        **kwargs: Parameters passed to optimizer constructor
        
    Returns:
        GAOptimizer instance
        
    Example:
        >>> optimizer = create_ga_optimizer(
        ...     mode='adaptive',
        ...     base_pop_size=30,
        ...     time_budget=5.0
        ... )
        >>> result = optimizer.optimize(workflow, resources)
    """
    if mode == 'standard':
        return GAOptimizer(**kwargs)
    elif mode == 'adaptive':
        return AdaptiveGAOptimizer(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")
