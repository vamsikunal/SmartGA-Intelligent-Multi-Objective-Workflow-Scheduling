"""
Fitness evaluation for GA chromosomes.

Provides efficient fitness calculation with caching and batch evaluation.
"""

from typing import List, Callable, Optional
from .chromosome import Chromosome
from .utils import Workflow
from .utils import ResourceConfig


class FitnessEvaluator:
    """
    Evaluates fitness of chromosomes.
    
    Currently supports single-objective fitness (makespan).
    Will be extended to multi-objective in Week 2.
    
    Features:
    - Caching to avoid redundant evaluations
    - Batch evaluation support
    - Custom fitness function support
    
    Example:
        >>> evaluator = FitnessEvaluator(workflow, resources)
        >>> evaluator.evaluate(chromosome)
        >>> print(chromosome.fitness)  # Makespan value
    """
    
    def __init__(
        self,
        workflow: Workflow,
        resources: ResourceConfig,
        fitness_func: Optional[Callable] = None
    ):
        """
        Initialize fitness evaluator.
        
        Args:
            workflow: Workflow to schedule
            resources: Available computing resources
            fitness_func: Optional custom fitness function.
                         If None, uses makespan (default)
        """
        self.workflow = workflow
        self.resources = resources
        self.fitness_func = fitness_func or self._makespan_fitness
        self._evaluation_count = 0
        self._cache = {}
    
    def evaluate(self, chromosome: Chromosome) -> float:
        """
        Evaluate fitness of a single chromosome.
        
        Args:
            chromosome: Chromosome to evaluate
            
        Returns:
            Fitness value (lower is better)
        """
        # Check if already evaluated
        if chromosome.fitness != float('inf'):
            return chromosome.fitness
        
        # Check cache
        chrom_hash = hash(chromosome)
        if chrom_hash in self._cache:
            chromosome.fitness = self._cache[chrom_hash]
            return chromosome.fitness
        
        # Evaluate using fitness function
        fitness = self.fitness_func(chromosome, self.workflow, self.resources)
        chromosome.fitness = fitness
        
        # Update cache and counter
        self._cache[chrom_hash] = fitness
        self._evaluation_count += 1
        
        return fitness
    
    def evaluate_batch(self, chromosomes: List[Chromosome]) -> List[float]:
        """
        Evaluate fitness of multiple chromosomes.
        
        More efficient than calling evaluate() multiple times
        as it can skip already-evaluated chromosomes.
        
        Args:
            chromosomes: List of chromosomes to evaluate
            
        Returns:
            List of fitness values
        """
        fitnesses = []
        
        for chromosome in chromosomes:
            fitness = self.evaluate(chromosome)
            fitnesses.append(fitness)
        
        return fitnesses
    
    def _makespan_fitness(
        self,
        chromosome: Chromosome,
        workflow: Workflow,
        resources: ResourceConfig
    ) -> float:
        """
        Default fitness function: makespan.
        
        Decodes chromosome to schedule and computes makespan
        (completion time of last task).
        
        Args:
            chromosome: Chromosome to evaluate
            workflow: Workflow to schedule
            resources: Available resources
            
        Returns:
            Makespan (lower is better)
        """
        # Decode chromosome to schedule
        # decode() returns (schedule_dict, makespan)
        result = chromosome.decode(workflow, resources)
        
        # Handle tuple return from decode
        if isinstance(result, tuple):
            schedule, makespan = result
            # Check if schedule is empty or makespan is invalid
            if not schedule or makespan == 0:
                return float('inf')
            return makespan
        
        # Fallback for older decode format (dict only)
        schedule = result
        if not schedule:
            return float('inf')
        
        try:
            makespan = max(
                finish_time for _, _, finish_time in schedule.values()
            )
            return makespan
        except (ValueError, AttributeError):
            # Empty schedule or invalid format
            return float('inf')
    
    def reset_cache(self):
        """Clear the evaluation cache."""
        self._cache.clear()
    
    def get_evaluation_count(self) -> int:
        """Get total number of fitness evaluations performed."""
        return self._evaluation_count
    
    def get_cache_size(self) -> int:
        """Get number of cached fitness values."""
        return len(self._cache)


class MultiObjectiveFitnessEvaluator(FitnessEvaluator):
    """
    Multi-objective fitness evaluator.
    
    Will be implemented in Week 2 for NSGA-II.
    Currently a placeholder that raises NotImplementedError.
    
    Planned objectives:
    - Makespan (execution time)
    - Cost (resource usage cost)
    - Energy consumption
    - Load balance
    """
    
    def __init__(
        self,
        workflow: Workflow,
        resources: ResourceConfig,
        objectives: Optional[List[str]] = None
    ):
        """
        Initialize multi-objective evaluator.
        
        Args:
            workflow: Workflow to schedule
            resources: Available resources
            objectives: List of objective names to optimize
                       Default: ['makespan', 'cost', 'energy']
        """
        super().__init__(workflow, resources)
        self.objectives = objectives or ['makespan', 'cost', 'energy']
    
    def evaluate(self, chromosome: Chromosome) -> List[float]:
        """
        Evaluate multiple objectives.
        
        Args:
            chromosome: Chromosome to evaluate
            
        Returns:
            List of objective values
            
        Raises:
            NotImplementedError: Not yet implemented (Week 2)
        """
        raise NotImplementedError(
            "Multi-objective evaluation will be implemented in Week 2"
        )
