"""
Crossover operators for Genetic Algorithm.

Implements various crossover strategies for combining parent chromosomes:
- Single-point crossover (simple, classical)
- Two-point crossover (more diverse recombination)
- Uniform crossover (maximum gene mixing)
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import random
from .chromosome import Chromosome


class CrossoverOperator(ABC):
    """Base class for crossover operators."""
    
    @abstractmethod
    def crossover(
        self, 
        parent1: Chromosome, 
        parent2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Tuple of (offspring1, offspring2)
        """
        pass


class SinglePointCrossover(CrossoverOperator):
    """
    Single-point crossover operator.
    
    The classical crossover method:
    1. Select random crossover point
    2. Child1 gets genes from parent1 up to point, then parent2 after
    3. Child2 gets genes from parent2 up to point, then parent1 after
    
    Example (point=2):
        Parent1: [0, 1, 2, 3, 4]
        Parent2: [5, 6, 7, 8, 9]
        -------- point --------
        Child1:  [0, 1, 7, 8, 9]
        Child2:  [5, 6, 2, 3, 4]
    """
    
    def __init__(self, crossover_rate: float = 0.8):
        """
        Initialize single-point crossover.
        
        Args:
            crossover_rate: Probability of performing crossover (0.0-1.0).
                           If crossover doesn't occur, return copies of parents.
                           Typical values: 0.6-0.9
        """
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("Crossover rate must be between 0.0 and 1.0")
        self.crossover_rate = crossover_rate
    
    def crossover(
        self, 
        parent1: Chromosome, 
        parent2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        """
        Perform single-point crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring chromosomes
        """
        # Check if crossover should occur
        if random.random() > self.crossover_rate:
            # No crossover - return copies of parents
            return parent1.copy(), parent2.copy()
        
        # Ensure parents have same length
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have same length for crossover")
        
        length = len(parent1)
        
        # Select crossover point (1 to length-1)
        # Exclude 0 and length to ensure some mixing
        if length <= 1:
            return parent1.copy(), parent2.copy()
        
        point = random.randint(1, length - 1)
        
        # Create offspring
        child1_genes = parent1.genes[:point] + parent2.genes[point:]
        child2_genes = parent2.genes[:point] + parent1.genes[point:]
        
        child1 = Chromosome(genes=child1_genes)
        child2 = Chromosome(genes=child2_genes)
        
        return child1, child2


class TwoPointCrossover(CrossoverOperator):
    """
    Two-point crossover operator.
    
    More flexible than single-point:
    1. Select two random crossover points
    2. Exchange the segment between points
    
    Example (points=2,4):
        Parent1: [0, 1, 2, 3, 4, 5]
        Parent2: [6, 7, 8, 9, 10, 11]
        -------- p1 -- p2 --------
        Child1:  [0, 1, 8, 9, 4, 5]
        Child2:  [6, 7, 2, 3, 10, 11]
    """
    
    def __init__(self, crossover_rate: float = 0.8):
        """
        Initialize two-point crossover.
        
        Args:
            crossover_rate: Probability of performing crossover (0.0-1.0)
        """
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("Crossover rate must be between 0.0 and 1.0")
        self.crossover_rate = crossover_rate
    
    def crossover(
        self, 
        parent1: Chromosome, 
        parent2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        """
        Perform two-point crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring chromosomes
        """
        # Check if crossover should occur
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Ensure parents have same length
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have same length for crossover")
        
        length = len(parent1)
        
        # Need at least 3 genes for meaningful two-point crossover
        if length < 3:
            return parent1.copy(), parent2.copy()
        
        # Select two different crossover points
        point1 = random.randint(1, length - 2)
        point2 = random.randint(point1 + 1, length - 1)
        
        # Create offspring by exchanging middle segment
        child1_genes = (
            parent1.genes[:point1] + 
            parent2.genes[point1:point2] + 
            parent1.genes[point2:]
        )
        child2_genes = (
            parent2.genes[:point1] + 
            parent1.genes[point1:point2] + 
            parent2.genes[point2:]
        )
        
        child1 = Chromosome(genes=child1_genes)
        child2 = Chromosome(genes=child2_genes)
        
        return child1, child2


class UniformCrossover(CrossoverOperator):
    """
    Uniform crossover operator.
    
    Gene-level recombination:
    1. For each gene position, randomly choose which parent to inherit from
    2. Child1 gets gene from parent1 or parent2 with probability 0.5
    3. Child2 gets the opposite choice
    
    Provides maximum mixing and exploration.
    
    Example:
        Parent1: [0, 1, 2, 3, 4]
        Parent2: [5, 6, 7, 8, 9]
        Mask:    [0, 1, 0, 1, 0]  (0=parent1, 1=parent2)
        -------------------------
        Child1:  [0, 6, 2, 8, 4]
        Child2:  [5, 1, 7, 3, 9]
    """
    
    def __init__(self, crossover_rate: float = 0.8, mixing_ratio: float = 0.5):
        """
        Initialize uniform crossover.
        
        Args:
            crossover_rate: Probability of performing crossover (0.0-1.0)
            mixing_ratio: Probability of selecting gene from parent2 (0.0-1.0).
                         0.5 = equal mixing (recommended)
                         > 0.5 = favor parent2
                         < 0.5 = favor parent1
        """
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("Crossover rate must be between 0.0 and 1.0")
        if not 0.0 <= mixing_ratio <= 1.0:
            raise ValueError("Mixing ratio must be between 0.0 and 1.0")
        self.crossover_rate = crossover_rate
        self.mixing_ratio = mixing_ratio
    
    def crossover(
        self, 
        parent1: Chromosome, 
        parent2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        """
        Perform uniform crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring chromosomes
        """
        # Check if crossover should occur
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Ensure parents have same length
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have same length for crossover")
        
        length = len(parent1)
        
        # Create offspring by randomly selecting genes
        child1_genes = []
        child2_genes = []
        
        for i in range(length):
            if random.random() < self.mixing_ratio:
                # Select from parent2 for child1, parent1 for child2
                child1_genes.append(parent2.genes[i])
                child2_genes.append(parent1.genes[i])
            else:
                # Select from parent1 for child1, parent2 for child2
                child1_genes.append(parent1.genes[i])
                child2_genes.append(parent2.genes[i])
        
        child1 = Chromosome(genes=child1_genes)
        child2 = Chromosome(genes=child2_genes)
        
        return child1, child2


class OrderCrossover(CrossoverOperator):
    """
    Order crossover (OX) operator.
    
    Preserves relative order of genes - useful for permutation problems.
    For workflow scheduling, this is less relevant since we don't need
    to preserve task order (only VM assignments).
    
    Included for completeness and future extensions.
    """
    
    def __init__(self, crossover_rate: float = 0.8):
        """
        Initialize order crossover.
        
        Args:
            crossover_rate: Probability of performing crossover (0.0-1.0)
        """
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("Crossover rate must be between 0.0 and 1.0")
        self.crossover_rate = crossover_rate
    
    def crossover(
        self, 
        parent1: Chromosome, 
        parent2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        """
        Perform order crossover.
        
        Note: For VM assignment encoding, order doesn't matter,
        so this behaves similarly to two-point crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring chromosomes
        """
        # Check if crossover should occur
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Ensure parents have same length
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have same length for crossover")
        
        length = len(parent1)
        
        if length < 3:
            return parent1.copy(), parent2.copy()
        
        # Select two points
        point1 = random.randint(1, length - 2)
        point2 = random.randint(point1 + 1, length - 1)
        
        # For VM assignment, simply copy segments
        # (order preservation not needed)
        child1_genes = (
            parent1.genes[:point1] + 
            parent2.genes[point1:point2] + 
            parent1.genes[point2:]
        )
        child2_genes = (
            parent2.genes[:point1] + 
            parent1.genes[point1:point2] + 
            parent2.genes[point2:]
        )
        
        child1 = Chromosome(genes=child1_genes)
        child2 = Chromosome(genes=child2_genes)
        
        return child1, child2


class AdaptiveCrossover(CrossoverOperator):
    """
    Adaptive crossover that switches between operators.
    
    Chooses crossover operator based on population diversity or
    generation number. Can help balance exploration vs exploitation.
    """
    
    def __init__(
        self, 
        operators: List[CrossoverOperator] = None,
        probabilities: List[float] = None
    ):
        """
        Initialize adaptive crossover.
        
        Args:
            operators: List of crossover operators to choose from
            probabilities: Probability of selecting each operator.
                          If None, use uniform distribution.
        """
        if operators is None:
            # Default: use all three main operators
            operators = [
                SinglePointCrossover(),
                TwoPointCrossover(),
                UniformCrossover()
            ]
        
        self.operators = operators
        
        if probabilities is None:
            # Uniform distribution
            self.probabilities = [1.0 / len(operators)] * len(operators)
        else:
            if len(probabilities) != len(operators):
                raise ValueError("Number of probabilities must match operators")
            if abs(sum(probabilities) - 1.0) > 0.001:
                raise ValueError("Probabilities must sum to 1.0")
            self.probabilities = probabilities
    
    def crossover(
        self, 
        parent1: Chromosome, 
        parent2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        """
        Perform crossover using randomly selected operator.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring chromosomes
        """
        # Select operator based on probabilities
        r = random.random()
        cumulative = 0.0
        
        for operator, prob in zip(self.operators, self.probabilities):
            cumulative += prob
            if r <= cumulative:
                return operator.crossover(parent1, parent2)
        
        # Fallback (should never reach here if probabilities sum to 1.0)
        return self.operators[0].crossover(parent1, parent2)
