"""
Mutation operators for Genetic Algorithm.

Implements various mutation strategies for introducing diversity:
- Swap mutation (exchange two genes)
- Insert mutation (move a gene to different position)
- Scramble mutation (shuffle a segment)
- Uniform mutation (random gene replacement)
"""

from abc import ABC, abstractmethod
import random
from .chromosome import Chromosome


class MutationOperator(ABC):
    """Base class for mutation operators."""
    
    @abstractmethod
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Perform mutation on a chromosome.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        pass


class SwapMutation(MutationOperator):
    """
    Swap mutation operator.
    
    Randomly selects two positions and swaps their values.
    This is the most common mutation for permutation-based encodings,
    but also works well for VM assignment encoding.
    
    Example:
        Original: [0, 1, 2, 3, 4]
        Swap positions 1 and 3
        Mutated:  [0, 3, 2, 1, 4]
    """
    
    def __init__(self, mutation_rate: float = 0.1):
        """
        Initialize swap mutation.
        
        Args:
            mutation_rate: Probability of mutating a chromosome (0.0-1.0).
                          Typical values: 0.01-0.1
                          Lower for exploitation, higher for exploration
        """
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("Mutation rate must be between 0.0 and 1.0")
        self.mutation_rate = mutation_rate
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Perform swap mutation.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome (new object)
        """
        # Check if mutation should occur
        if random.random() > self.mutation_rate:
            return chromosome.copy()
        
        # Create copy for mutation
        mutated = chromosome.copy()
        
        if len(mutated) < 2:
            return mutated
        
        # Select two different positions
        pos1 = random.randint(0, len(mutated) - 1)
        pos2 = random.randint(0, len(mutated) - 1)
        
        # Ensure different positions
        while pos2 == pos1 and len(mutated) > 1:
            pos2 = random.randint(0, len(mutated) - 1)
        
        # Swap genes
        mutated.genes[pos1], mutated.genes[pos2] = mutated.genes[pos2], mutated.genes[pos1]
        
        # Reset fitness (needs recalculation)
        mutated.fitness = float('inf')
        
        return mutated


class InsertMutation(MutationOperator):
    """
    Insert (displacement) mutation operator.
    
    Randomly selects a gene and moves it to a different position.
    Shifts other genes to accommodate the move.
    
    Example:
        Original: [0, 1, 2, 3, 4]
        Move position 1 to position 3
        Mutated:  [0, 2, 3, 1, 4]
    """
    
    def __init__(self, mutation_rate: float = 0.1):
        """
        Initialize insert mutation.
        
        Args:
            mutation_rate: Probability of mutating a chromosome (0.0-1.0)
        """
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("Mutation rate must be between 0.0 and 1.0")
        self.mutation_rate = mutation_rate
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Perform insert mutation.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        # Check if mutation should occur
        if random.random() > self.mutation_rate:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        
        if len(mutated) < 2:
            return mutated
        
        # Select source and destination positions
        src_pos = random.randint(0, len(mutated) - 1)
        dst_pos = random.randint(0, len(mutated) - 1)
        
        # Extract gene and insert at new position
        gene = mutated.genes.pop(src_pos)
        mutated.genes.insert(dst_pos, gene)
        
        mutated.fitness = float('inf')
        
        return mutated


class ScrambleMutation(MutationOperator):
    """
    Scramble mutation operator.
    
    Randomly selects a segment and shuffles the genes within it.
    Provides moderate disruption while maintaining some structure.
    
    Example:
        Original: [0, 1, 2, 3, 4]
        Scramble segment [1:4]
        Mutated:  [0, 3, 1, 2, 4]  (genes 1,2,3 shuffled)
    """
    
    def __init__(self, mutation_rate: float = 0.1):
        """
        Initialize scramble mutation.
        
        Args:
            mutation_rate: Probability of mutating a chromosome (0.0-1.0)
        """
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("Mutation rate must be between 0.0 and 1.0")
        self.mutation_rate = mutation_rate
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Perform scramble mutation.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        # Check if mutation should occur
        if random.random() > self.mutation_rate:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        
        if len(mutated) < 2:
            return mutated
        
        # Select segment to scramble
        pos1 = random.randint(0, len(mutated) - 1)
        pos2 = random.randint(0, len(mutated) - 1)
        
        # Ensure pos1 < pos2
        if pos1 > pos2:
            pos1, pos2 = pos2, pos1
        
        # Scramble the segment
        if pos2 > pos1:
            segment = mutated.genes[pos1:pos2+1]
            random.shuffle(segment)
            mutated.genes[pos1:pos2+1] = segment
        
        mutated.fitness = float('inf')
        
        return mutated


class UniformMutation(MutationOperator):
    """
    Uniform (random reset) mutation operator.
    
    Each gene has independent probability of being replaced with
    a random value from the valid range.
    
    Most disruptive mutation operator - provides maximum exploration.
    
    Example (num_vms=3, gene_mutation_rate=0.2):
        Original: [0, 1, 2, 3, 4]
        Mutate positions 1 and 3
        Mutated:  [0, 2, 2, 1, 4]
    """
    
    def __init__(
        self, 
        mutation_rate: float = 0.1,
        gene_mutation_rate: float = 0.1,
        num_vms: int = None
    ):
        """
        Initialize uniform mutation.
        
        Args:
            mutation_rate: Probability of attempting mutation on chromosome
            gene_mutation_rate: Probability of mutating each individual gene
            num_vms: Number of VMs (valid range for genes). 
                    If None, must be provided in mutate() call
        """
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("Mutation rate must be between 0.0 and 1.0")
        if not 0.0 <= gene_mutation_rate <= 1.0:
            raise ValueError("Gene mutation rate must be between 0.0 and 1.0")
        self.mutation_rate = mutation_rate
        self.gene_mutation_rate = gene_mutation_rate
        self.num_vms = num_vms
    
    def mutate(self, chromosome: Chromosome, num_vms: int = None) -> Chromosome:
        """
        Perform uniform mutation.
        
        Args:
            chromosome: Chromosome to mutate
            num_vms: Number of VMs (overrides constructor value)
            
        Returns:
            Mutated chromosome
        """
        # Check if mutation should occur
        if random.random() > self.mutation_rate:
            return chromosome.copy()
        
        # Determine num_vms
        vms = num_vms if num_vms is not None else self.num_vms
        if vms is None:
            raise ValueError("num_vms must be provided either in constructor or mutate()")
        
        mutated = chromosome.copy()
        
        # Mutate each gene independently
        for i in range(len(mutated)):
            if random.random() < self.gene_mutation_rate:
                mutated.genes[i] = random.randint(0, vms - 1)
        
        mutated.fitness = float('inf')
        
        return mutated


class InversionMutation(MutationOperator):
    """
    Inversion mutation operator.
    
    Selects a segment and reverses the order of genes within it.
    Less commonly used but can be effective for certain problems.
    
    Example:
        Original: [0, 1, 2, 3, 4]
        Invert segment [1:4]
        Mutated:  [0, 3, 2, 1, 4]
    """
    
    def __init__(self, mutation_rate: float = 0.1):
        """
        Initialize inversion mutation.
        
        Args:
            mutation_rate: Probability of mutating a chromosome (0.0-1.0)
        """
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("Mutation rate must be between 0.0 and 1.0")
        self.mutation_rate = mutation_rate
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Perform inversion mutation.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        # Check if mutation should occur
        if random.random() > self.mutation_rate:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        
        if len(mutated) < 2:
            return mutated
        
        # Select segment to invert
        pos1 = random.randint(0, len(mutated) - 1)
        pos2 = random.randint(0, len(mutated) - 1)
        
        # Ensure pos1 < pos2
        if pos1 > pos2:
            pos1, pos2 = pos2, pos1
        
        # Reverse the segment
        if pos2 > pos1:
            mutated.genes[pos1:pos2+1] = mutated.genes[pos1:pos2+1][::-1]
        
        mutated.fitness = float('inf')
        
        return mutated


class AdaptiveMutation(MutationOperator):
    """
    Adaptive mutation that adjusts rate based on population diversity.
    
    Increases mutation rate when diversity is low (converging),
    decreases when diversity is high (exploring).
    
    Can help balance exploration and exploitation automatically.
    """
    
    def __init__(
        self,
        base_mutation_rate: float = 0.1,
        min_mutation_rate: float = 0.01,
        max_mutation_rate: float = 0.3,
        operator: MutationOperator = None
    ):
        """
        Initialize adaptive mutation.
        
        Args:
            base_mutation_rate: Starting mutation rate
            min_mutation_rate: Minimum allowed mutation rate
            max_mutation_rate: Maximum allowed mutation rate
            operator: Base mutation operator to use (default: SwapMutation)
        """
        if not 0.0 <= base_mutation_rate <= 1.0:
            raise ValueError("Base mutation rate must be between 0.0 and 1.0")
        if not 0.0 <= min_mutation_rate <= max_mutation_rate <= 1.0:
            raise ValueError("Invalid mutation rate bounds")
        
        self.base_mutation_rate = base_mutation_rate
        self.min_mutation_rate = min_mutation_rate
        self.max_mutation_rate = max_mutation_rate
        self.current_mutation_rate = base_mutation_rate
        
        self.operator = operator if operator is not None else SwapMutation(mutation_rate=1.0)
    
    def adapt_rate(self, diversity: float):
        """
        Adapt mutation rate based on population diversity.
        
        Args:
            diversity: Population diversity metric (0.0-1.0)
                      0.0 = no diversity (all identical)
                      1.0 = maximum diversity
        """
        # Low diversity -> increase mutation
        # High diversity -> decrease mutation
        if diversity < 0.2:
            self.current_mutation_rate = min(
                self.max_mutation_rate,
                self.base_mutation_rate * 1.5
            )
        elif diversity > 0.6:
            self.current_mutation_rate = max(
                self.min_mutation_rate,
                self.base_mutation_rate * 0.5
            )
        else:
            self.current_mutation_rate = self.base_mutation_rate
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Perform adaptive mutation.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        # Use current mutation rate
        if random.random() > self.current_mutation_rate:
            return chromosome.copy()
        
        # Apply base operator
        return self.operator.mutate(chromosome)
