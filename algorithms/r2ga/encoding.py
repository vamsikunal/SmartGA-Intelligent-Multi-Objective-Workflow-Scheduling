"""
Encoding Module for R²GA

Implements Algorithm 1: Chromosome Encoding
Generates 2n-length chromosomes with random real values [0, 1)

Based on: J. Jiang et al., IEEE TPDS 2025
"""

import numpy as np
from typing import Optional


class Encoder:
    """
    Implements Algorithm 1: Encoding.
    
    Creates chromosomes representing potential workflow schedules.
    A chromosome is a 2n-length array of random real numbers in [0, 1).
    
    Chromosome Structure:
    - First n values: Task sequencing genes (determine task execution order)
    - Second n values: Unit assignment genes (determine processor allocation)
    
    The values are interpreted relatively during decoding based on available
    choices at each scheduling step.
    """
    
    def __init__(self, n: int, random_seed: Optional[int] = None):
        """
        Initialize the Encoder with number of tasks.
        
        Args:
            n: Total number of tasks in the workflow (must be positive)
            random_seed: Optional seed for reproducibility (default: None)
            
        Raises:
            ValueError: If n <= 0
        """
        if n <= 0:
            raise ValueError(f"Number of tasks (n) must be positive, got {n}")
        
        self.n = n
        self.chromosome_length = 2 * n
        
        # Set random seed if provided (for reproducibility)
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def encode_chromosome(self) -> np.ndarray:
        """
        Generate a single random chromosome (Algorithm 1).
        
        The chromosome consists of:
        - First n genes [0:n]: Task sequencing (determines task selection order)
        - Second n genes [n:2n]: Unit assignment (determines processor selection)
        
        All values are uniformly distributed in [0, 1).
        
        Returns:
            np.ndarray: 1D array of length 2n with random floats in [0, 1)
            
        Example:
            For n=4:
            [0.23, 0.67, 0.12, 0.89 | 0.45, 0.78, 0.34, 0.91]
             ^^^^^^^^^^^^^^^^^^^^  |  ^^^^^^^^^^^^^^^^^^^^
             Task sequencing genes  |  Unit assignment genes
        """
        # Generate 2n random real numbers uniformly distributed in [0, 1)
        chromosome = np.random.rand(self.chromosome_length)
        return chromosome
    
    def encode_population(self, pop_size: int) -> np.ndarray:
        """
        Generate multiple random chromosomes (population).
        
        Args:
            pop_size: Number of chromosomes to generate
            
        Returns:
            np.ndarray: 2D array of shape (pop_size, 2n) with random chromosomes
            
        Example:
            For n=4, pop_size=3:
            [[0.23, 0.67, 0.12, 0.89, 0.45, 0.78, 0.34, 0.91],
             [0.56, 0.34, 0.78, 0.21, 0.90, 0.12, 0.67, 0.43],
             [0.87, 0.45, 0.23, 0.67, 0.34, 0.89, 0.12, 0.56]]
        """
        if pop_size <= 0:
            raise ValueError(f"Population size must be positive, got {pop_size}")
        
        # Generate pop_size chromosomes
        population = np.random.rand(pop_size, self.chromosome_length)
        return population
    
    def validate_chromosome(self, chromosome: np.ndarray) -> bool:
        """
        Validate a chromosome structure and values.
        
        Checks:
        1. Length is exactly 2n
        2. All values are in [0, 1)
        3. No NaN or infinite values
        
        Args:
            chromosome: Chromosome to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Check length
        if len(chromosome) != self.chromosome_length:
            return False
        
        # Check all values in [0, 1)
        if not np.all((chromosome >= 0) & (chromosome < 1)):
            return False
        
        # Check for NaN or inf
        if not np.all(np.isfinite(chromosome)):
            return False
        
        return True
    
    def get_task_genes(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Extract task sequencing genes from chromosome.
        
        Args:
            chromosome: 2n-length chromosome
            
        Returns:
            np.ndarray: First n genes (task sequencing part)
        """
        return chromosome[:self.n]
    
    def get_unit_genes(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Extract unit assignment genes from chromosome.
        
        Args:
            chromosome: 2n-length chromosome
            
        Returns:
            np.ndarray: Second n genes (unit assignment part)
        """
        return chromosome[self.n:]
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Encoder(n={self.n}, chromosome_length={self.chromosome_length})"


if __name__ == '__main__':
    # --- Example Usage and Testing ---
    print("="*60)
    print("Encoder Class Demonstration (Algorithm 1)")
    print("="*60)
    
    num_tasks = 8
    
    # 1. Create encoder instance
    encoder = Encoder(n=num_tasks)
    print(f"\nCreated: {encoder}")
    
    # 2. Generate a single chromosome
    print(f"\n--- Single Chromosome Generation ---")
    chromosome = encoder.encode_chromosome()
    print(f"Chromosome (length {len(chromosome)}):")
    print(f"  {chromosome}")
    
    # 3. Extract task and unit parts
    task_genes = encoder.get_task_genes(chromosome)
    unit_genes = encoder.get_unit_genes(chromosome)
    
    print(f"\nTask Sequencing Genes (first {num_tasks}):")
    print(f"  {task_genes}")
    print(f"\nUnit Assignment Genes (second {num_tasks}):")
    print(f"  {unit_genes}")
    
    # 4. Validate chromosome
    print(f"\n--- Chromosome Validation ---")
    is_valid = encoder.validate_chromosome(chromosome)
    print(f"Chromosome valid: {is_valid}")
    
    # 5. Generate population
    print(f"\n--- Population Generation ---")
    pop_size = 5
    population = encoder.encode_population(pop_size)
    print(f"Population shape: {population.shape} (pop_size={pop_size}, chromosome_length={2*num_tasks})")
    print("Population:")
    for i, chrom in enumerate(population):
        print(f"  Chromosome {i}: [{chrom[0]:.3f}, {chrom[1]:.3f}, ..., {chrom[-2]:.3f}, {chrom[-1]:.3f}]")
    
    # 6. Test reproducibility with seed
    print(f"\n--- Reproducibility Test (with seed) ---")
    encoder_seeded1 = Encoder(n=4, random_seed=42)
    encoder_seeded2 = Encoder(n=4, random_seed=42)
    
    chrom1 = encoder_seeded1.encode_chromosome()
    chrom2 = encoder_seeded2.encode_chromosome()
    
    print(f"Chromosome 1 (seed=42): {chrom1}")
    print(f"Chromosome 2 (seed=42): {chrom2}")
    print(f"Identical: {np.array_equal(chrom1, chrom2)}")
    
    print("\n" + "="*60)
    print("✅ Encoder implementation complete and tested!")
    print("="*60)
