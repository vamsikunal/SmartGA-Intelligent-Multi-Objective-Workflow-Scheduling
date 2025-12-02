"""
R²GA Genetic Operators - Algorithm 5 & 6 (TA-Approved Specifications)
======================================================================
Implements crossover and mutation operators for R²GA.

⭐ CRITICAL: These implementations follow the EXACT specifications from TA feedback:
- Crossover: TWO-POINT crossover (Algorithm 5, Figure 3)
- Mutation: TWO-POINT random replacement (Algorithm 6, Figure 4)

Components:
- Crossover: Two-point crossover with segment swap
- Mutator: Two-point random replacement

Author: GA Scheduling Team
Date: October 2025
Version: 1.1.0 (TA-Approved)
"""

import numpy as np
from typing import Tuple


class Crossover:
    """
    Two-point crossover operator for R²GA (Algorithm 5, Figure 3).
    
    ⭐ TA-APPROVED SPECIFICATION:
    - Select point1 in task-sequencing part [0, n)
    - Select point2 in unit-assignment part [n, 2n)
    - Swap segment between point1 and point2 between two parents
    
    This ensures crossover affects both task ordering and processor assignment.
    """
    
    def __init__(self, crossover_rate: float = 0.9):
        """
        Initialize crossover operator.
        
        Args:
            crossover_rate: Probability of applying crossover (default: 0.9)
        """
        self.crossover_rate = crossover_rate
        
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError(f"Crossover rate must be in [0, 1], got {crossover_rate}")
    
    def apply(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply two-point crossover to create two offspring (Algorithm 5).
        
        Args:
            parent1: First parent chromosome (2n,)
            parent2: Second parent chromosome (2n,)
        
        Returns:
            offspring1: First offspring (2n,)
            offspring2: Second offspring (2n,)
        """
        n = len(parent1) // 2
        
        # Check if crossover should be applied
        if np.random.random() > self.crossover_rate:
            # Return copies of parents (no crossover)
            return parent1.copy(), parent2.copy()
        
        # ⭐ TA-APPROVED TWO-POINT CROSSOVER:
        # Point 1: Random position in task-sequencing part [0, n)
        point1 = np.random.randint(0, n)
        
        # Point 2: Random position in unit-assignment part [n, 2n)
        point2 = np.random.randint(n, 2 * n)
        
        # Create offspring as copies of parents
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # Swap segment between point1 and point2
        # Offspring1 gets segment from Parent2
        # Offspring2 gets segment from Parent1
        offspring1[point1:point2] = parent2[point1:point2]
        offspring2[point1:point2] = parent1[point1:point2]
        
        return offspring1, offspring2
    
    def __repr__(self):
        return f"Crossover(rate={self.crossover_rate}, type='two-point')"


class Mutator:
    """
    Two-point random replacement mutation for R²GA (Algorithm 6, Figure 4).
    
    ⭐ TA-APPROVED SPECIFICATION:
    - Replace ONE gene in task-sequencing part [0, n) with new random value [0, 1)
    - Replace ONE gene in unit-assignment part [n, 2n) with new random value [0, 1)
    
    This ensures mutation affects both task ordering and processor assignment.
    """
    
    def __init__(self, mutation_rate: float = 0.1):
        """
        Initialize mutation operator.
        
        Args:
            mutation_rate: Probability of applying mutation (default: 0.1)
        """
        self.mutation_rate = mutation_rate
        
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError(f"Mutation rate must be in [0, 1], got {mutation_rate}")
    
    def apply(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Apply two-point random replacement mutation (Algorithm 6).
        
        Args:
            chromosome: Input chromosome (2n,)
        
        Returns:
            mutated: Mutated chromosome (2n,)
        """
        n = len(chromosome) // 2
        
        # Check if mutation should be applied
        if np.random.random() > self.mutation_rate:
            # Return copy (no mutation)
            return chromosome.copy()
        
        # ⭐ TA-APPROVED TWO-POINT MUTATION:
        # Create mutated chromosome as copy
        mutated = chromosome.copy()
        
        # Point 1: Random position in task-sequencing part [0, n)
        # Replace with new random value [0, 1)
        task_point = np.random.randint(0, n)
        mutated[task_point] = np.random.random()
        
        # Point 2: Random position in unit-assignment part [n, 2n)
        # Replace with new random value [0, 1)
        unit_point = np.random.randint(n, 2 * n)
        mutated[unit_point] = np.random.random()
        
        return mutated
    
    def __repr__(self):
        return f"Mutator(rate={self.mutation_rate}, type='two-point-random-replacement')"


# ============================================================================
# Testing & Demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Genetic Operators Demonstration (Algorithms 5 & 6 - TA-Approved)")
    print("=" * 80)
    
    # Test parameters
    n = 8  # tasks
    chromosome_length = 2 * n
    
    print(f"\nTest Configuration:")
    print(f"  Tasks: {n}")
    print(f"  Chromosome length: {chromosome_length}")
    print(f"  ⭐ Operators: TWO-POINT (TA-Approved)")
    
    # Create sample chromosomes
    np.random.seed(42)
    parent1 = np.random.random(chromosome_length)
    parent2 = np.random.random(chromosome_length)
    
    print(f"\nParent Chromosomes:")
    print(f"  Parent 1 (task): {parent1[:n]}")
    print(f"  Parent 1 (unit): {parent1[n:]}")
    print(f"  Parent 2 (task): {parent2[:n]}")
    print(f"  Parent 2 (unit): {parent2[n:]}")
    
    # Test 1: Two-point Crossover
    print("\n--- Test 1: Two-Point Crossover (Algorithm 5) ---")
    crossover = Crossover(crossover_rate=1.0)  # Always apply for testing
    print(f"Created: {crossover}")
    
    np.random.seed(42)  # For reproducible test
    offspring1, offspring2 = crossover.apply(parent1, parent2)
    
    print(f"\nOffspring 1:")
    print(f"  Task genes: {offspring1[:n]}")
    print(f"  Unit genes: {offspring1[n:]}")
    print(f"\nOffspring 2:")
    print(f"  Task genes: {offspring2[:n]}")
    print(f"  Unit genes: {offspring2[n:]}")
    
    # Verify crossover worked
    different_from_p1 = not np.array_equal(offspring1, parent1)
    different_from_p2 = not np.array_equal(offspring1, parent2)
    valid_length = len(offspring1) == chromosome_length
    
    print(f"\nCrossover Validation:")
    print(f"  Offspring1 != Parent1: {different_from_p1}")
    print(f"  Offspring1 != Parent2: {different_from_p2}")
    print(f"  Valid length: {valid_length}")
    
    # Test 2: Two-point Mutation
    print("\n--- Test 2: Two-Point Random Replacement (Algorithm 6) ---")
    mutator = Mutator(mutation_rate=1.0)  # Always apply for testing
    print(f"Created: {mutator}")
    
    original = np.random.random(chromosome_length)
    print(f"\nOriginal chromosome:")
    print(f"  Task genes: {original[:n]}")
    print(f"  Unit genes: {original[n:]}")
    
    np.random.seed(42)  # For reproducible test
    mutated = mutator.apply(original)
    
    print(f"\nMutated chromosome:")
    print(f"  Task genes: {mutated[:n]}")
    print(f"  Unit genes: {mutated[n:]}")
    
    # Verify mutation worked
    differences = np.where(original != mutated)[0]
    task_mutated = any(diff < n for diff in differences)
    unit_mutated = any(diff >= n for diff in differences)
    
    print(f"\nMutation Validation:")
    print(f"  Differences at indices: {differences.tolist()}")
    print(f"  Task part mutated: {task_mutated}")
    print(f"  Unit part mutated: {unit_mutated}")
    print(f"  Total genes changed: {len(differences)}")
    
    # Test 3: Crossover Rate Effect
    print("\n--- Test 3: Crossover Rate Effect ---")
    crossover_50 = Crossover(crossover_rate=0.5)
    print(f"Testing: {crossover_50}")
    
    np.random.seed(100)
    applications = 0
    trials = 100
    for _ in range(trials):
        off1, off2 = crossover_50.apply(parent1, parent2)
        if not np.array_equal(off1, parent1):
            applications += 1
    
    application_rate = applications / trials
    print(f"Crossover applied in {applications}/{trials} trials ({application_rate:.1%})")
    print(f"Expected rate: 50% (close to {application_rate:.1%} ✓)" if 0.4 <= application_rate <= 0.6 else f"Expected ~50%, got {application_rate:.1%}")
    
    # Test 4: Mutation Rate Effect
    print("\n--- Test 4: Mutation Rate Effect ---")
    mutator_10 = Mutator(mutation_rate=0.1)
    print(f"Testing: {mutator_10}")
    
    np.random.seed(100)
    applications = 0
    trials = 100
    test_chromosome = np.random.random(chromosome_length)
    for _ in range(trials):
        mut = mutator_10.apply(test_chromosome)
        if not np.array_equal(mut, test_chromosome):
            applications += 1
    
    application_rate = applications / trials
    print(f"Mutation applied in {applications}/{trials} trials ({application_rate:.1%})")
    print(f"Expected rate: 10% (close to {application_rate:.1%} ✓)" if 0.05 <= application_rate <= 0.15 else f"Expected ~10%, got {application_rate:.1%}")
    
    # Test 5: Multiple Crossover/Mutation Rounds
    print("\n--- Test 5: Evolution Simulation (10 generations) ---")
    crossover_op = Crossover(crossover_rate=0.9)
    mutator_op = Mutator(mutation_rate=0.1)
    
    population = np.random.random((10, chromosome_length))
    print(f"Initial population: {population.shape}")
    
    for gen in range(10):
        # Simulate breeding
        new_population = []
        for i in range(0, len(population), 2):
            p1 = population[i]
            p2 = population[(i+1) % len(population)]
            
            # Crossover
            off1, off2 = crossover_op.apply(p1, p2)
            
            # Mutation
            off1 = mutator_op.apply(off1)
            off2 = mutator_op.apply(off2)
            
            new_population.extend([off1, off2])
        
        population = np.array(new_population)
    
    print(f"Final population (after 10 generations): {population.shape}")
    print(f"Sample individual:")
    print(f"  Task genes: {population[0, :n]}")
    print(f"  Unit genes: {population[0, n:]}")
    
    print("\n" + "=" * 80)
    print("✅ Genetic operators implementation complete and tested!")
    print("   ⭐ TWO-POINT crossover and mutation (TA-Approved)")
    print("=" * 80)
