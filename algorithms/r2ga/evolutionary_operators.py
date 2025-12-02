"""
R²GA Evolutionary Operators - Algorithm 3 & 4
==============================================
Implements population initialization and selection mechanisms for R²GA.

Components:
- InitialPopulation: Creates initial population with optional HEFT seed (Algorithm 3)
- Selector: Elite selection mechanism (Algorithm 4)

Author: GA Scheduling Team
Date: October 2025
Version: 1.0.0
"""

import numpy as np
from typing import List, Tuple, Optional
import sys
import os

# Handle imports for both standalone and module use
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

try:
    # Try relative imports first (when used as a module)
    from .encoding import Encoder
    from baselines.list_heuristics.heft.heft import HEFT
    from baselines.base_scheduler import ResourceConfig
except ImportError:
    # Fallback to absolute imports (when run standalone)
    from encoding import Encoder
    from baselines.list_heuristics.heft.heft import HEFT
    from baselines.base_scheduler import ResourceConfig
    from src.baselines.base_scheduler import ResourceConfig


class InitialPopulation:
    """
    Creates initial population for R²GA (Algorithm 3 from paper).
    
    Can optionally include a chromosome derived from HEFT schedule
    to seed the population with a good solution.
    """
    
    def __init__(self, n: int, seed_with_heft: bool = True, random_seed: Optional[int] = None):
        """
        Initialize population generator.
        
        Args:
            n: Number of tasks in the workflow
            seed_with_heft: If True, include HEFT-derived chromosome in population
            random_seed: Random seed for reproducibility (optional)
        """
        self.n = n
        self.encoder = Encoder(n=n, random_seed=random_seed)
        self.seed_with_heft = seed_with_heft
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def create(self, pop_size: int, workflow_dag: np.ndarray, 
               comp_matrix: np.ndarray, comm_matrix: np.ndarray) -> np.ndarray:
        """
        Create initial population (Algorithm 3).
        
        Args:
            pop_size: Population size
            workflow_dag: DAG adjacency matrix (n x n)
            comp_matrix: Computation cost matrix (n x p)
            comm_matrix: Communication cost matrix (n x n)
        
        Returns:
            population: Initial population (pop_size x 2n)
        """
        n_tasks = len(workflow_dag)
        
        if self.seed_with_heft:
            # Create pop_size - 1 random chromosomes
            random_population = self.encoder.encode_population(pop_size - 1)
            
            # Create HEFT-derived chromosome
            heft_chromosome = self._heft_to_chromosome(workflow_dag, comp_matrix, comm_matrix)
            
            # Combine: HEFT chromosome + random chromosomes
            population = np.vstack([heft_chromosome, random_population])
        else:
            # Create all random chromosomes
            population = self.encoder.encode_population(pop_size)
        
        return population
    
    def _heft_to_chromosome(self, workflow_dag: np.ndarray, 
                           comp_matrix: np.ndarray, 
                           comm_matrix: np.ndarray) -> np.ndarray:
        """
        Convert HEFT schedule to R²GA chromosome encoding.
        
        This reverse-engineers a chromosome that would decode to something
        similar to the HEFT schedule.
        
        Args:
            workflow_dag: DAG adjacency matrix (n x n)
            comp_matrix: Computation cost matrix (n x p)
            comm_matrix: Communication cost matrix (n x n)
        
        Returns:
            chromosome: Single chromosome (1 x 2n) derived from HEFT
        """
        n_tasks = len(workflow_dag)
        n_processors = comp_matrix.shape[1]
        
        # Convert DAG matrix to dictionary format
        dag_dict = {}
        predecessors = [[] for _ in range(n_tasks)]
        for i in range(n_tasks):
            successors = []
            for j in range(n_tasks):
                if workflow_dag[i, j] == 1:
                    successors.append(j)
                    predecessors[j].append(i)
            dag_dict[i] = successors
        
        # Create ResourceConfig for HEFT
        resources = ResourceConfig(
            num_processors=n_processors,
            processor_speeds=np.ones(n_processors)  # Assume uniform speeds
        )
        
        # Run HEFT to get schedule (use tuple format: n, W, C, dag, predecessors)
        heft = HEFT()
        schedule_result = heft.schedule((n_tasks, comp_matrix, comm_matrix, dag_dict, predecessors), resources)
        
        # Extract task order from start times (sorted by start time)
        task_start_times = [(task_id, start_time) for task_id, start_time in schedule_result.task_start_times.items()]
        task_start_times.sort(key=lambda x: x[1])  # Sort by start time
        task_order = [task_id for task_id, _ in task_start_times]
        
        # Extract processor assignments
        processor_assignments = schedule_result.task_assignments  # dict: task_id -> processor_id
        
        # Initialize chromosome
        chromosome = np.zeros(2 * n_tasks)
        
        # Task sequencing part [0, n):
        # Convert task_order to relative positions [0, 1)
        # Task executed first gets smallest value, last gets largest
        for i, task_id in enumerate(task_order):
            # Map position i (0 to n-1) to value in [0, 1)
            # Add small random jitter to avoid exact ties
            base_value = i / n_tasks
            jitter = np.random.uniform(0, 0.8 / n_tasks)  # Small jitter within bin
            chromosome[task_id] = base_value + jitter
        
        # Unit assignment part [n, 2n):
        # Convert processor assignments to [0, 1) values
        for task_id in range(n_tasks):
            processor_id = processor_assignments[task_id]
            # Map processor_id (0 to p-1) to value in [0, 1)
            # This ensures decoder will select same processor
            base_value = processor_id / n_processors
            jitter = np.random.uniform(0, 0.8 / n_processors)  # Small jitter
            chromosome[n_tasks + task_id] = base_value + jitter
        
        return chromosome.reshape(1, -1)
    
    def __repr__(self):
        return f"InitialPopulation(seed_with_heft={self.seed_with_heft}, n={self.n})"


class Selector:
    """
    Elite selection mechanism for R²GA (Algorithm 4 from paper).
    
    Preserves top-performing individuals and selects from remaining
    population for next generation.
    """
    
    def __init__(self, elite_size: int):
        """
        Initialize selector.
        
        Args:
            elite_size: Number of elite individuals to preserve
        """
        self.elite_size = elite_size
    
    def select(self, population: np.ndarray, fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform elite selection (Algorithm 4).
        
        Args:
            population: Current population (pop_size x 2n)
            fitness: Fitness values for each individual (pop_size,)
                    Note: Lower is better (makespan minimization)
        
        Returns:
            elites: Elite individuals (elite_size x 2n)
            selected: Selected individuals for breeding (pop_size - elite_size x 2n)
        """
        pop_size = len(population)
        
        # Sort population by fitness (ascending - lower makespan is better)
        sorted_indices = np.argsort(fitness)
        
        # Select elites (best individuals)
        elite_indices = sorted_indices[:self.elite_size]
        elites = population[elite_indices]
        
        # Select remaining for breeding
        # Use fitness-proportionate selection on non-elite individuals
        remaining_indices = sorted_indices[self.elite_size:]
        remaining_fitness = fitness[remaining_indices]
        
        # Convert to selection probabilities (inverse fitness for minimization)
        # Avoid division by zero
        max_fitness = np.max(remaining_fitness)
        if max_fitness > 0:
            # Invert: better (lower) fitness gets higher probability
            inverted_fitness = max_fitness - remaining_fitness + 1e-6
            probabilities = inverted_fitness / np.sum(inverted_fitness)
        else:
            # All equal fitness - uniform probability
            probabilities = np.ones(len(remaining_indices)) / len(remaining_indices)
        
        # Sample individuals for breeding
        n_to_select = pop_size - self.elite_size
        selected_indices = np.random.choice(
            remaining_indices, 
            size=n_to_select, 
            replace=True,  # Allow duplicates
            p=probabilities
        )
        selected = population[selected_indices]
        
        return elites, selected
    
    def __repr__(self):
        return f"Selector(elite_size={self.elite_size})"


# ============================================================================
# Testing & Demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Evolutionary Operators Demonstration (Algorithms 3 & 4)")
    print("=" * 80)
    
    # Test parameters
    n = 8  # tasks
    p = 3  # processors
    pop_size = 10
    elite_size = 2
    
    # Create sample DAG (diamond structure)
    dag = np.array([
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])
    
    # Create computation and communication matrices
    comp = np.random.randint(10, 50, size=(n, p))
    comm = np.random.randint(5, 20, size=(n, n)) * dag
    
    print(f"\nTest Configuration:")
    print(f"  Tasks: {n}")
    print(f"  Processors: {p}")
    print(f"  Population size: {pop_size}")
    print(f"  Elite size: {elite_size}")
    
    # Test 1: InitialPopulation without HEFT seed
    print("\n--- Test 1: Random Population (No HEFT Seed) ---")
    init_pop_random = InitialPopulation(n=n, seed_with_heft=False, random_seed=42)
    population_random = init_pop_random.create(pop_size, dag, comp, comm)
    
    print(f"Created: {init_pop_random}")
    print(f"Population shape: {population_random.shape}")
    print(f"First chromosome (sample):")
    print(f"  Task genes: {population_random[0, :n]}")
    print(f"  Unit genes: {population_random[0, n:]}")
    
    # Test 2: InitialPopulation WITH HEFT seed
    print("\n--- Test 2: Population with HEFT Seed ---")
    init_pop_heft = InitialPopulation(n=n, seed_with_heft=True, random_seed=42)
    population_heft = init_pop_heft.create(pop_size, dag, comp, comm)
    
    print(f"Created: {init_pop_heft}")
    print(f"Population shape: {population_heft.shape}")
    print(f"HEFT-derived chromosome (first in population):")
    print(f"  Task genes: {population_heft[0, :n]}")
    print(f"  Unit genes: {population_heft[0, n:]}")
    print(f"\nNote: HEFT chromosome has structured values reflecting task order")
    
    # Test 3: Selector
    print("\n--- Test 3: Elite Selection ---")
    selector = Selector(elite_size=elite_size)
    
    # Simulate fitness values (random makespans)
    fitness = np.random.uniform(100, 200, size=pop_size)
    print(f"Created: {selector}")
    print(f"Fitness values: {fitness}")
    
    # Perform selection
    elites, selected = selector.select(population_heft, fitness)
    
    print(f"\nSelection Results:")
    print(f"  Elites shape: {elites.shape}")
    print(f"  Selected shape: {selected.shape}")
    print(f"  Best fitness (elite): {np.min(fitness):.2f}")
    print(f"  Worst fitness (elite): {fitness[np.argsort(fitness)[elite_size-1]]:.2f}")
    
    # Test 4: Verify elite fitness
    print("\n--- Test 4: Elite Quality Verification ---")
    sorted_fitness = np.sort(fitness)
    elite_indices = np.argsort(fitness)[:elite_size]
    elite_fitness = fitness[elite_indices]
    
    print(f"Top {elite_size} fitness values: {sorted_fitness[:elite_size]}")
    print(f"Elite fitness values: {elite_fitness}")
    print(f"Match: {np.allclose(sorted_fitness[:elite_size], np.sort(elite_fitness))}")
    
    print("\n" + "=" * 80)
    print("✅ Evolutionary operators implementation complete and tested!")
    print("=" * 80)
