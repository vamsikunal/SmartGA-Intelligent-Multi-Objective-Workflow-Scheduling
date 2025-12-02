"""
Population management for GA.

Handles initialization, statistics, and population operations.
"""

from typing import List, Optional, Callable
import random
import statistics
from .chromosome import Chromosome
from .seeding import PopulationSeeder


class Population:
    """
    Manages a population of chromosomes.
    
    Attributes:
        chromosomes: List of Chromosome objects
        generation: Current generation number
        best_fitness_history: Track best fitness over generations
    """
    
    def __init__(self, chromosomes: Optional[List[Chromosome]] = None):
        """
        Initialize population.
        
        Args:
            chromosomes: Initial list of chromosomes (optional)
        """
        self.chromosomes = chromosomes or []
        self.generation = 0
        self.best_fitness_history = []
    
    @classmethod
    def initialize_random(
        cls,
        pop_size: int,
        num_tasks: int,
        num_vms: int,
        seed: Optional[int] = None
    ) -> 'Population':
        """
        Create a random population.
        
        Args:
            pop_size: Number of chromosomes in population
            num_tasks: Number of tasks in workflow
            num_vms: Number of VMs available
            seed: Random seed for reproducibility
            
        Returns:
            Population with random valid chromosomes
        """
        if seed is not None:
            random.seed(seed)
        
        chromosomes = [
            Chromosome.random(num_tasks, num_vms)
            for _ in range(pop_size)
        ]
        
        return cls(chromosomes=chromosomes)
    
    @classmethod
    def initialize_with_heuristics(
        cls,
        pop_size: int,
        workflow,
        resource_config,
        heuristic_ratio: float = 0.3,
        heuristics: Optional[List[str]] = None,
        seed: Optional[int] = None
    ) -> 'Population':
        """
        Initialize population with mix of heuristic solutions and random.
        
        This is the "smart initialization" for Week 3.
        Seeds population with good solutions from heuristics (HEFT, EFT, MinMin, MaxMin).
        
        Args:
            pop_size: Number of chromosomes in population
            workflow: Workflow object with tasks and dependencies
            resource_config: Resource configuration with VMs
            heuristic_ratio: Fraction of population to initialize with heuristics (default: 0.3)
            heuristics: List of heuristic names to use (default: ['heft', 'eft', 'minmin', 'maxmin'])
            seed: Random seed for reproducibility
            
        Returns:
            Population with mix of heuristic-based and random chromosomes
        """
        if seed is not None:
            random.seed(seed)
        
        # Use PopulationSeeder for smart initialization
        seeder = PopulationSeeder(workflow, resource_config)
        chromosomes = seeder.seed_population(
            pop_size=pop_size,
            heuristic_ratio=heuristic_ratio,
            heuristics=heuristics
        )
        
        return cls(chromosomes=chromosomes)
    
    def evaluate(
        self,
        fitness_func: Callable[[Chromosome], float],
        workflow,
        resource_config
    ) -> None:
        """
        Evaluate fitness for all chromosomes.
        
        Args:
            fitness_func: Function that takes (chromosome, workflow, resource_config)
                         and returns fitness value
            workflow: Workflow object
            resource_config: Resource configuration
        """
        for chrom in self.chromosomes:
            if chrom.fitness == float('inf'):
                chrom.fitness = fitness_func(chrom, workflow, resource_config)
    
    def get_best(self, n: int = 1) -> List[Chromosome]:
        """
        Get the n best chromosomes (lowest fitness).
        
        Args:
            n: Number of best chromosomes to return
            
        Returns:
            List of n best chromosomes sorted by fitness
        """
        sorted_pop = sorted(self.chromosomes, key=lambda c: c.fitness)
        return sorted_pop[:n]
    
    def get_worst(self, n: int = 1) -> List[Chromosome]:
        """
        Get the n worst chromosomes (highest fitness).
        
        Args:
            n: Number of worst chromosomes to return
            
        Returns:
            List of n worst chromosomes sorted by fitness (worst first)
        """
        sorted_pop = sorted(self.chromosomes, key=lambda c: c.fitness, reverse=True)
        return sorted_pop[:n]
    
    def get_best_fitness(self) -> float:
        """Get fitness of best chromosome."""
        if not self.chromosomes:
            return float('inf')
        return min(c.fitness for c in self.chromosomes)
    
    def get_worst_fitness(self) -> float:
        """Get fitness of worst chromosome."""
        if not self.chromosomes:
            return float('inf')
        return max(c.fitness for c in self.chromosomes)
    
    def get_average_fitness(self) -> float:
        """Get average fitness across population."""
        if not self.chromosomes:
            return float('inf')
        return statistics.mean(c.fitness for c in self.chromosomes)
    
    def get_fitness_std(self) -> float:
        """Get standard deviation of fitness."""
        if len(self.chromosomes) < 2:
            return 0.0
        return statistics.stdev(c.fitness for c in self.chromosomes)
    
    def get_statistics(self) -> dict:
        """
        Get comprehensive population statistics.
        
        Returns:
            Dict with best, worst, average, std fitness and diversity metrics
        """
        if not self.chromosomes:
            return {
                'best_fitness': float('inf'),
                'worst_fitness': float('inf'),
                'avg_fitness': float('inf'),
                'std_fitness': 0.0,
                'diversity': 0.0,
                'size': 0
            }
        
        return {
            'best_fitness': self.get_best_fitness(),
            'worst_fitness': self.get_worst_fitness(),
            'avg_fitness': self.get_average_fitness(),
            'std_fitness': self.get_fitness_std(),
            'diversity': self.get_diversity(),
            'size': len(self.chromosomes)
        }
    
    def get_diversity(self) -> float:
        """
        Calculate population diversity.
        
        Diversity is measured as the average Hamming distance
        between all pairs of chromosomes.
        
        Returns:
            Average Hamming distance (0.0 = identical, 1.0 = maximally diverse)
        """
        if len(self.chromosomes) < 2:
            return 0.0
        
        total_distance = 0.0
        num_pairs = 0
        
        for i in range(len(self.chromosomes)):
            for j in range(i + 1, len(self.chromosomes)):
                distance = self._hamming_distance(
                    self.chromosomes[i].genes,
                    self.chromosomes[j].genes
                )
                total_distance += distance
                num_pairs += 1
        
        # Normalize by length of chromosome
        avg_distance = total_distance / num_pairs
        normalized = avg_distance / len(self.chromosomes[0].genes)
        
        return normalized
    
    def _hamming_distance(self, genes1: List[int], genes2: List[int]) -> int:
        """Calculate Hamming distance between two gene sequences."""
        return sum(g1 != g2 for g1, g2 in zip(genes1, genes2))
    
    def select_elites(self, num_elites: int) -> List[Chromosome]:
        """
        Select elite chromosomes for next generation.
        
        Args:
            num_elites: Number of elite chromosomes to preserve
            
        Returns:
            List of elite chromosomes (copies)
        """
        return [c.copy() for c in self.get_best(num_elites)]
    
    def replace_worst(self, new_chromosomes: List[Chromosome]) -> None:
        """
        Replace worst chromosomes with new ones.
        
        Args:
            new_chromosomes: Chromosomes to add to population
        """
        # Sort population by fitness
        self.chromosomes.sort(key=lambda c: c.fitness)
        
        # Replace worst with new
        num_replace = min(len(new_chromosomes), len(self.chromosomes))
        self.chromosomes[-num_replace:] = new_chromosomes[:num_replace]
    
    def add(self, chromosome: Chromosome) -> None:
        """Add a chromosome to population."""
        self.chromosomes.append(chromosome)
    
    def extend(self, chromosomes: List[Chromosome]) -> None:
        """Add multiple chromosomes to population."""
        self.chromosomes.extend(chromosomes)
    
    def clear(self) -> None:
        """Clear all chromosomes."""
        self.chromosomes = []
    
    def next_generation(self) -> None:
        """Increment generation counter and record statistics."""
        self.generation += 1
        self.best_fitness_history.append(self.get_best_fitness())
    
    def __len__(self) -> int:
        """Return population size."""
        return len(self.chromosomes)
    
    def __iter__(self):
        """Iterate over chromosomes."""
        return iter(self.chromosomes)
    
    def __getitem__(self, idx):
        """Get chromosome by index."""
        return self.chromosomes[idx]
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"Population(size={len(self)}, generation={self.generation}, "
                f"best_fitness={self.get_best_fitness():.2f})")
