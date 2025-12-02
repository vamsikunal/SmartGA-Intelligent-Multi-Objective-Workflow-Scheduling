"""
Selection operators for Genetic Algorithm.

Implements various selection strategies for choosing parents:
- Tournament selection (most common)
- Roulette wheel selection (fitness-proportionate)
- Rank-based selection (avoids premature convergence)
"""

from abc import ABC, abstractmethod
from typing import List
import random
from .chromosome import Chromosome


class SelectionOperator(ABC):
    """Base class for selection operators."""
    
    @abstractmethod
    def select(self, population: List[Chromosome], num_parents: int) -> List[Chromosome]:
        """
        Select parents from population.
        
        Args:
            population: List of chromosomes to select from
            num_parents: Number of parents to select
            
        Returns:
            List of selected parent chromosomes
        """
        pass


class TournamentSelection(SelectionOperator):
    """
    Tournament selection operator.
    
    Randomly selects k individuals and chooses the best one.
    This is the most commonly used selection method in modern GAs.
    
    Advantages:
    - Simple and efficient
    - Easy to parallelize
    - Selection pressure controlled by tournament size
    - Works well even with negative fitness values
    """
    
    def __init__(self, tournament_size: int = 3):
        """
        Initialize tournament selection.
        
        Args:
            tournament_size: Number of individuals in each tournament (k).
                            k=2 gives mild selection pressure
                            k=3-5 gives moderate pressure (recommended)
                            k>5 gives high pressure
        """
        if tournament_size < 2:
            raise ValueError("Tournament size must be at least 2")
        self.tournament_size = tournament_size
    
    def select(self, population: List[Chromosome], num_parents: int) -> List[Chromosome]:
        """
        Select parents using tournament selection.
        
        For each parent needed:
        1. Randomly select k individuals
        2. Choose the best one (lowest fitness)
        3. Add to parent pool
        
        Args:
            population: List of chromosomes
            num_parents: Number of parents to select
            
        Returns:
            List of selected parents
        """
        if not population:
            return []
        
        if len(population) < self.tournament_size:
            # Tournament size larger than population, use population size
            tournament_size = len(population)
        else:
            tournament_size = self.tournament_size
        
        parents = []
        for _ in range(num_parents):
            # Run tournament
            tournament = random.sample(population, tournament_size)
            winner = min(tournament, key=lambda c: c.fitness)
            parents.append(winner)
        
        return parents


class RouletteWheelSelection(SelectionOperator):
    """
    Roulette wheel (fitness-proportionate) selection.
    
    Each individual's selection probability is proportional to its fitness.
    For minimization problems, we convert fitness to selection probability
    using fitness transformation.
    
    Note: Can suffer from premature convergence if one individual is
    much better than others. Tournament selection is often preferred.
    """
    
    def __init__(self):
        """Initialize roulette wheel selection."""
        pass
    
    def select(self, population: List[Chromosome], num_parents: int) -> List[Chromosome]:
        """
        Select parents using roulette wheel selection.
        
        For minimization (lower fitness is better):
        1. Convert fitness to selection probability using inverse
        2. Create cumulative probability distribution
        3. Spin wheel (random number) and select individual
        
        Args:
            population: List of chromosomes
            num_parents: Number of parents to select
            
        Returns:
            List of selected parents
        """
        if not population:
            return []
        
        # For minimization, convert fitness to selection probability
        # Use inverse fitness: prob ∝ 1/fitness
        # Add small constant to avoid division by zero
        epsilon = 1e-10
        inverse_fitness = [1.0 / (c.fitness + epsilon) for c in population]
        
        # Normalize to get probabilities
        total = sum(inverse_fitness)
        probabilities = [f / total for f in inverse_fitness]
        
        # Create cumulative distribution
        cumulative = []
        cumsum = 0.0
        for prob in probabilities:
            cumsum += prob
            cumulative.append(cumsum)
        
        # Select parents by spinning the wheel
        parents = []
        for _ in range(num_parents):
            r = random.random()
            # Find first individual with cumulative prob >= r
            for i, cum_prob in enumerate(cumulative):
                if r <= cum_prob:
                    parents.append(population[i])
                    break
        
        return parents


class RankSelection(SelectionOperator):
    """
    Rank-based selection.
    
    Selection probability based on rank rather than raw fitness.
    This avoids problems with:
    - Very large fitness differences (premature convergence)
    - Negative fitness values
    - Early loss of diversity
    
    More robust than roulette wheel, less aggressive than tournament.
    """
    
    def __init__(self, selection_pressure: float = 1.5):
        """
        Initialize rank selection.
        
        Args:
            selection_pressure: Controls selection pressure (1.0-2.0).
                               1.0 = uniform selection (no pressure)
                               2.0 = maximum pressure (best has 2x prob of worst)
                               1.5 = moderate pressure (recommended)
        """
        if not 1.0 <= selection_pressure <= 2.0:
            raise ValueError("Selection pressure must be between 1.0 and 2.0")
        self.selection_pressure = selection_pressure
    
    def select(self, population: List[Chromosome], num_parents: int) -> List[Chromosome]:
        """
        Select parents using rank-based selection.
        
        Linear ranking:
        P(i) = (2 - SP) / N + 2 * i * (SP - 1) / (N * (N - 1))
        where SP = selection pressure, N = population size, i = rank (0 to N-1)
        
        Args:
            population: List of chromosomes
            num_parents: Number of parents to select
            
        Returns:
            List of selected parents
        """
        if not population:
            return []
        
        # Sort population by fitness (best = lowest fitness = rank 0)
        sorted_pop = sorted(population, key=lambda c: c.fitness)
        n = len(sorted_pop)
        sp = self.selection_pressure
        
        # Calculate selection probabilities using linear ranking
        probabilities = []
        for i in range(n):
            # i=0 (best) gets highest probability
            # i=n-1 (worst) gets lowest probability
            prob = (2 - sp) / n + 2 * (n - 1 - i) * (sp - 1) / (n * (n - 1))
            probabilities.append(prob)
        
        # Create cumulative distribution
        cumulative = []
        cumsum = 0.0
        for prob in probabilities:
            cumsum += prob
            cumulative.append(cumsum)
        
        # Select parents
        parents = []
        for _ in range(num_parents):
            r = random.random()
            for i, cum_prob in enumerate(cumulative):
                if r <= cum_prob:
                    parents.append(sorted_pop[i])
                    break
        
        return parents


class ElitistSelection(SelectionOperator):
    """
    Elitist selection - simply select the best individuals.
    
    Used for elitism (preserving best solutions across generations).
    Not used for parent selection as it provides no diversity.
    """
    
    def select(self, population: List[Chromosome], num_parents: int) -> List[Chromosome]:
        """
        Select the best individuals.
        
        Args:
            population: List of chromosomes
            num_parents: Number of individuals to select
            
        Returns:
            Best individuals sorted by fitness
        """
        if not population:
            return []
        
        sorted_pop = sorted(population, key=lambda c: c.fitness)
        return sorted_pop[:num_parents]


class RandomSelection(SelectionOperator):
    """
    Random selection - uniform random sampling.
    
    Provides maximum diversity but no selection pressure.
    Useful for testing or as part of hybrid strategies.
    """
    
    def select(self, population: List[Chromosome], num_parents: int) -> List[Chromosome]:
        """
        Randomly select individuals.
        
        Args:
            population: List of chromosomes
            num_parents: Number of individuals to select
            
        Returns:
            Randomly selected individuals
        """
        if not population:
            return []
        
        if num_parents >= len(population):
            return random.sample(population, len(population))
        else:
            return random.sample(population, num_parents)
