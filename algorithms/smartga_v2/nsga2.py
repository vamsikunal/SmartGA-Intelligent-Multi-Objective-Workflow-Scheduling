"""
NSGA-II (Non-dominated Sorting Genetic Algorithm II)

Complete implementation of NSGA-II for multi-objective optimization.
Integrates non-dominated sorting, crowding distance, and elitist selection.

Reference: Deb et al. (2002) "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II"
"""

from typing import List, Tuple, Optional
import numpy as np
from .chromosome import Chromosome
from .population import Population
from .multi_objective import MultiObjectiveFitness
from .pareto import FastNonDominatedSort, CrowdingDistance, ParetoFront
from .selection import SelectionOperator
from .crossover import CrossoverOperator
from .mutation import MutationOperator
from .utils import Workflow
from .utils import ResourceConfig


class NSGA2Selection(SelectionOperator):
    """
    NSGA-II binary tournament selection using Pareto dominance and crowding distance.
    
    Selection criteria:
    1. If solutions are in different fronts, prefer the one in better front
    2. If solutions are in same front, prefer the one with larger crowding distance
    
    This promotes both convergence (better fronts) and diversity (larger crowding distance).
    """
    
    def __init__(self, tournament_size: int = 2):
        """
        Initialize NSGA-II selection operator.
        
        Args:
            tournament_size: Number of solutions in tournament (typically 2)
        """
        self.tournament_size = tournament_size
    
    def select(
        self, 
        population: Population, 
        fitness_values: List[float],
        n: int = 1
    ) -> List[Chromosome]:
        """
        Select parents using NSGA-II tournament selection.
        
        Args:
            population: Current population
            fitness_values: Not used (NSGA-II uses objectives, not scalar fitness)
            n: Number of parents to select
            
        Returns:
            List of selected parent chromosomes
        """
        if not hasattr(population, '_objectives') or population._objectives is None:
            raise ValueError("Population must have objectives computed. Call evaluate_multi_objective() first.")
        
        if not hasattr(population, '_fronts') or population._fronts is None:
            raise ValueError("Population must have fronts computed. Call compute_fronts() first.")
        
        if not hasattr(population, '_crowding_distances') or population._crowding_distances is None:
            raise ValueError("Population must have crowding distances computed. Call compute_crowding_distance() first.")
        
        selected = []
        pop_size = len(population.chromosomes)
        
        for _ in range(n):
            # Tournament selection
            candidates = np.random.choice(pop_size, size=self.tournament_size, replace=False)
            
            # Find best candidate using NSGA-II comparison
            best_idx = candidates[0]
            best_rank = population._rank_map[best_idx]
            best_distance = population._crowding_distances[best_idx]
            
            for candidate_idx in candidates[1:]:
                candidate_rank = population._rank_map[candidate_idx]
                candidate_distance = population._crowding_distances[candidate_idx]
                
                # Compare: better rank (lower number) or same rank with larger distance
                if (candidate_rank < best_rank or 
                    (candidate_rank == best_rank and candidate_distance > best_distance)):
                    best_idx = candidate_idx
                    best_rank = candidate_rank
                    best_distance = candidate_distance
            
            selected.append(population.chromosomes[best_idx])
        
        return selected


class NSGA2EnvironmentalSelection:
    """
    NSGA-II environmental selection (survivor selection) using elitism.
    
    Combines parent and offspring populations, then selects best N solutions
    using non-dominated sorting and crowding distance.
    """
    
    @staticmethod
    def select(
        parents: List[Chromosome],
        offspring: List[Chromosome],
        objectives_parents: List[Tuple[float, ...]],
        objectives_offspring: List[Tuple[float, ...]],
        n: int
    ) -> Tuple[List[Chromosome], List[Tuple[float, ...]]]:
        """
        Select N best solutions from combined parent and offspring populations.
        
        Args:
            parents: Parent chromosomes
            offspring: Offspring chromosomes
            objectives_parents: Objective values for parents
            objectives_offspring: Objective values for offspring
            n: Number of solutions to select
            
        Returns:
            Tuple of (selected chromosomes, selected objectives)
        """
        # Combine populations
        combined_chromosomes = parents + offspring
        combined_objectives = objectives_parents + objectives_offspring
        
        if len(combined_chromosomes) <= n:
            return combined_chromosomes, combined_objectives
        
        # Perform non-dominated sorting
        fronts = FastNonDominatedSort.sort(combined_objectives)
        
        # Select solutions front by front
        selected_indices = []
        
        for front in fronts:
            if len(selected_indices) + len(front) <= n:
                # Include entire front
                selected_indices.extend(front)
            else:
                # Partial front selection using crowding distance
                remaining = n - len(selected_indices)
                
                # Calculate crowding distance for this front
                distances = CrowdingDistance.calculate(combined_objectives, front)
                
                # Sort by crowding distance (descending) and select top remaining
                front_with_distances = list(zip(front, distances))
                front_with_distances.sort(key=lambda x: x[1], reverse=True)
                
                selected_indices.extend([idx for idx, _ in front_with_distances[:remaining]])
                break
        
        # Extract selected chromosomes and objectives
        selected_chromosomes = [combined_chromosomes[i] for i in selected_indices]
        selected_objectives = [combined_objectives[i] for i in selected_indices]
        
        return selected_chromosomes, selected_objectives


class NSGA2:
    """
    Complete NSGA-II algorithm for multi-objective workflow scheduling.
    
    Features:
    - Multi-objective optimization (makespan, cost, energy, load balance)
    - Fast non-dominated sorting
    - Crowding distance for diversity
    - Elitist selection
    - Integration with Week 1 GA operators
    
    Example:
        >>> nsga2 = NSGA2(workflow, resources, pop_size=100)
        >>> nsga2.run(generations=50)
        >>> pareto_front = nsga2.get_pareto_front()
    """
    
    def __init__(
        self,
        workflow: Workflow,
        resources: ResourceConfig,
        pop_size: int = 100,
        crossover_operator: Optional[CrossoverOperator] = None,
        mutation_operator: Optional[MutationOperator] = None,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1
    ):
        """
        Initialize NSGA-II algorithm.
        
        Args:
            workflow: Workflow to schedule
            resources: Available computing resources
            pop_size: Population size
            crossover_operator: Crossover operator (default: single-point)
            mutation_operator: Mutation operator (default: swap)
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability
        """
        self.workflow = workflow
        self.resources = resources
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        # Initialize operators
        if crossover_operator is None:
            from .crossover import SinglePointCrossover
            crossover_operator = SinglePointCrossover()
        if mutation_operator is None:
            from .mutation import SwapMutation
            mutation_operator = SwapMutation()
        
        self.crossover_op = crossover_operator
        self.mutation_op = mutation_operator
        self.selection_op = NSGA2Selection(tournament_size=2)
        
        # Initialize fitness evaluator
        self.fitness_evaluator = MultiObjectiveFitness(workflow, resources)
        
        # Initialize population
        self.population = None
        self.objectives = None
        self.fronts = None
        
        # Evolution tracking
        self.generation = 0
        self.history = {
            'hypervolume': [],
            'pareto_size': [],
            'diversity': []
        }
    
    def initialize(self, use_heuristic_seeding: bool = False, heuristic_ratio: float = 0.3):
        """
        Initialize population with random or smart heuristic seeding.
        
        Args:
            use_heuristic_seeding: If True, use heuristic seeding for smart initialization
            heuristic_ratio: Fraction of population to initialize with heuristics (default: 0.3)
        """
        if use_heuristic_seeding:
            # Smart initialization with heuristic seeding (Week 3)
            self.population = Population.initialize_with_heuristics(
                pop_size=self.pop_size,
                workflow=self.workflow,
                resource_config=self.resources,
                heuristic_ratio=heuristic_ratio
            )
        else:
            # Random initialization (standard NSGA-II)
            self.population = Population.initialize_random(
                pop_size=self.pop_size,
                num_tasks=len(self.workflow.tasks),
                num_vms=self.resources.num_processors
            )
        
        # Evaluate initial population
        self._evaluate_population()
        self.generation = 0
    
    def _evaluate_population(self):
        """Evaluate all objectives for current population."""
        self.objectives = [
            self.fitness_evaluator.evaluate(chrom) 
            for chrom in self.population.chromosomes
        ]
        
        # Store objectives in population for selection
        self.population._objectives = self.objectives
        
        # Compute fronts and crowding distances
        self._compute_fronts_and_distances()
    
    def _compute_fronts_and_distances(self):
        """Compute Pareto fronts and crowding distances for current population."""
        # Perform non-dominated sorting
        self.fronts = FastNonDominatedSort.sort(self.objectives)
        self.population._fronts = self.fronts
        
        # Create rank map (which front each solution belongs to)
        rank_map = {}
        for rank, front in enumerate(self.fronts):
            for idx in front:
                rank_map[idx] = rank
        self.population._rank_map = rank_map
        
        # Compute crowding distance for each solution
        crowding_distances = np.zeros(len(self.population.chromosomes))
        for front in self.fronts:
            if len(front) > 0:
                distances = CrowdingDistance.calculate(self.objectives, front)
                for i, idx in enumerate(front):
                    crowding_distances[idx] = distances[i]
        
        self.population._crowding_distances = crowding_distances
    
    def evolve_generation(self):
        """Evolve one generation using NSGA-II."""
        offspring_chromosomes = []
        
        # Generate offspring (pop_size children)
        while len(offspring_chromosomes) < self.pop_size:
            # Select parents using NSGA-II tournament selection
            parent1, parent2 = self.selection_op.select(
                self.population,
                fitness_values=[],  # Not used by NSGA-II
                n=2
            )
            
            # Crossover
            if np.random.random() < self.crossover_prob:
                child1, child2 = self.crossover_op.crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if np.random.random() < self.mutation_prob:
                child1 = self.mutation_op.mutate(child1)
            if np.random.random() < self.mutation_prob:
                child2 = self.mutation_op.mutate(child2)
            
            offspring_chromosomes.extend([child1, child2])
        
        # Truncate to exact pop_size
        offspring_chromosomes = offspring_chromosomes[:self.pop_size]
        
        # Evaluate offspring
        offspring_objectives = [
            self.fitness_evaluator.evaluate(chrom)
            for chrom in offspring_chromosomes
        ]
        
        # Environmental selection (combine parents and offspring, select best N)
        selected_chromosomes, selected_objectives = NSGA2EnvironmentalSelection.select(
            parents=self.population.chromosomes,
            offspring=offspring_chromosomes,
            objectives_parents=self.objectives,
            objectives_offspring=offspring_objectives,
            n=self.pop_size
        )
        
        # Update population
        self.population.chromosomes = selected_chromosomes
        self.objectives = selected_objectives
        self.population._objectives = self.objectives
        
        # Recompute fronts and distances
        self._compute_fronts_and_distances()
        
        self.generation += 1
    
    def run(self, generations: int, verbose: bool = False):
        """
        Run NSGA-II for specified number of generations.
        
        Args:
            generations: Number of generations to evolve
            verbose: Print progress information
        """
        if self.population is None:
            self.initialize()
        
        for gen in range(generations):
            self.evolve_generation()
            
            # Track metrics
            pareto_front = self.get_pareto_front()
            self.history['pareto_size'].append(len(pareto_front))
            
            if verbose and (gen + 1) % 10 == 0:
                print(f"Generation {self.generation}: Pareto front size = {len(pareto_front)}")
    
    def get_pareto_front(self) -> List[Tuple[Chromosome, Tuple[float, ...]]]:
        """
        Get current Pareto front (first front).
        
        Returns:
            List of (chromosome, objectives) tuples for non-dominated solutions
        """
        if self.fronts is None or len(self.fronts) == 0:
            return []
        
        first_front_indices = self.fronts[0]
        pareto_solutions = [
            (self.population.chromosomes[i], self.objectives[i])
            for i in first_front_indices
        ]
        
        return pareto_solutions
    
    def get_best_for_objective(self, objective_idx: int) -> Tuple[Chromosome, Tuple[float, ...]]:
        """
        Get best solution for a specific objective.
        
        Args:
            objective_idx: Index of objective (0=makespan, 1=cost, 2=energy, 3=balance)
            
        Returns:
            Tuple of (best chromosome, its objectives)
        """
        best_idx = min(range(len(self.objectives)), key=lambda i: self.objectives[i][objective_idx])
        return self.population.chromosomes[best_idx], self.objectives[best_idx]
    
    def get_statistics(self) -> dict:
        """
        Get statistics about current population.
        
        Returns:
            Dictionary with statistics
        """
        pareto_front = self.get_pareto_front()
        
        if not pareto_front:
            return {
                'generation': self.generation,
                'pareto_size': 0,
                'objectives': {}
            }
        
        # Extract objectives from Pareto front
        pareto_objectives = [obj for _, obj in pareto_front]
        pareto_array = np.array(pareto_objectives)
        
        stats = {
            'generation': self.generation,
            'pareto_size': len(pareto_front),
            'objectives': {
                'makespan': {
                    'min': float(np.min(pareto_array[:, 0])),
                    'max': float(np.max(pareto_array[:, 0])),
                    'mean': float(np.mean(pareto_array[:, 0]))
                },
                'cost': {
                    'min': float(np.min(pareto_array[:, 1])),
                    'max': float(np.max(pareto_array[:, 1])),
                    'mean': float(np.mean(pareto_array[:, 1]))
                },
                'energy': {
                    'min': float(np.min(pareto_array[:, 2])),
                    'max': float(np.max(pareto_array[:, 2])),
                    'mean': float(np.mean(pareto_array[:, 2]))
                },
                'balance': {
                    'min': float(np.min(pareto_array[:, 3])),
                    'max': float(np.max(pareto_array[:, 3])),
                    'mean': float(np.mean(pareto_array[:, 3]))
                }
            }
        }
        
        return stats
