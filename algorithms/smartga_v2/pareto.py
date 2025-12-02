"""
Pareto Dominance and NSGA-II Algorithms

This module implements Pareto dominance checking, non-dominated sorting,
and crowding distance calculation for multi-objective optimization.
Used by NSGA-II to maintain diversity and convergence toward Pareto front.
"""

from typing import List, Tuple, Set
import numpy as np


class ParetoComparator:
    """Handles Pareto dominance comparisons between solutions."""
    
    @staticmethod
    def dominates(obj1: Tuple[float, ...], obj2: Tuple[float, ...]) -> bool:
        """
        Check if obj1 Pareto-dominates obj2 (all objectives minimize).
        
        Solution A dominates B if:
        - A is no worse than B in all objectives
        - A is strictly better than B in at least one objective
        
        Args:
            obj1: Tuple of objective values for solution 1
            obj2: Tuple of objective values for solution 2
            
        Returns:
            True if obj1 dominates obj2, False otherwise
        """
        if len(obj1) != len(obj2):
            raise ValueError(f"Objective vectors must have same length: {len(obj1)} vs {len(obj2)}")
        
        # Check if obj1 is no worse in all objectives and strictly better in at least one
        better_in_any = False
        for v1, v2 in zip(obj1, obj2):
            if v1 > v2:  # Worse in this objective (minimizing)
                return False
            if v1 < v2:  # Better in this objective
                better_in_any = True
        
        return better_in_any
    
    @staticmethod
    def is_non_dominated(obj: Tuple[float, ...], population_objs: List[Tuple[float, ...]]) -> bool:
        """
        Check if a solution is non-dominated in a population.
        
        Args:
            obj: Objective values to check
            population_objs: List of all objective values in population
            
        Returns:
            True if obj is non-dominated, False otherwise
        """
        for other_obj in population_objs:
            if other_obj == obj:
                continue
            if ParetoComparator.dominates(other_obj, obj):
                return False
        return True


class FastNonDominatedSort:
    """NSGA-II fast non-dominated sorting algorithm."""
    
    @staticmethod
    def sort(objectives: List[Tuple[float, ...]]) -> List[List[int]]:
        """
        Perform fast non-dominated sorting on a population.
        
        Deb's fast non-dominated sorting algorithm (O(MN²) complexity):
        1. For each solution, find which solutions it dominates and count how many dominate it
        2. Solutions with domination count 0 form the first front
        3. Remove first front and repeat to find subsequent fronts
        
        Args:
            objectives: List of objective tuples for each solution
            
        Returns:
            List of fronts, where each front is a list of solution indices
        """
        n = len(objectives)
        if n == 0:
            return []
        
        # For each solution: which solutions does it dominate
        dominated_solutions = [set() for _ in range(n)]
        # For each solution: how many solutions dominate it
        domination_count = [0] * n
        
        # Compare all pairs of solutions
        for i in range(n):
            for j in range(i + 1, n):
                if ParetoComparator.dominates(objectives[i], objectives[j]):
                    dominated_solutions[i].add(j)
                    domination_count[j] += 1
                elif ParetoComparator.dominates(objectives[j], objectives[i]):
                    dominated_solutions[j].add(i)
                    domination_count[i] += 1
        
        # Build fronts
        fronts = []
        current_front = [i for i in range(n) if domination_count[i] == 0]
        
        while current_front:
            fronts.append(current_front)
            next_front = []
            
            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            current_front = next_front
        
        return fronts


class CrowdingDistance:
    """NSGA-II crowding distance calculation for diversity preservation."""
    
    @staticmethod
    def calculate(objectives: List[Tuple[float, ...]], front_indices: List[int]) -> List[float]:
        """
        Calculate crowding distance for solutions in a front.
        
        Crowding distance measures how close a solution is to its neighbors.
        Higher values indicate more isolated (diverse) solutions.
        Boundary solutions get infinite distance.
        
        Algorithm:
        1. Initialize all distances to 0
        2. For each objective:
           - Sort solutions by that objective
           - Set boundary solutions to infinity
           - For middle solutions: distance += (next - prev) / (max - min)
        
        Args:
            objectives: All objective values
            front_indices: Indices of solutions in this front
            
        Returns:
            List of crowding distances (same length as front_indices)
        """
        n = len(front_indices)
        if n == 0:
            return []
        if n <= 2:
            return [float('inf')] * n
        
        # Get number of objectives
        n_obj = len(objectives[front_indices[0]])
        
        # Initialize distances
        distances = [0.0] * n
        
        # Calculate distance for each objective
        for obj_idx in range(n_obj):
            # Sort solutions by this objective
            sorted_indices = sorted(range(n), key=lambda i: objectives[front_indices[i]][obj_idx])
            
            # Get objective values for this front
            obj_values = [objectives[front_indices[i]][obj_idx] for i in range(n)]
            
            # Find range (avoid division by zero)
            obj_min = obj_values[sorted_indices[0]]
            obj_max = obj_values[sorted_indices[-1]]
            obj_range = obj_max - obj_min
            
            if obj_range < 1e-10:  # All values are the same
                continue
            
            # Boundary solutions get infinite distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Middle solutions get distance based on neighbors
            for i in range(1, n - 1):
                if distances[sorted_indices[i]] != float('inf'):
                    prev_val = obj_values[sorted_indices[i - 1]]
                    next_val = obj_values[sorted_indices[i + 1]]
                    distances[sorted_indices[i]] += (next_val - prev_val) / obj_range
        
        return distances


class ParetoFront:
    """Utilities for working with Pareto fronts."""
    
    @staticmethod
    def extract_first_front(objectives: List[Tuple[float, ...]]) -> List[int]:
        """
        Extract the first (best) Pareto front from a population.
        
        Args:
            objectives: List of objective tuples
            
        Returns:
            Indices of solutions in the first Pareto front
        """
        fronts = FastNonDominatedSort.sort(objectives)
        return fronts[0] if fronts else []
    
    @staticmethod
    def select_best_n(objectives: List[Tuple[float, ...]], n: int) -> List[int]:
        """
        Select n best solutions using NSGA-II selection criteria.
        
        Uses non-dominated sorting and crowding distance to select diverse,
        high-quality solutions.
        
        Args:
            objectives: List of objective tuples
            n: Number of solutions to select
            
        Returns:
            Indices of selected solutions
        """
        if n >= len(objectives):
            return list(range(len(objectives)))
        
        # Perform non-dominated sorting
        fronts = FastNonDominatedSort.sort(objectives)
        
        # Select entire fronts until we reach capacity
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= n:
                selected.extend(front)
            else:
                # Need to select partial front using crowding distance
                remaining = n - len(selected)
                distances = CrowdingDistance.calculate(objectives, front)
                
                # Sort by crowding distance (descending) and take top remaining
                sorted_front = sorted(zip(front, distances), key=lambda x: x[1], reverse=True)
                selected.extend([idx for idx, _ in sorted_front[:remaining]])
                break
        
        return selected
    
    @staticmethod
    def compare_solutions(obj1: Tuple[float, ...], obj2: Tuple[float, ...],
                         dist1: float, dist2: float) -> int:
        """
        Compare two solutions using NSGA-II comparison operator.
        
        Returns:
            -1 if solution 1 is better, 1 if solution 2 is better, 0 if equal
        """
        # Check domination
        if ParetoComparator.dominates(obj1, obj2):
            return -1
        if ParetoComparator.dominates(obj2, obj1):
            return 1
        
        # Neither dominates, use crowding distance
        if dist1 > dist2:
            return -1
        if dist2 > dist1:
            return 1
        
        return 0
