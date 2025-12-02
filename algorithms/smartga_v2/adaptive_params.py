"""
Adaptive parameter control for genetic algorithms.

Dynamically adjusts mutation and crossover rates based on population
convergence and diversity metrics.

Reference: Eiben et al. (1999) "Parameter Control in Evolutionary Algorithms"
"""

from typing import Dict, List, Optional
import numpy as np


class AdaptiveParameterController:
    """
    Controls GA parameters adaptively based on population state.
    
    Adjusts mutation and crossover rates to balance exploration and exploitation:
    - High diversity + slow convergence → Increase exploitation (lower mutation)
    - Low diversity + stagnation → Increase exploration (higher mutation)
    - Fast convergence → Maintain current strategy
    
    Attributes:
        mutation_rate: Current mutation probability
        crossover_rate: Current crossover probability
        history: Track parameter changes over time
    """
    
    def __init__(
        self,
        initial_mutation_rate: float = 0.1,
        initial_crossover_rate: float = 0.9,
        min_mutation_rate: float = 0.01,
        max_mutation_rate: float = 0.3,
        min_crossover_rate: float = 0.6,
        max_crossover_rate: float = 0.99,
        adaptation_strength: float = 0.1
    ):
        """
        Initialize adaptive parameter controller.
        
        Args:
            initial_mutation_rate: Starting mutation probability
            initial_crossover_rate: Starting crossover probability
            min_mutation_rate: Minimum mutation rate bound
            max_mutation_rate: Maximum mutation rate bound
            min_crossover_rate: Minimum crossover rate bound
            max_crossover_rate: Maximum crossover rate bound
            adaptation_strength: How aggressively to adapt (0-1, higher = more aggressive)
        """
        self.mutation_rate = initial_mutation_rate
        self.crossover_rate = initial_crossover_rate
        
        self.min_mutation = min_mutation_rate
        self.max_mutation = max_mutation_rate
        self.min_crossover = min_crossover_rate
        self.max_crossover = max_crossover_rate
        
        self.adaptation_strength = adaptation_strength
        
        # Tracking
        self.history = {
            'mutation_rate': [initial_mutation_rate],
            'crossover_rate': [initial_crossover_rate],
            'diversity': [],
            'improvement': [],
            'reason': []
        }
        
        # Previous state for detecting stagnation
        self.prev_best_fitness = float('inf')
        self.prev_diversity = 0.0
        self.stagnation_counter = 0
        self.improvement_counter = 0
    
    def update(
        self,
        diversity: float,
        best_fitness: float,
        avg_fitness: float,
        generation: int
    ) -> Dict[str, float]:
        """
        Update parameters based on current population state.
        
        Args:
            diversity: Population diversity metric (0-1)
            best_fitness: Best fitness value in current generation
            avg_fitness: Average fitness in current generation
            generation: Current generation number
            
        Returns:
            Dict with updated mutation_rate and crossover_rate
        """
        # Calculate improvement from previous generation
        improvement = self._calculate_improvement(best_fitness, avg_fitness)
        
        # Detect stagnation (no improvement for several generations)
        if improvement < 0.001:  # Less than 0.1% improvement
            self.stagnation_counter += 1
            self.improvement_counter = 0
        else:
            self.stagnation_counter = 0
            self.improvement_counter += 1
        
        # Decide adaptation strategy
        reason = self._decide_adaptation(diversity, improvement, generation)
        
        # Apply adaptation
        self._adapt_parameters(diversity, improvement, reason)
        
        # Store history
        self.history['mutation_rate'].append(self.mutation_rate)
        self.history['crossover_rate'].append(self.crossover_rate)
        self.history['diversity'].append(diversity)
        self.history['improvement'].append(improvement)
        self.history['reason'].append(reason)
        
        # Update previous state
        self.prev_best_fitness = best_fitness
        self.prev_diversity = diversity
        
        return {
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'reason': reason
        }
    
    def _calculate_improvement(self, best_fitness: float, avg_fitness: float) -> float:
        """
        Calculate relative improvement from previous generation.
        
        Returns:
            Improvement ratio (0 = no improvement, 1 = 100% improvement)
        """
        if self.prev_best_fitness == float('inf'):
            return 0.0
        
        if self.prev_best_fitness == 0:
            return 0.0
        
        # Relative improvement
        improvement = (self.prev_best_fitness - best_fitness) / abs(self.prev_best_fitness)
        return max(0.0, improvement)
    
    def _decide_adaptation(
        self,
        diversity: float,
        improvement: float,
        generation: int
    ) -> str:
        """
        Decide which adaptation strategy to apply.
        
        Returns:
            String describing the adaptation reason
        """
        # Case 1: Stagnation (no improvement for multiple generations)
        if self.stagnation_counter >= 5:
            if diversity < 0.2:
                return "stagnation_low_diversity"  # Increase exploration
            else:
                return "stagnation_high_diversity"  # Increase exploitation
        
        # Case 2: Low diversity (premature convergence risk)
        if diversity < 0.15:
            return "low_diversity"  # Increase mutation to restore diversity
        
        # Case 3: High diversity but slow convergence
        if diversity > 0.6 and improvement < 0.01:
            return "high_diversity_slow"  # Increase exploitation
        
        # Case 4: Good improvement
        if improvement > 0.05:
            return "good_improvement"  # Maintain strategy
        
        # Case 5: Early generations (initial exploration)
        if generation < 10:
            return "early_exploration"  # Higher exploration
        
        # Case 6: Late generations (final tuning)
        if generation > 80:
            return "late_exploitation"  # Lower mutation for fine-tuning
        
        # Default: gradual adjustment
        return "normal"
    
    def _adapt_parameters(self, diversity: float, improvement: float, reason: str):
        """Apply parameter adaptation based on strategy."""
        delta_mut = self.adaptation_strength * 0.05  # Small adjustment
        delta_cross = self.adaptation_strength * 0.05
        
        if reason == "stagnation_low_diversity":
            # Increase mutation significantly to escape local optima
            self.mutation_rate = min(self.max_mutation, self.mutation_rate + 0.1)
            self.crossover_rate = max(self.min_crossover, self.crossover_rate - 0.05)
        
        elif reason == "stagnation_high_diversity":
            # Increase exploitation to converge
            self.mutation_rate = max(self.min_mutation, self.mutation_rate - delta_mut)
            self.crossover_rate = min(self.max_crossover, self.crossover_rate + delta_cross)
        
        elif reason == "low_diversity":
            # Increase mutation to restore diversity
            self.mutation_rate = min(self.max_mutation, self.mutation_rate + delta_mut * 2)
        
        elif reason == "high_diversity_slow":
            # Increase exploitation
            self.mutation_rate = max(self.min_mutation, self.mutation_rate - delta_mut)
            self.crossover_rate = min(self.max_crossover, self.crossover_rate + delta_cross)
        
        elif reason == "good_improvement":
            # Maintain current strategy (no change)
            pass
        
        elif reason == "early_exploration":
            # Ensure higher exploration in early generations
            self.mutation_rate = max(self.mutation_rate, 0.15)
            self.crossover_rate = min(self.crossover_rate, 0.85)
        
        elif reason == "late_exploitation":
            # Fine-tuning in late generations
            self.mutation_rate = max(self.min_mutation, self.mutation_rate * 0.9)
            self.crossover_rate = min(self.max_crossover, self.crossover_rate * 1.02)
        
        else:  # "normal"
            # Gradual adjustment toward balanced values
            target_mutation = 0.1
            target_crossover = 0.9
            
            if self.mutation_rate > target_mutation:
                self.mutation_rate = max(target_mutation, self.mutation_rate - delta_mut * 0.5)
            elif self.mutation_rate < target_mutation:
                self.mutation_rate = min(target_mutation, self.mutation_rate + delta_mut * 0.5)
            
            if self.crossover_rate > target_crossover:
                self.crossover_rate = max(target_crossover, self.crossover_rate - delta_cross * 0.5)
            elif self.crossover_rate < target_crossover:
                self.crossover_rate = min(target_crossover, self.crossover_rate + delta_cross * 0.5)
        
        # Ensure bounds
        self.mutation_rate = np.clip(self.mutation_rate, self.min_mutation, self.max_mutation)
        self.crossover_rate = np.clip(self.crossover_rate, self.min_crossover, self.max_crossover)
    
    def get_current_params(self) -> Dict[str, float]:
        """
        Get current parameter values.
        
        Returns:
            Dict with current mutation_rate and crossover_rate
        """
        return {
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate
        }
    
    def get_history(self) -> Dict[str, List]:
        """
        Get parameter adaptation history.
        
        Returns:
            Dict with history of all tracked metrics
        """
        return self.history
    
    def reset(self):
        """Reset controller to initial state."""
        self.mutation_rate = self.history['mutation_rate'][0]
        self.crossover_rate = self.history['crossover_rate'][0]
        self.prev_best_fitness = float('inf')
        self.prev_diversity = 0.0
        self.stagnation_counter = 0
        self.improvement_counter = 0
        
        # Keep history but mark reset point
        self.history['reason'].append('RESET')


class MultiObjectiveAdaptiveController(AdaptiveParameterController):
    """
    Adaptive parameter controller specialized for multi-objective optimization.
    
    Extends base controller to handle multiple objectives and Pareto front metrics.
    Uses hypervolume and Pareto front size to measure convergence.
    """
    
    def __init__(
        self,
        initial_mutation_rate: float = 0.1,
        initial_crossover_rate: float = 0.9,
        min_mutation_rate: float = 0.01,
        max_mutation_rate: float = 0.3,
        min_crossover_rate: float = 0.6,
        max_crossover_rate: float = 0.99,
        adaptation_strength: float = 0.1
    ):
        """Initialize multi-objective adaptive controller."""
        super().__init__(
            initial_mutation_rate=initial_mutation_rate,
            initial_crossover_rate=initial_crossover_rate,
            min_mutation_rate=min_mutation_rate,
            max_mutation_rate=max_mutation_rate,
            min_crossover_rate=min_crossover_rate,
            max_crossover_rate=max_crossover_rate,
            adaptation_strength=adaptation_strength
        )
        
        # Multi-objective specific tracking
        self.prev_pareto_size = 0
        self.prev_hypervolume = 0.0
        self.history['pareto_size'] = []
        self.history['hypervolume'] = []
    
    def update_multi_objective(
        self,
        diversity: float,
        pareto_size: int,
        hypervolume: float,
        generation: int
    ) -> Dict[str, float]:
        """
        Update parameters for multi-objective optimization.
        
        Args:
            diversity: Population diversity metric (0-1)
            pareto_size: Number of solutions in Pareto front
            hypervolume: Hypervolume indicator (convergence + diversity)
            generation: Current generation number
            
        Returns:
            Dict with updated parameters
        """
        # Calculate improvement using hypervolume
        if self.prev_hypervolume == 0:
            improvement = 0.0
        else:
            improvement = (hypervolume - self.prev_hypervolume) / max(abs(self.prev_hypervolume), 1e-6)
            improvement = max(0.0, improvement)
        
        # Detect stagnation
        if improvement < 0.001 and abs(pareto_size - self.prev_pareto_size) <= 1:
            self.stagnation_counter += 1
            self.improvement_counter = 0
        else:
            self.stagnation_counter = 0
            self.improvement_counter += 1
        
        # Decide adaptation
        reason = self._decide_mo_adaptation(diversity, improvement, pareto_size, generation)
        
        # Apply adaptation
        self._adapt_parameters(diversity, improvement, reason)
        
        # Store history
        self.history['mutation_rate'].append(self.mutation_rate)
        self.history['crossover_rate'].append(self.crossover_rate)
        self.history['diversity'].append(diversity)
        self.history['improvement'].append(improvement)
        self.history['pareto_size'].append(pareto_size)
        self.history['hypervolume'].append(hypervolume)
        self.history['reason'].append(reason)
        
        # Update previous state
        self.prev_hypervolume = hypervolume
        self.prev_pareto_size = pareto_size
        self.prev_diversity = diversity
        
        return {
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'reason': reason
        }
    
    def _decide_mo_adaptation(
        self,
        diversity: float,
        improvement: float,
        pareto_size: int,
        generation: int
    ) -> str:
        """Decide adaptation strategy for multi-objective case."""
        # Small Pareto front needs more diversity
        if pareto_size < 5:
            return "small_pareto_front"
        
        # Very large Pareto front might need more convergence pressure
        if pareto_size > 30:
            return "large_pareto_front"
        
        # Otherwise use standard decision logic
        return self._decide_adaptation(diversity, improvement, generation)
