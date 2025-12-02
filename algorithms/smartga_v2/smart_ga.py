"""
SmartGA: Intelligent Multi-Objective Genetic Algorithm

Combines NSGA-II with:
1. Heuristic population seeding (HEFT, EFT, MinMin, MaxMin)
2. Adaptive mutation and crossover rates
3. Enhanced diversity management

Week 3 deliverable for faster convergence and better solution quality.
"""

from typing import Optional, Dict, List
import numpy as np
from .nsga2 import NSGA2
from .adaptive_params import MultiObjectiveAdaptiveController
from .crossover import CrossoverOperator
from .mutation import MutationOperator
from .utils import Workflow
from .utils import ResourceConfig


class SmartGA:
    """
    Smart Genetic Algorithm with heuristic seeding and adaptive parameters.
    
    Enhancements over standard NSGA-II:
    - Smart initialization: 30% heuristic seeds + 70% random for diversity
    - Adaptive parameters: Dynamically adjust mutation/crossover rates
    - Better convergence: Target 30%+ faster than standard NSGA-II
    
    Attributes:
        nsga2: Underlying NSGA-II algorithm
        param_controller: Adaptive parameter controller
        use_seeding: Whether to use heuristic seeding
        use_adaptive: Whether to use adaptive parameters
    """
    
    def __init__(
        self,
        workflow: Workflow,
        resources: ResourceConfig,
        pop_size: int = 50,
        use_heuristic_seeding: bool = True,
        use_adaptive_params: bool = True,
        heuristic_ratio: float = 0.3,
        initial_mutation_rate: float = 0.1,
        initial_crossover_rate: float = 0.9,
        crossover_operator: Optional[CrossoverOperator] = None,
        mutation_operator: Optional[MutationOperator] = None,
        adaptation_strength: float = 0.1
    ):
        """
        Initialize SmartGA.
        
        Args:
            workflow: Workflow to schedule
            resources: Available computational resources
            pop_size: Population size
            use_heuristic_seeding: Enable smart initialization with heuristics
            use_adaptive_params: Enable adaptive parameter control
            heuristic_ratio: Fraction of population to seed with heuristics (default: 0.3)
            initial_mutation_rate: Starting mutation probability
            initial_crossover_rate: Starting crossover probability
            crossover_operator: Custom crossover operator (optional)
            mutation_operator: Custom mutation operator (optional)
            adaptation_strength: How aggressively to adapt parameters (0-1)
        """
        self.workflow = workflow
        self.resources = resources
        self.pop_size = pop_size
        self.use_seeding = use_heuristic_seeding
        self.use_adaptive = use_adaptive_params
        self.heuristic_ratio = heuristic_ratio
        
        # Initialize NSGA-II with initial parameters
        self.nsga2 = NSGA2(
            workflow=workflow,
            resources=resources,
            pop_size=pop_size,
            crossover_operator=crossover_operator,
            mutation_operator=mutation_operator,
            crossover_prob=initial_crossover_rate,
            mutation_prob=initial_mutation_rate
        )
        
        # Initialize adaptive parameter controller
        if use_adaptive_params:
            self.param_controller = MultiObjectiveAdaptiveController(
                initial_mutation_rate=initial_mutation_rate,
                initial_crossover_rate=initial_crossover_rate,
                adaptation_strength=adaptation_strength
            )
        else:
            self.param_controller = None
        
        # Tracking
        self.generation = 0
        self.history = {
            'hypervolume': [],
            'pareto_size': [],
            'diversity': [],
            'best_makespan': [],
            'mutation_rate': [initial_mutation_rate],
            'crossover_rate': [initial_crossover_rate],
            'adaptation_reason': []
        }
    
    def initialize(self):
        """
        Initialize population with smart seeding.
        
        Uses heuristic seeding if enabled, otherwise random initialization.
        """
        self.nsga2.initialize(
            use_heuristic_seeding=self.use_seeding,
            heuristic_ratio=self.heuristic_ratio
        )
        self.generation = 0
        
        # Record initial statistics
        self._record_statistics()
    
    def run(
        self,
        num_generations: int,
        verbose: bool = False,
        callback: Optional[callable] = None
    ) -> Dict:
        """
        Run SmartGA for specified number of generations.
        
        Args:
            num_generations: Number of generations to evolve
            verbose: Print progress information
            callback: Optional callback function(generation, stats) called each generation
            
        Returns:
            Dict with final statistics and Pareto front
        """
        # Initialize if not already done
        if self.nsga2.population is None:
            self.initialize()
        
        for gen in range(num_generations):
            # Update adaptive parameters if enabled
            if self.use_adaptive and self.param_controller is not None:
                self._update_adaptive_parameters()
            
            # Evolve one generation
            self.nsga2.evolve_generation()
            self.generation += 1
            
            # Record statistics
            self._record_statistics()
            
            # Print progress
            if verbose and gen % 10 == 0:
                stats = self.get_current_statistics()
                print(f"Generation {gen}: "
                      f"Pareto size = {stats['pareto_size']}, "
                      f"Diversity = {stats['diversity']:.3f}, "
                      f"Best makespan = {stats['best_makespan']:.2f}")
                if self.use_adaptive:
                    print(f"  → Mutation = {stats['mutation_rate']:.4f}, "
                          f"Crossover = {stats['crossover_rate']:.4f}, "
                          f"Reason = {stats['adaptation_reason']}")
            
            # Call callback if provided
            if callback is not None:
                stats = self.get_current_statistics()
                callback(gen, stats)
        
        # Return final results
        return self.get_results()
    
    def _update_adaptive_parameters(self):
        """Update mutation and crossover rates based on population state."""
        stats = self.nsga2.get_statistics()
        
        # Get diversity from population
        diversity = self.nsga2.population.get_diversity()
        
        # Update controller
        result = self.param_controller.update_multi_objective(
            diversity=diversity,
            pareto_size=stats['pareto_size'],
            hypervolume=stats.get('hypervolume', 0.0),
            generation=self.generation
        )
        
        # Apply updated parameters to NSGA-II
        self.nsga2.mutation_prob = result['mutation_rate']
        self.nsga2.crossover_prob = result['crossover_rate']
        
        # Store reason for tracking
        self.history['adaptation_reason'].append(result['reason'])
    
    def _record_statistics(self):
        """Record current generation statistics."""
        stats = self.nsga2.get_statistics()
        
        # Get diversity from population
        diversity = self.nsga2.population.get_diversity()
        
        self.history['hypervolume'].append(stats.get('hypervolume', 0.0))
        self.history['pareto_size'].append(stats['pareto_size'])
        self.history['diversity'].append(diversity)
        
        # Get best solution for makespan (objective 0)
        pareto_front = self.nsga2.get_pareto_front()
        if pareto_front:
            best_makespan = min(obj[0] for _, obj in pareto_front)
            self.history['best_makespan'].append(best_makespan)
        else:
            self.history['best_makespan'].append(float('inf'))
        
        # Record current parameters
        self.history['mutation_rate'].append(self.nsga2.mutation_prob)
        self.history['crossover_rate'].append(self.nsga2.crossover_prob)
    
    def get_current_statistics(self) -> Dict:
        """
        Get current generation statistics.
        
        Returns:
            Dict with current metrics
        """
        stats = self.nsga2.get_statistics()
        stats['generation'] = self.generation
        stats['mutation_rate'] = self.nsga2.mutation_prob
        stats['crossover_rate'] = self.nsga2.crossover_prob
        stats['diversity'] = self.nsga2.population.get_diversity()
        
        # Add best makespan from history
        if self.history['best_makespan']:
            stats['best_makespan'] = self.history['best_makespan'][-1]
        else:
            stats['best_makespan'] = float('inf')
        
        if self.use_adaptive and len(self.history['adaptation_reason']) > 0:
            stats['adaptation_reason'] = self.history['adaptation_reason'][-1]
        else:
            stats['adaptation_reason'] = 'none'
        
        return stats
    
    def get_results(self) -> Dict:
        """
        Get final results after evolution.
        
        Returns:
            Dict with Pareto front, history, and final statistics
        """
        pareto_front = self.nsga2.get_pareto_front()
        
        return {
            'pareto_front': pareto_front,
            'pareto_size': len(pareto_front),
            'history': self.history,
            'final_generation': self.generation,
            'final_diversity': self.history['diversity'][-1] if self.history['diversity'] else 0.0,
            'final_hypervolume': self.history['hypervolume'][-1] if self.history['hypervolume'] else 0.0,
            'best_makespan': self.history['best_makespan'][-1] if self.history['best_makespan'] else float('inf'),
            'used_seeding': self.use_seeding,
            'used_adaptive': self.use_adaptive
        }
    
    def get_pareto_front(self) -> List[Dict]:
        """
        Get current Pareto front solutions.
        
        Returns:
            List of dicts with chromosome and objectives for each Pareto solution
        """
        pareto_tuples = self.nsga2.get_pareto_front()
        return [
            {'chromosome': chrom, 'objectives': obj}
            for chrom, obj in pareto_tuples
        ]
    
    def get_best_for_objective(self, objective_idx: int) -> Dict:
        """
        Get best solution for specific objective.
        
        Args:
            objective_idx: Objective index (0=makespan, 1=cost, 2=energy, 3=load_balance)
            
        Returns:
            Dict with chromosome and objectives for best solution
        """
        chrom, obj = self.nsga2.get_best_for_objective(objective_idx)
        return {'chromosome': chrom, 'objectives': obj}
    
    def get_history(self) -> Dict:
        """
        Get evolution history.
        
        Returns:
            Dict with tracked metrics over generations
        """
        return self.history
    
    def reset(self):
        """Reset SmartGA to initial state."""
        self.nsga2.population = None
        self.nsga2.objectives = None
        self.nsga2.fronts = None
        self.generation = 0
        
        # Reset history
        initial_mut = self.history['mutation_rate'][0]
        initial_cross = self.history['crossover_rate'][0]
        
        self.history = {
            'hypervolume': [],
            'pareto_size': [],
            'diversity': [],
            'best_makespan': [],
            'mutation_rate': [initial_mut],
            'crossover_rate': [initial_cross],
            'adaptation_reason': []
        }
        
        # Reset adaptive controller
        if self.param_controller is not None:
            self.param_controller.reset()


def compare_with_standard_nsga2(
    workflow: Workflow,
    resources: ResourceConfig,
    num_generations: int = 100,
    pop_size: int = 50,
    num_runs: int = 5,
    verbose: bool = True
) -> Dict:
    """
    Compare SmartGA vs standard NSGA-II performance.
    
    Runs both algorithms multiple times and reports convergence speed and
    final solution quality.
    
    Args:
        workflow: Workflow to schedule
        resources: Computational resources
        num_generations: Generations to evolve
        pop_size: Population size
        num_runs: Number of independent runs
        verbose: Print progress
        
    Returns:
        Dict with comparison statistics
    """
    smart_results = []
    standard_results = []
    
    for run in range(num_runs):
        if verbose:
            print(f"\n=== Run {run + 1}/{num_runs} ===")
        
        # Run SmartGA
        if verbose:
            print("Running SmartGA...")
        smart_ga = SmartGA(
            workflow=workflow,
            resources=resources,
            pop_size=pop_size,
            use_heuristic_seeding=True,
            use_adaptive_params=True
        )
        smart_result = smart_ga.run(num_generations=num_generations, verbose=False)
        smart_results.append(smart_result)
        
        # Run standard NSGA-II
        if verbose:
            print("Running standard NSGA-II...")
        standard_ga = SmartGA(
            workflow=workflow,
            resources=resources,
            pop_size=pop_size,
            use_heuristic_seeding=False,
            use_adaptive_params=False
        )
        standard_result = standard_ga.run(num_generations=num_generations, verbose=False)
        standard_results.append(standard_result)
    
    # Analyze results
    def convergence_generation(history, threshold=0.9):
        """Find generation where 90% of final hypervolume is reached."""
        if not history['hypervolume']:
            return num_generations
        
        final_hv = history['hypervolume'][-1]
        target_hv = final_hv * threshold
        
        for gen, hv in enumerate(history['hypervolume']):
            if hv >= target_hv:
                return gen
        
        return num_generations
    
    smart_convergence = [convergence_generation(r['history']) for r in smart_results]
    standard_convergence = [convergence_generation(r['history']) for r in standard_results]
    
    smart_final_hv = [r['final_hypervolume'] for r in smart_results]
    standard_final_hv = [r['final_hypervolume'] for r in standard_results]
    
    smart_best_makespan = [r['best_makespan'] for r in smart_results]
    standard_best_makespan = [r['best_makespan'] for r in standard_results]
    
    # Calculate improvements
    avg_smart_conv = np.mean(smart_convergence)
    avg_standard_conv = np.mean(standard_convergence)
    convergence_speedup = (avg_standard_conv - avg_smart_conv) / avg_standard_conv * 100
    
    avg_smart_hv = np.mean(smart_final_hv)
    avg_standard_hv = np.mean(standard_final_hv)
    hv_improvement = (avg_smart_hv - avg_standard_hv) / avg_standard_hv * 100
    
    avg_smart_makespan = np.mean(smart_best_makespan)
    avg_standard_makespan = np.mean(standard_best_makespan)
    makespan_improvement = (avg_standard_makespan - avg_smart_makespan) / avg_standard_makespan * 100
    
    comparison = {
        'smart_convergence_gen': {
            'mean': avg_smart_conv,
            'std': np.std(smart_convergence),
            'values': smart_convergence
        },
        'standard_convergence_gen': {
            'mean': avg_standard_conv,
            'std': np.std(standard_convergence),
            'values': standard_convergence
        },
        'convergence_speedup_percent': convergence_speedup,
        'smart_final_hypervolume': {
            'mean': avg_smart_hv,
            'std': np.std(smart_final_hv),
            'values': smart_final_hv
        },
        'standard_final_hypervolume': {
            'mean': avg_standard_hv,
            'std': np.std(standard_final_hv),
            'values': standard_final_hv
        },
        'hypervolume_improvement_percent': hv_improvement,
        'smart_best_makespan': {
            'mean': avg_smart_makespan,
            'std': np.std(smart_best_makespan),
            'values': smart_best_makespan
        },
        'standard_best_makespan': {
            'mean': avg_standard_makespan,
            'std': np.std(standard_best_makespan),
            'values': standard_best_makespan
        },
        'makespan_improvement_percent': makespan_improvement,
        'num_runs': num_runs,
        'num_generations': num_generations
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("COMPARISON RESULTS")
        print(f"{'='*60}")
        print(f"Convergence speed:")
        print(f"  SmartGA:        {avg_smart_conv:.1f} ± {np.std(smart_convergence):.1f} generations")
        print(f"  Standard NSGA2: {avg_standard_conv:.1f} ± {np.std(standard_convergence):.1f} generations")
        print(f"  → Speedup:      {convergence_speedup:+.1f}% {'✓' if convergence_speedup > 30 else '✗'}")
        print(f"\nFinal hypervolume:")
        print(f"  SmartGA:        {avg_smart_hv:.4f} ± {np.std(smart_final_hv):.4f}")
        print(f"  Standard NSGA2: {avg_standard_hv:.4f} ± {np.std(standard_final_hv):.4f}")
        print(f"  → Improvement:  {hv_improvement:+.1f}%")
        print(f"\nBest makespan:")
        print(f"  SmartGA:        {avg_smart_makespan:.2f} ± {np.std(smart_best_makespan):.2f}")
        print(f"  Standard NSGA2: {avg_standard_makespan:.2f} ± {np.std(standard_best_makespan):.2f}")
        print(f"  → Improvement:  {makespan_improvement:+.1f}%")
        print(f"{'='*60}\n")
    
    return comparison
