"""
R²GA - Real Relative Encoding Genetic Algorithm

Adapted to use the BaseScheduler interface for MADGA project.

Based on: J. Jiang, Z. Sun, R. Lu, L. Pan, and Z. Peng,
"Real Relative Encoding Genetic Algorithm for Workflow Scheduling in 
Heterogeneous Distributed Computing Systems," 
IEEE Transactions on Parallel and Distributed Systems, vol. 36, no. 1, 
pp. 1-14, Jan. 2025.

Algorithm:
1. Encoding: Generate 2n-length chromosome with random values [0,1)
2. Decoding: Translate chromosome to task-processor mapping using CTS
3. Initial Population: Create pop_size chromosomes (with optional HEFT seed)
4. Evolution Loop:
   - Evaluate fitness (makespan)
   - Selection (elitism)
   - Crossover (two-point crossover - Algorithm 5)
   - Mutation (two-point random replacement - Algorithm 6)
   - Early stopping if converged
5. Return best solution
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

try:
    # Relative imports (when used as a module)
    from baselines.base_scheduler import BaseScheduler, ResourceConfig, ScheduleResult
    from .encoding import Encoder
    from .decoding import Decoder
    from .evolutionary_operators import InitialPopulation, Selector
    from .genetic_operators import Crossover, Mutator
except ImportError:
    # Absolute imports (when run standalone)
    from baselines.base_scheduler import BaseScheduler, ResourceConfig, ScheduleResult
    from encoding import Encoder
    from decoding import Decoder
    from evolutionary_operators import InitialPopulation, Selector
    from genetic_operators import Crossover, Mutator


class R2GA(BaseScheduler):
    """
    R²GA (Real Relative Encoding Genetic Algorithm) Scheduling Algorithm.
    
    A genetic algorithm-based metaheuristic that evolves a population
    of candidate schedules over multiple iterations.
    
    Key Features:
    - Real-valued relative encoding (2n chromosome)
    - HEFT-seeded initial population for quality
    - Elite selection preserving best solutions
    - Two-point crossover (Algorithm 5) respecting chromosome structure
    - Two-point random replacement mutation (Algorithm 6)
    - Iterative improvement through evolution
    
    Time Complexity: O(iterations × pop_size × n²)
    Space Complexity: O(pop_size × n)
    
    GA Parameters:
    - pop_size: Population size (default: 50)
    - elite_size: Number of elites to preserve (default: 5)
    - iterations: Maximum number of generations (default: 100)
    - mutation_rate: Probability of mutation per chromosome (default: 0.1)
    - crossover_rate: Probability of crossover per pair (default: 0.9)
    - seed_with_heft: Include HEFT solution in initial population (default: True)
    - convergence_threshold: Iterations without improvement for early stopping (default: 20)
    """
    
    def __init__(self, 
                 pop_size: int = 50,
                 elite_size: int = 5,
                 iterations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.9,
                 seed_with_heft: bool = True,
                 convergence_threshold: int = 20,
                 verbose: bool = False):
        """
        Initialize R²GA scheduler.
        
        Args:
            pop_size: Population size (number of chromosomes)
            elite_size: Number of elite solutions to preserve each generation
            iterations: Maximum number of generations to evolve
            mutation_rate: Probability of mutation per chromosome [0, 1]
            crossover_rate: Probability of crossover per parent pair [0, 1]
            seed_with_heft: Whether to include HEFT solution in initial population
            convergence_threshold: Number of iterations without improvement for early stopping
            verbose: Enable verbose logging
        """
        super().__init__(name="R2GA")
        
        # GA parameters
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.iterations = iterations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.seed_with_heft = seed_with_heft
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose
        
        # Validation
        if elite_size >= pop_size:
            raise ValueError(f"elite_size ({elite_size}) must be < pop_size ({pop_size})")
        if not 0 <= mutation_rate <= 1:
            raise ValueError(f"mutation_rate must be in [0, 1], got {mutation_rate}")
        if not 0 <= crossover_rate <= 1:
            raise ValueError(f"crossover_rate must be in [0, 1], got {crossover_rate}")
        
        self._log(f"R²GA initialized: pop={pop_size}, elite={elite_size}, "
                  f"iter={iterations}, mut={mutation_rate:.2f}, "
                  f"cross={crossover_rate:.2f}, heft_seed={seed_with_heft}")
    
    def schedule(self, workflow, resources: ResourceConfig, 
                 current_time: float = 0.0) -> ScheduleResult:
        """
        Schedule workflow using R²GA genetic algorithm.
        
        Args:
            workflow: Workflow object or legacy format (n, W, C, dag, predecessors)
            resources: Resource configuration
            current_time: Current simulation time (for dynamic scheduling)
            
        Returns:
            ScheduleResult with complete schedule and GA-specific metadata
        """
        start_time = time.time()
        
        # Convert input to internal format
        # Returns: (n, C, W, dag, predecessors)
        # where C is computation (n x p) and W is communication (n x n)
        n, comp_matrix, comm_matrix, dag, predecessors = self._convert_input(workflow, resources)
        p = resources.num_processors
        
        self._log(f"Scheduling {n} tasks on {p} processors using R²GA")
        self._log(f"Population: {self.pop_size}, Iterations: {self.iterations}")
        
        # Initialize GA components
        encoder = Encoder(n=n)
        
        # Convert dag to list of edges for Decoder
        dag_edges = []
        for parent_id, children in dag.items():
            for child_id in children:
                dag_edges.append((parent_id, child_id))
        
        # Convert predecessors list to dict for Decoder
        predecessors_dict = {i: predecessors[i] if i < len(predecessors) else [] for i in range(n)}
        
        # Decoder expects W=computation (n x p), C=communication (n x n)
        decoder = Decoder(n=n, p=p, W=comp_matrix, C=comm_matrix, dag=dag_edges, predecessors=predecessors_dict)
        init_pop = InitialPopulation(n=n, seed_with_heft=self.seed_with_heft)
        selector = Selector(elite_size=self.elite_size)
        crossover = Crossover(crossover_rate=self.crossover_rate)
        mutator = Mutator(mutation_rate=self.mutation_rate)
        
        # Create initial population
        # InitialPopulation.create expects: (pop_size, dag_matrix, comp_matrix, comm_matrix)
        # We need to convert dag back to matrix form for HEFT
        dag_matrix = np.zeros((n, n))
        for parent_id, children in dag.items():
            for child_id in children:
                dag_matrix[parent_id, child_id] = 1
        
        population = init_pop.create(self.pop_size, dag_matrix, comp_matrix, comm_matrix)
        
        # Evolution metrics tracking
        best_fitness_history = []
        avg_fitness_history = []
        best_chromosome = None
        best_fitness = float('inf')
        stagnation_counter = 0
        
        # Evolution loop
        for iteration in range(self.iterations):
            # Evaluate fitness for all chromosomes
            fitness_values = np.array([
                self._calculate_fitness(chromosome, decoder, comp_matrix, comm_matrix, dag, predecessors, p)
                for chromosome in population
            ])
            
            # Track best solution
            iter_best_idx = np.argmin(fitness_values)
            iter_best_fitness = fitness_values[iter_best_idx]
            
            if iter_best_fitness < best_fitness:
                best_fitness = iter_best_fitness
                best_chromosome = population[iter_best_idx].copy()
                stagnation_counter = 0
                self._log(f"Iteration {iteration+1}: New best fitness = {best_fitness:.2f}")
            else:
                stagnation_counter += 1
            
            # Track metrics
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(np.mean(fitness_values))
            
            # Check early stopping
            if stagnation_counter >= self.convergence_threshold:
                self._log(f"Early stopping at iteration {iteration+1} "
                         f"(no improvement for {self.convergence_threshold} iterations)")
                break
            
            # Selection: Get elites and selected for breeding
            elites, selected = selector.select(population, fitness_values)
            
            # Generate offspring through crossover and mutation
            offspring = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % len(selected)]  # Handle odd population
                
                # Crossover
                child1, child2 = crossover.apply(parent1, parent2)
                
                # Mutation
                child1 = mutator.apply(child1)
                child2 = mutator.apply(child2)
                
                offspring.extend([child1, child2])
            
            # Create next generation: elites + offspring (truncate to pop_size)
            offspring = np.array(offspring[:self.pop_size - self.elite_size])
            population = np.vstack([elites, offspring])
        
        # Decode best chromosome to schedule
        best_schedule = decoder.decode(best_chromosome)
        
        # Calculate actual start and finish times
        task_assignments, task_start_times, task_finish_times = self._calculate_schedule_times(
            best_schedule, comp_matrix, comm_matrix, dag, predecessors, p
        )
        
        # Create result
        execution_time = time.time() - start_time
        result = ScheduleResult(
            task_assignments=task_assignments,
            task_start_times=task_start_times,
            task_finish_times=task_finish_times,
            makespan=best_fitness
        )
        result.algorithm_name = self.name
        result.execution_time = execution_time
        
        # Compute metrics (SLR, speedup, efficiency, etc.)
        result = self._compute_metrics(result, workflow, resources)
        
        # Add GA-specific metadata
        result.ga_metadata = {
            'iterations_run': min(iteration + 1, self.iterations),
            'best_fitness_history': best_fitness_history,
            'avg_fitness_history': avg_fitness_history,
            'final_population_size': len(population),
            'early_stopped': stagnation_counter >= self.convergence_threshold,
            'convergence_iteration': iteration + 1 if stagnation_counter >= self.convergence_threshold else None
        }
        
        self._log(f"R²GA complete. Makespan: {result.makespan:.2f}, Time: {execution_time:.3f}s")
        
        return result
    
    def _convert_input(self, workflow, resources: ResourceConfig) -> Tuple:
        """
        Convert workflow to internal format.
        
        Handles multiple input formats:
        - Workflow object (from parser)
        - Tuple format (n, W, C, dag, predecessors)
        - Legacy formats
        
        Returns:
            Tuple: (n, C, W, dag, predecessors)
                n: Number of tasks
                C: n×p computation cost matrix
                W: n×n communication cost matrix (task-to-task)
                dag: Dictionary mapping task_id to list of successor task_ids
                predecessors: List of lists mapping task_id to list of predecessor task_ids
        """
        if hasattr(workflow, 'tasks'):
            # Workflow object
            n = len(workflow.tasks)
            p = resources.num_processors
            
            # Build C matrix (computation costs)
            C = np.zeros((n, p))
            for task_id, task in workflow.tasks.items():
                if hasattr(task, 'computation_costs'):
                    C[task_id] = task.computation_costs[:p]
                else:
                    # Default: uniform cost
                    C[task_id] = np.ones(p)
            
            # Build W matrix (communication costs between tasks)
            W = np.zeros((n, n))
            if hasattr(workflow, 'communication_costs'):
                W = workflow.communication_costs[:n, :n]
            
            # Build DAG structure as dictionary
            dag = {i: [] for i in range(n)}
            predecessors = [[] for i in range(n)]
            
            if hasattr(workflow, 'edges'):
                for parent_id, child_id in workflow.edges:
                    dag[parent_id].append(child_id)
                    predecessors[child_id].append(parent_id)
            elif hasattr(workflow, 'dependencies'):
                for task_id, deps in workflow.dependencies.items():
                    for parent_id in deps:
                        dag[parent_id].append(task_id)
                        predecessors[task_id].append(parent_id)
            
            return n, C, W, dag, predecessors
            
        elif isinstance(workflow, tuple):
            # Already in tuple format (n, W, C, dag, predecessors)
            if len(workflow) == 5:
                n, C_or_W, W_or_C, dag_input, predecessors = workflow
                
                # Convert dag to dictionary if it's a list or matrix
                if isinstance(dag_input, dict):
                    dag = dag_input
                elif isinstance(dag_input, np.ndarray):
                    # Matrix format - convert to dictionary
                    dag = {}
                    for i in range(n):
                        dag[i] = [j for j in range(n) if dag_input[i, j] == 1]
                else:
                    # List of edges
                    dag = {i: [] for i in range(n)}
                    for parent, child in dag_input:
                        dag[parent].append(child)
                
                return n, C_or_W, W_or_C, dag, predecessors
            else:
                raise ValueError(f"Invalid tuple format: expected 5 elements, got {len(workflow)}")
        
        else:
            raise ValueError(f"Unsupported workflow format: {type(workflow)}")
    
    def _calculate_fitness(self, chromosome: np.ndarray, decoder: Decoder,
                          comp_matrix: np.ndarray, comm_matrix: np.ndarray, dag: dict, 
                          predecessors: List, p: int) -> float:
        """
        Calculate fitness (makespan) for a chromosome.
        
        Args:
            chromosome: 2n-length array of real values [0, 1)
            decoder: Decoder instance to translate chromosome to schedule
            comp_matrix: Computation cost matrix (n x p)
            comm_matrix: Communication cost matrix (n x n)
            dag: DAG as dictionary {task_id: [successor_ids]}
            predecessors: List of predecessor lists for each task
            p: Number of processors
            
        Returns:
            Makespan (lower is better)
        """
        # Decode chromosome to task-processor mapping
        schedule = decoder.decode(chromosome)
        
        # Calculate schedule times to get makespan
        _, _, task_finish_times = self._calculate_schedule_times(
            schedule, comp_matrix, comm_matrix, dag, predecessors, p
        )
        
        # Makespan is the maximum finish time
        makespan = max(task_finish_times.values()) if task_finish_times else 0.0
        
        return makespan
    
    def _calculate_schedule_times(self, schedule_dict: Dict[int, int], 
                                   comp_matrix: np.ndarray, comm_matrix: np.ndarray,
                                   dag: dict, predecessors: List,
                                   p: int) -> Tuple[Dict, Dict, Dict]:
        """
        Calculate actual start and finish times for tasks.
        
        Similar to HEFT's calculation but using fixed task-processor assignments.
        
        Args:
            schedule_dict: Mapping of task_id to processor_id
            comp_matrix: Computation cost matrix (n x p)
            comm_matrix: Communication cost matrix (n x n)
            dag: DAG as dictionary {task_id: [successor_ids]}
            predecessors: List of predecessor lists for each task
            p: Number of processors
            
        Returns:
            Tuple of (task_assignments, task_start_times, task_finish_times)
        """
        n = len(schedule_dict)
        task_start_times = {}
        task_finish_times = {}
        processor_available_time = {i: 0.0 for i in range(p)}
        
        # Process tasks in topological order (from schedule_dict keys)
        # We need to respect precedence constraints
        scheduled_tasks = set()
        
        while len(scheduled_tasks) < n:
            # Find tasks that can be scheduled (all predecessors done)
            ready_tasks = []
            for task_id in range(n):
                if task_id not in scheduled_tasks:
                    preds = predecessors[task_id] if task_id < len(predecessors) else []
                    if all(p in scheduled_tasks for p in preds):
                        ready_tasks.append(task_id)
            
            if not ready_tasks:
                # Circular dependency or error - break
                break
            
            # Schedule ready tasks
            for task_id in ready_tasks:
                processor_id = schedule_dict[task_id]
                preds = predecessors[task_id] if task_id < len(predecessors) else []
                
                # Calculate earliest start time considering:
                # 1. Processor availability
                # 2. Data arrival from predecessors
                earliest_start = processor_available_time[processor_id]
                
                for pred_id in preds:
                    pred_processor = schedule_dict[pred_id]
                    pred_finish = task_finish_times[pred_id]
                    
                    # Add communication cost if on different processors
                    if pred_processor != processor_id:
                        comm_cost = comm_matrix[pred_id, task_id] if pred_id < comm_matrix.shape[0] and task_id < comm_matrix.shape[1] else 0
                        data_ready_time = pred_finish + comm_cost
                    else:
                        data_ready_time = pred_finish
                    
                    earliest_start = max(earliest_start, data_ready_time)
                
                # Calculate finish time
                comp_cost = comp_matrix[task_id, processor_id]
                start_time = earliest_start
                finish_time = start_time + comp_cost
                
                # Update tracking
                task_start_times[task_id] = start_time
                task_finish_times[task_id] = finish_time
                processor_available_time[processor_id] = finish_time
                scheduled_tasks.add(task_id)
        
        return schedule_dict, task_start_times, task_finish_times
    
    def __repr__(self) -> str:
        """String representation of R²GA scheduler."""
        return (f"R2GA(pop_size={self.pop_size}, elite_size={self.elite_size}, "
                f"iterations={self.iterations}, mutation_rate={self.mutation_rate}, "
                f"crossover_rate={self.crossover_rate}, seed_with_heft={self.seed_with_heft})")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"R²GA Genetic Algorithm Scheduler"


if __name__ == "__main__":
    # Quick test
    print("R²GA Scheduler - Testing Instantiation")
    
    # Create scheduler with default parameters
    r2ga = R2GA(verbose=True)
    print(f"Created: {r2ga}")
    print(f"Repr: {repr(r2ga)}")
    
    # Create scheduler with custom parameters
    r2ga_custom = R2GA(
        pop_size=100,
        elite_size=10,
        iterations=200,
        mutation_rate=0.15,
        crossover_rate=0.85,
        seed_with_heft=True,
        verbose=True
    )
    print(f"\nCustom: {r2ga_custom}")
    
    print("\n✅ R²GA class skeleton created successfully!")
