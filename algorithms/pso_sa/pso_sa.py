"""
PSO-SA (Particle Swarm Optimization + Simulated Annealing) Scheduler

Hybrid metaheuristic that combines PSO's swarm intelligence with
SA's probabilistic acceptance to escape local optima.

Converted from: relatedWork/PSO-SA-main/PSO_SA.py
"""

import sys
import os
import time
import numpy as np
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass

from .particle import PSO_SA_Particle

@dataclass
class ResourceConfig:
    num_processors: int

@dataclass
class ScheduleResult:
    task_assignments: Dict[int, int]
    task_start_times: Dict[int, float]
    task_finish_times: Dict[int, float]
    makespan: float
    algorithm_name: str = "PSO-SA"
    execution_time: float = 0.0
    convergence_history: List[float] = None
    algorithm_metadata: Dict[str, Any] = None

class PSOSAScheduler:
    """
    PSO-SA Hybrid Scheduler for Workflow Scheduling.
    
    Combines:
    - PSO: Swarm-based global search with velocity/position updates
    - SA: Probabilistic acceptance of worse solutions to escape local optima
    
    Key Innovation:
    Each particle uses SA acceptance criterion when updating its personal
    best, allowing exploration of worse solutions early in search and
    gradually becoming greedy as temperature cools.
    
    Parameters:
        pop_size: Number of particles in swarm (default: 20)
        iterations: Maximum iterations (default: 100)
        w: Inertia weight for velocity (default: 0.7)
        c1: Cognitive coefficient - personal best attraction (default: 1.5)
        c2: Social coefficient - global best attraction (default: 1.5)
        initial_temp: Initial SA temperature (default: 50)
        cooling: Temperature cooling rate (default: 0.85)
        sa_attempts: SA evaluation attempts per iteration (default: 20)
        verbose: Enable verbose logging (default: False)
    
    References:
        Original PSO-SA implementation for continuous optimization
        Adapted for discrete workflow scheduling problems
    """
    
    def __init__(self,
                 pop_size=20,
                 iterations=100,
                 w=0.7,
                 c1=1.5,
                 c2=1.5,
                 initial_temp=50,
                 cooling=0.85,
                 sa_attempts=20,
                 verbose=False):
        """Initialize PSO-SA scheduler with hybrid parameters."""
        # super().__init__()
        
        self.pop_size = pop_size
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.initial_temp = initial_temp
        self.cooling = cooling
        self.sa_attempts = sa_attempts
        self.verbose = verbose
        
        self.swarm = []
        self.gBest = None
        self.gBest_fitness = float('inf')
        self.convergence_history = []
        
        self._log(f"PSO-SA initialized: pop={pop_size}, iter={iterations}, w={w}, c1={c1}, c2={c2}")
    
    def _log(self, message: str):
        """Log message if verbose mode enabled."""
        if self.verbose:
            print(f"[PSO-SA] {message}")
    
    def schedule(self, workflow: Tuple, resources: ResourceConfig) -> ScheduleResult:
        """
        Schedule workflow using PSO-SA hybrid algorithm.
        
        Args:
            workflow: Tuple (n, W, C, dag, predecessors)
            resources: Resource configuration
        
        Returns:
            ScheduleResult with task assignments and metrics
        """
        start_time = time.time()
        
        n, W, C, dag, predecessors = workflow
        p = resources.num_processors
        
        self._log(f"Scheduling {n} tasks on {p} processors")
        
        # Store problem data for fitness function
        self.problem_data = {
            'n': n,
            'p': p,
            'W': W,
            'C': C,
            'dag': dag,
            'predecessors': predecessors,
            'resources': resources
        }
        
        # Initialize swarm
        self._initialize_swarm(n, p)
        self._log(f"Initialized swarm with {self.pop_size} particles")
        
        # Initialize temperature
        current_temp = self.initial_temp
        
        # Main PSO-SA loop
        for iteration in range(self.iterations):
            # SA attempts per iteration
            for attempt in range(self.sa_attempts):
                # Evaluate all particles with SA acceptance
                for particle in self.swarm:
                    accepted = particle.evaluate_with_SA(self._fitness_function, current_temp)
                    
                    # Update global best if needed
                    if particle.pBest_fitness < self.gBest_fitness:
                        self.gBest = particle.pBest.copy()
                        self.gBest_fitness = particle.pBest_fitness
                
                # Update velocities and positions (PSO)
                for particle in self.swarm:
                    particle.update_velocity(self.gBest, self.w, self.c1, self.c2)
                    particle.update_position()
            
            # Cool temperature (SA)
            current_temp = current_temp * self.cooling
            
            # Record convergence
            self.convergence_history.append(self.gBest_fitness)
            
            if self.verbose and (iteration + 1) % 10 == 0:
                self._log(f"Iter {iteration + 1}/{self.iterations}: Best={self.gBest_fitness:.2f}, Temp={current_temp:.2f}")
        
        self._log(f"Optimization complete. Best fitness: {self.gBest_fitness:.2f}")
        
        # Decode best solution to schedule
        schedule = self._decode_solution(self.gBest, self.problem_data)
        
        # Create result
        execution_time = time.time() - start_time
        result = ScheduleResult(
            task_assignments=schedule['task_assignments'],
            task_start_times=schedule['task_start_times'],
            task_finish_times=schedule['task_finish_times'],
            makespan=schedule['makespan']
        )
        result.algorithm_name = "PSO-SA"
        result.execution_time = execution_time
        
        # Compute additional metrics
        # result = self._compute_metrics(result, workflow, resources)
        
        # Add algorithm metadata
        result.convergence_history = self.convergence_history
        result.algorithm_metadata = {
            'algorithm': 'PSO-SA',
            'type': 'Hybrid (PSO + SA)',
            'pop_size': self.pop_size,
            'iterations': self.iterations,
            'w': self.w,
            'c1': self.c1,
            'c2': self.c2,
            'initial_temp': self.initial_temp,
            'final_temp': current_temp,
            'cooling': self.cooling,
            'sa_attempts': self.sa_attempts,
            'final_fitness': self.gBest_fitness
        }
        
        self._log(f"Scheduling complete. Makespan: {result.makespan:.2f}, Time: {execution_time:.3f}s")
        
        return result
    
    def _initialize_swarm(self, n, p):
        """
        Initialize particle swarm with random positions.
        
        Args:
            n: Number of tasks
            p: Number of processors
        """
        self.swarm = []
        self.gBest = None
        self.gBest_fitness = float('inf')
        
        for i in range(self.pop_size):
            particle = PSO_SA_Particle(n, p, self.initial_temp)
            self.swarm.append(particle)
    
    def _fitness_function(self, position: np.ndarray) -> float:
        """
        Fitness function for PSO-SA optimization.
        
        Converts continuous particle position to discrete processor
        assignments and calculates makespan.
        
        Args:
            position: Particle position (continuous [0, p-1])
        
        Returns:
            Makespan (fitness to minimize)
        """
        # Round to integers for discrete processor assignments
        proc_assignments = np.round(position).astype(int)
        proc_assignments = np.clip(proc_assignments, 0, self.problem_data['p'] - 1)
        
        # Decode to schedule
        schedule = self._decode_solution(proc_assignments, self.problem_data)
        
        return schedule['makespan']
    
    def _decode_solution(self, solution: np.ndarray, problem_data: Dict) -> Dict:
        """
        Decode processor assignments to task schedule.
        
        Args:
            solution: Array of processor assignments [0, p-1]
            problem_data: Workflow and resource information
        
        Returns:
            Dictionary with schedule and makespan
        """
        n = problem_data['n']
        p = problem_data['p']
        W = problem_data['W']
        C = problem_data['C']
        dag = problem_data['dag']
        predecessors = problem_data['predecessors']
        
        # Initialize data structures
        task_assignments = {}
        task_start_times = {}
        task_finish_times = {}
        processor_available_time = {proc_id: 0.0 for proc_id in range(p)}
        
        # Compute topological levels
        levels = self._compute_task_levels(n, predecessors)
        
        # Sort tasks by level (topological order)
        sorted_tasks = sorted(range(n), key=lambda t: levels[t])
        
        # Schedule each task
        for task_id in sorted_tasks:
            proc_id = int(solution[task_id])
            
            # Calculate EST
            est = self._calculate_earliest_start_time(
                task_id, proc_id, W, C, dag, predecessors,
                task_assignments, task_start_times, task_finish_times,
                processor_available_time
            )
            
            # Calculate EFT
            eft = est + W[task_id][proc_id]
            
            # Update schedule
            task_assignments[task_id] = proc_id
            task_start_times[task_id] = est
            task_finish_times[task_id] = eft
            processor_available_time[proc_id] = eft
        
        # Calculate makespan
        makespan = max(task_finish_times.values()) if task_finish_times else 0.0
        
        return {
            'task_assignments': task_assignments,
            'task_start_times': task_start_times,
            'task_finish_times': task_finish_times,
            'makespan': makespan
        }
    
    def _compute_task_levels(self, n: int, predecessors: Dict) -> List[int]:
        """Compute topological levels for tasks."""
        levels = [-1] * n
        
        changed = True
        while changed:
            changed = False
            for task_id in range(n):
                if levels[task_id] == -1:
                    preds = predecessors.get(task_id, [])
                    
                    if not preds:
                        levels[task_id] = 0
                        changed = True
                    else:
                        pred_levels = [levels[p] for p in preds]
                        if all(l >= 0 for l in pred_levels):
                            levels[task_id] = max(pred_levels) + 1
                            changed = True
        
        return levels
    
    def _calculate_earliest_start_time(self, task_id: int, proc_id: int,
                                       W: np.ndarray, C: np.ndarray,
                                       dag: Dict, predecessors: Dict,
                                       task_assignments: Dict,
                                       task_start_times: Dict,
                                       task_finish_times: Dict,
                                       processor_available_time: Dict) -> float:
        """Calculate earliest start time for a task on a processor."""
        est = processor_available_time[proc_id]
        
        preds = predecessors.get(task_id, [])
        for pred_id in preds:
            if pred_id in task_finish_times:
                pred_finish = task_finish_times[pred_id]
                pred_proc = task_assignments[pred_id]
                
                if pred_proc != proc_id:
                    comm_cost = C[pred_id][task_id]
                else:
                    comm_cost = 0.0
                
                est = max(est, pred_finish + comm_cost)
        
        return est

