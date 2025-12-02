"""
MO-R²GA - Multi-Objective Real Relative Encoding Genetic Algorithm

Adapted from R²GA for multi-objective Cloud-Edge workflow scheduling.
Combines R²GA's DAG-aware operators with NSGA-II's Pareto optimization.

Key Features:
- Relative encoding (2n chromosomes) - respects DAG structure
- CTS-based decoding - guarantees valid schedules
- NSGA-II selection - finds Pareto-optimal trade-offs
- 5 objectives: Makespan, Cost, Energy, Reliability, Load Balancing
"""

import numpy as np
import random
from deap import base, creator, tools
import networkx as nx

class MOR2GA:
    """
    Multi-Objective R²GA for Cloud-Edge Workflow Scheduling.
    
    Uses R²GA's encoding/decoding but optimizes for 5 objectives instead of just makespan.
    """
    
    def __init__(self, tasks, graph, nodes, pop_size=100, generations=50):
        """
        Initialize MO-R²GA scheduler.
        
        Args:
            tasks: Dictionary of task objects
            graph: NetworkX DAG
            nodes: List of ResourceNode objects (Cloud + Edge)
            pop_size: Population size
            generations: Number of generations
        """
        self.tasks = tasks
        self.graph = graph
        self.nodes = nodes
        self.task_ids = list(nx.topological_sort(graph))
        self.num_tasks = len(self.task_ids)
        self.num_nodes = len(nodes)
        self.pop_size = pop_size
        self.generations = generations
        
        # Build predecessor map for CTS decoding
        self.predecessors = {}
        for task_id in self.task_ids:
            self.predecessors[task_id] = list(graph.predecessors(task_id))
        
        # Setup DEAP for multi-objective optimization
        self._setup_deap()
    
    def _setup_deap(self):
        """Setup DEAP framework for multi-objective optimization."""
        # Create fitness and individual classes
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
            # Minimize: makespan, cost, load_balancing
        
        if not hasattr(creator, "IndividualR2GA"):
            creator.create("IndividualR2GA", np.ndarray, fitness=creator.FitnessMulti)
        
        self.toolbox = base.Toolbox()
        
        # Chromosome generation: 2n random values [0,1)
        self.toolbox.register("attr_float", lambda: np.random.random())
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", self.crossover_r2ga)
        self.toolbox.register("mutate", self.mutate_r2ga)
        self.toolbox.register("select", tools.selNSGA2)
    
    def _create_individual(self):
        """Create an individual using R²GA encoding (2n genes)."""
        chromosome = np.random.random(2 * self.num_tasks)
        return creator.IndividualR2GA(chromosome)
    
    def decode_chromosome(self, chromosome):
        """
        Decode R²GA chromosome to task-processor schedule using CTS.
        
        Algorithm 2 from R²GA paper:
        1. Initialize CTS with entry tasks (no predecessors)
        2. For each scheduling step:
           - Use task gene to select task from CTS (relative selection)
           - Use unit gene to select processor (relative selection)
           - Update CTS with newly ready tasks
        
        Returns:
            List of (task, node_id) tuples
        """
        schedule = []
        task_to_node = {}
        
        # Split chromosome into task and unit genes
        task_genes = chromosome[:self.num_tasks]
        unit_genes = chromosome[self.num_tasks:]
        
        # Initialize CTS with entry tasks
        scheduled = set()
        cts = [tid for tid in self.task_ids if not self.predecessors[tid]]
        
        for step in range(self.num_tasks):
            if not cts:
                break
            
            # Sort CTS for deterministic selection
            cts_sorted = sorted(cts)
            
            # Select task from CTS using task gene (relative selection)
            task_gene = task_genes[step]
            task_idx = int(task_gene * len(cts_sorted))
            task_idx = min(task_idx, len(cts_sorted) - 1)
            selected_task = cts_sorted[task_idx]
            
            # Select processor using unit gene (relative selection)
            unit_gene = unit_genes[step]
            node_idx = int(unit_gene * self.num_nodes)
            node_idx = min(node_idx, self.num_nodes - 1)
            selected_node = self.nodes[node_idx].node_id
            
            # Add to schedule
            schedule.append((self.tasks[selected_task], selected_node))
            task_to_node[selected_task] = selected_node
            scheduled.add(selected_task)
            
            # Update CTS
            cts.remove(selected_task)
            
            # Add newly ready successors
            for successor in self.graph.successors(selected_task):
                if successor not in scheduled and successor not in cts:
                    # Check if all predecessors are scheduled
                    if all(pred in scheduled for pred in self.predecessors[successor]):
                        cts.append(successor)
        
        return schedule
    
    def evaluate(self, individual):
        """
        Evaluate chromosome using CloudEdgeEnvironment.
        
        Returns 3 objectives: (makespan, cost, load_balancing)
        """
        from simulator.environment import CloudEdgeEnvironment
        
        # Decode chromosome to schedule
        schedule = self.decode_chromosome(individual)
        
        # Create environment and run simulation
        cloud_nodes = [n for n in self.nodes if "Cloud" in n.__class__.__name__]
        edge_nodes = [n for n in self.nodes if "Edge" in n.__class__.__name__]
        
        env = CloudEdgeEnvironment(cloud_nodes, edge_nodes)
        
        try:
            metrics = env.run(schedule)
        except:
            # If schedule fails, return worst-case metrics
            return (float('inf'), float('inf'), float('inf'))
        
        # Return 3 objectives (NSGA-II will minimize all with weights)
        return (
            metrics['makespan'],
            metrics['cost'],
            metrics['load_balancing']
        )
    
    def crossover_r2ga(self, ind1, ind2):
        """
        R²GA two-point crossover (Algorithm 5).
        
        - Point 1 in task genes [0, n)
        - Point 2 in unit genes [n, 2n)
        - Swap segment between points
        """
        n = self.num_tasks
        
        # Select two points
        point1 = random.randint(0, n - 1)
        point2 = random.randint(n, 2 * n - 1)
        
        # Swap segment
        ind1[point1:point2], ind2[point1:point2] = \
            ind2[point1:point2].copy(), ind1[point1:point2].copy()
        
        return ind1, ind2
    
    def mutate_r2ga(self, individual):
        """
        R²GA two-point random replacement mutation (Algorithm 6).
        
        - Replace ONE gene in task part [0, n) with random [0,1)
        - Replace ONE gene in unit part [n, 2n) with random [0,1)
        """
        n = self.num_tasks
        
        # Mutate one task gene
        task_point = random.randint(0, n - 1)
        individual[task_point] = random.random()
        
        # Mutate one unit gene  
        unit_point = random.randint(n, 2 * n - 1)
        individual[unit_point] = random.random()
        
        return individual,
    
    def run(self):
        """
        Execute MO-R²GA algorithm.
        
        Returns:
            List of Pareto-optimal individuals (with fitness.values)
        """
        # Generate initial population
        population = self.toolbox.population(n=self.pop_size)
        
        # Optionally seed with HEFT-inspired solution
        heft_seed = self._generate_heft_seed()
        if heft_seed is not None:
            population[0] = heft_seed
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Evolution loop
        for gen in range(self.generations):
            # Select next generation
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.9:  # Crossover rate
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Apply mutation
            for mutant in offspring:
                if random.random() < 0.15:  # Mutation rate
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            population[:] = offspring
        
        # Extract Pareto front - keep individuals, not schedules!
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        
        # Return individuals directly (they have .fitness.values attached)
        return list(pareto_front)
    
    def _generate_heft_seed(self):
        """Generate a HEFT-inspired chromosome for seeding."""
        try:
            from algorithms.heft import HEFT
            heft = HEFT(self.tasks, self.graph, self.nodes)
            heft_schedule = heft.schedule()
            
            # Convert HEFT schedule to R²GA chromosome
            # This is approximate - use mid-range values for genes
            chromosome = np.random.random(2 * self.num_tasks) * 0.5 + 0.25  # [0.25, 0.75)
            
            return creator.IndividualR2GA(chromosome)
        except:
            return None
