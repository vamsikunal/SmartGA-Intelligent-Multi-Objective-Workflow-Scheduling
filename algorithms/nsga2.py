import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import networkx as nx

class WorkflowSchedulingProblem(Problem):
    """
    Pymoo-compatible workflow scheduling problem.
    (Copied from SmartGA to keep NSGA-II self-contained)
    """
    def __init__(self, tasks, graph, nodes):
        self.tasks = tasks
        self.graph = graph
        self.nodes = nodes
        self.task_list = list(nx.topological_sort(graph))
        
        n_tasks = len(self.task_list)
        n_nodes = len(nodes)
        
        super().__init__(
            n_var=n_tasks,
            n_obj=3, # Makespan, Cost, Load Balancing
            n_constr=0,
            xl=0,
            xu=n_nodes - 1,
            type_var=int
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        from simulator.environment import CloudEdgeEnvironment
        
        n_pop = X.shape[0]
        objectives = np.zeros((n_pop, 3))
        
        for i in range(n_pop):
            individual = X[i]
            schedule = self._decode(individual)
            
            cloud_nodes = [n for n in self.nodes if "Cloud" in n.__class__.__name__]
            edge_nodes = [n for n in self.nodes if "Edge" in n.__class__.__name__]
            
            env = CloudEdgeEnvironment(cloud_nodes, edge_nodes)
            
            try:
                metrics = env.run(schedule)
                objectives[i, 0] = metrics['makespan']
                objectives[i, 1] = metrics['cost']
                objectives[i, 2] = metrics['load_balancing']
            except:
                objectives[i, :] = [1e6, 1e6, 1e6]
        
        out["F"] = objectives
    
    def _decode(self, individual):
        schedule = []
        for task_id in self.task_list:
            task_idx = self.task_list.index(task_id)
            node_idx = int(individual[task_idx])
            node_id = self.nodes[node_idx].node_id
            task = self.tasks[task_id]
            schedule.append((task, node_id))
        return schedule

class IndividualWrapper:
    """Wrapper to mimic DEAP individual for compatibility with main.py"""
    def __init__(self, objectives):
        # Create a dummy fitness object with values attribute
        self.fitness = type('Fitness', (), {'values': tuple(objectives)})()

class NSGA2Scheduler:
    def __init__(self, workflow_tasks, workflow_graph, nodes, population_size=50, generations=10):
        self.tasks = workflow_tasks
        self.graph = workflow_graph
        self.nodes = nodes
        self.pop_size = population_size
        self.generations = generations
        
        self.problem = WorkflowSchedulingProblem(workflow_tasks, workflow_graph, nodes)

    def run(self):
        algorithm = NSGA2(pop_size=self.pop_size)
        
        res = minimize(
            self.problem,
            algorithm,
            ('n_gen', self.generations),
            verbose=False
        )
        
        # Convert pymoo result to list of IndividualWrapper for compatibility
        pareto_front = []
        if res.F is not None:
            for objectives in res.F:
                pareto_front.append(IndividualWrapper(objectives))
                
        return pareto_front
