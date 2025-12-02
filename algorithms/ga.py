import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import networkx as nx

class SingleObjectiveSchedulingProblem(Problem):
    """
    Single-objective workflow scheduling problem (Makespan only).
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
            n_obj=1, # Makespan only
            n_constr=0,
            xl=0,
            xu=n_nodes - 1,
            type_var=int
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        from simulator.environment import CloudEdgeEnvironment
        
        n_pop = X.shape[0]
        objectives = np.zeros((n_pop, 1))
        
        for i in range(n_pop):
            individual = X[i]
            schedule = self._decode(individual)
            
            cloud_nodes = [n for n in self.nodes if "Cloud" in n.__class__.__name__]
            edge_nodes = [n for n in self.nodes if "Edge" in n.__class__.__name__]
            
            env = CloudEdgeEnvironment(cloud_nodes, edge_nodes)
            
            try:
                metrics = env.run(schedule)
                objectives[i, 0] = metrics['makespan']
            except:
                objectives[i, 0] = 1e6
        
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

class GAScheduler:
    def __init__(self, workflow_tasks, workflow_graph, nodes, population_size=50, generations=50):
        self.tasks = workflow_tasks
        self.graph = workflow_graph
        self.nodes = nodes
        self.pop_size = population_size
        self.generations = generations
        
        self.problem = SingleObjectiveSchedulingProblem(workflow_tasks, workflow_graph, nodes)

    def run(self):
        algorithm = GA(pop_size=self.pop_size)
        
        res = minimize(
            self.problem,
            algorithm,
            ('n_gen', self.generations),
            verbose=False
        )
        
        # Return raw result object
        return res
