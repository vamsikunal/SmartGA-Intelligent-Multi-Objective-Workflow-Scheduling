import random

class PAESScheduler:
    def __init__(self, workflow_tasks, workflow_graph, nodes, max_evals=1000, archive_size=50):
        self.tasks = workflow_tasks
        self.graph = workflow_graph
        self.nodes = nodes
        self.max_evals = max_evals
        self.archive_size = archive_size
        self.task_ids = list(workflow_tasks.keys())
        self.num_tasks = len(self.task_ids)
        self.num_nodes = len(nodes)
        self.archive = []

    def evaluate(self, solution):
        import networkx as nx
        from simulator.environment import CloudEdgeEnvironment
        
        cloud_nodes = [n for n in self.nodes if "Cloud" in n.__class__.__name__]
        edge_nodes = [n for n in self.nodes if "Edge" in n.__class__.__name__]
        
        env = CloudEdgeEnvironment(cloud_nodes, edge_nodes)
        
        topo_order = list(nx.topological_sort(self.graph))
        
        schedule = []
        for task_id in topo_order:
            task_idx = self.task_ids.index(task_id)
            node_idx = solution[task_idx]
            node_id = self.nodes[node_idx].node_id
            task = self.tasks[task_id]
            schedule.append((task, node_id))
            
        metrics = env.run(schedule)
        
        return (metrics['makespan'], metrics['cost'], metrics['load_balancing'])

    def dominates(self, fit1, fit2):
        better = False
        for f1, f2 in zip(fit1, fit2):
            if f1 > f2:
                return False
            if f1 < f2:
                better = True
        return better

    def mutate(self, solution):
        new_sol = list(solution)
        idx = random.randint(0, self.num_tasks - 1)
        new_sol[idx] = random.randint(0, self.num_nodes - 1)
        return new_sol

    def add_to_archive(self, solution, fitness):
        # Add if not dominated
        is_dominated = False
        to_remove = []
        for item in self.archive:
            if self.dominates(item['fitness'], fitness):
                is_dominated = True
                break
            if self.dominates(fitness, item['fitness']):
                to_remove.append(item)
        
        if not is_dominated:
            for r in to_remove:
                self.archive.remove(r)
            self.archive.append({'position': solution, 'fitness': fitness})
            
            # Prune if full (simplified random removal)
            if len(self.archive) > self.archive_size:
                self.archive.pop(random.randint(0, len(self.archive)-1))
        
        return not is_dominated

    def run(self):
        # Initial solution
        current = [random.randint(0, self.num_nodes - 1) for _ in range(self.num_tasks)]
        current_fit = self.evaluate(current)
        self.add_to_archive(current, current_fit)
        
        for _ in range(self.max_evals):
            mutant = self.mutate(current)
            mutant_fit = self.evaluate(mutant)
            
            if self.dominates(mutant_fit, current_fit):
                current = mutant
                current_fit = mutant_fit
                self.add_to_archive(mutant, mutant_fit)
            elif self.dominates(current_fit, mutant_fit):
                pass # Discard mutant
            else:
                # Non-dominated, add to archive and check grid/crowding (simplified)
                if self.add_to_archive(mutant, mutant_fit):
                    current = mutant
                    current_fit = mutant_fit
                    
        return self.archive
