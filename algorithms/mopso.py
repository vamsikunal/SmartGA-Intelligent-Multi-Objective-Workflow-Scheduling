import random
import numpy as np
from deap import base, creator, tools

class MOPSOScheduler:
    def __init__(self, workflow_tasks, workflow_graph, nodes, population_size=50, generations=20):
        self.tasks = workflow_tasks
        self.graph = workflow_graph
        self.nodes = nodes
        self.pop_size = population_size
        self.generations = generations
        self.task_ids = list(workflow_tasks.keys())
        self.num_tasks = len(self.task_ids)
        self.num_nodes = len(nodes)
        
        # MOPSO Parameters
        self.w = 0.5 # Inertia
        self.c1 = 1.0 # Personal best weight
        self.c2 = 1.0 # Global best weight
        
        # Initialize Particles
        self.particles = []
        for _ in range(self.pop_size):
            p = [random.randint(0, self.num_nodes - 1) for _ in range(self.num_tasks)]
            self.particles.append({
                'position': p,
                'velocity': [0.0] * self.num_tasks,
                'best_pos': list(p),
                'best_fitness': None,
                'fitness': None
            })
            
        self.archive = [] # Pareto Archive

    def evaluate(self, position):
        import networkx as nx
        from simulator.environment import CloudEdgeEnvironment
        
        # Identify cloud/edge nodes for env
        cloud_nodes = [n for n in self.nodes if "Cloud" in n.__class__.__name__]
        edge_nodes = [n for n in self.nodes if "Edge" in n.__class__.__name__]
        
        env = CloudEdgeEnvironment(cloud_nodes, edge_nodes)
        
        topo_order = list(nx.topological_sort(self.graph))
        
        schedule = []
        for task_id in topo_order:
            task_idx = self.task_ids.index(task_id)
            # Position is continuous, map to int index
            node_idx = int(position[task_idx])
            node_idx = max(0, min(node_idx, self.num_nodes - 1))
            
            node_id = self.nodes[node_idx].node_id
            task = self.tasks[task_id]
            schedule.append((task, node_id))
            
        metrics = env.run(schedule)
        
        return (metrics['makespan'], metrics['cost'], metrics['load_balancing'])

    def dominates(self, fit1, fit2):
        # Returns True if fit1 dominates fit2
        # Assuming minimization for all objectives
        better = False
        for f1, f2 in zip(fit1, fit2):
            if f1 > f2:
                return False
            if f1 < f2:
                better = True
        return better

    def update_archive(self, particle):
        # Add particle to archive if not dominated
        is_dominated = False
        to_remove = []
        
        for other in self.archive:
            if self.dominates(other['fitness'], particle['fitness']):
                is_dominated = True
                break
            if self.dominates(particle['fitness'], other['fitness']):
                to_remove.append(other)
        
        if not is_dominated:
            for r in to_remove:
                self.archive.remove(r)
            self.archive.append(particle)

    def run(self):
        # Initial Evaluation
        for p in self.particles:
            p['fitness'] = self.evaluate(p['position'])
            p['best_fitness'] = p['fitness']
            self.update_archive(p)
            
        for gen in range(self.generations):
            for i, p in enumerate(self.particles):
                # Select Global Best (gBest) from Archive (e.g., random or crowding distance)
                if not self.archive:
                    gbest = p['best_pos']
                else:
                    gbest = random.choice(self.archive)['position']
                
                # Update Velocity and Position
                new_pos = []
                for d in range(self.num_tasks):
                    r1, r2 = random.random(), random.random()
                    vel = (self.w * p['velocity'][d] + 
                           self.c1 * r1 * (p['best_pos'][d] - p['position'][d]) +
                           self.c2 * r2 * (gbest[d] - p['position'][d]))
                    
                    # Update position (discretize)
                    pos = int(p['position'][d] + vel)
                    pos = max(0, min(pos, self.num_nodes - 1))
                    
                    p['velocity'][d] = vel
                    new_pos.append(pos)
                
                p['position'] = new_pos
                p['fitness'] = self.evaluate(new_pos)
                
                # Update Personal Best
                if self.dominates(p['fitness'], p['best_fitness']):
                    p['best_pos'] = list(new_pos)
                    p['best_fitness'] = p['fitness']
                
                self.update_archive(p)
                
        return self.archive
