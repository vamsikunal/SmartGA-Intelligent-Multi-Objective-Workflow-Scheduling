class HEFT:
    def __init__(self, workflow_tasks, workflow_graph, nodes):
        self.tasks = workflow_tasks
        self.graph = workflow_graph
        self.nodes = nodes
        self.rank = {}

    def calculate_rank(self):
        """
        Calculates the upward rank for all tasks.
        Rank_u(n) = w_n + max(c_nj + Rank_u(j))
        """
        # Topological sort reverse order
        import networkx as nx
        topo_order = list(reversed(list(nx.topological_sort(self.graph))))
        
        # Average computation cost
        avg_mips = sum(n.mips for n in self.nodes) / len(self.nodes)
        
        for task_id in topo_order:
            task = self.tasks[task_id]
            avg_comp_cost = task.instruction_count / avg_mips
            
            max_successor_cost = 0
            for child in task.children:
                # Simplified communication cost (assuming 0 for now or avg bandwidth)
                comm_cost = 0 
                max_successor_cost = max(max_successor_cost, comm_cost + self.rank[child.task_id])
            
            self.rank[task_id] = avg_comp_cost + max_successor_cost

    def schedule(self):
        self.calculate_rank()
        
        # Sort tasks by rank descending
        sorted_tasks = sorted(self.tasks.values(), key=lambda t: self.rank[t.task_id], reverse=True)
        
        schedule = []
        node_avail_time = {n.node_id: 0 for n in self.nodes}
        task_finish_times = {}
        
        for task in sorted_tasks:
            best_node = None
            min_finish_time = float('inf')
            
            for node in self.nodes:
                # Calculate Earliest Start Time (EST)
                # EST = max(node_avail_time, max(parent_finish_time + comm_cost))
                est = node_avail_time[node.node_id]
                
                for parent in task.parents:
                    if parent.task_id in task_finish_times:
                        parent_finish = task_finish_times[parent.task_id]
                        # Simplified comm cost: 0 if same node, else simplified value
                        # In full sim, we use bandwidth. Here we approximate or just use finish time.
                        # For HEFT correctness, we MUST wait for parents.
                        est = max(est, parent_finish)
                
                exec_time = task.instruction_count / node.mips
                finish_time = est + exec_time
                
                if finish_time < min_finish_time:
                    min_finish_time = finish_time
                    best_node = node
            
            schedule.append((task, best_node.node_id))
            node_avail_time[best_node.node_id] = min_finish_time
            task_finish_times[task.task_id] = min_finish_time
            
        return schedule
