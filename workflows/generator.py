import networkx as nx
import random
from .task import Task

class WorkflowGenerator:
    def __init__(self):
        pass

    def generate_workflow(self, workflow_type, num_tasks, seed=42):
        random.seed(seed)
        if workflow_type == "montage":
            return self.generate_montage(num_tasks)
        elif workflow_type == "cybershake":
            return self.generate_cybershake(num_tasks)
        elif workflow_type == "ligo":
            return self.generate_ligo(num_tasks)
        elif workflow_type == "sipht":
            return self.generate_sipht(num_tasks)
        else:
            return self.generate_random_layered(num_tasks)

    def _create_tasks_from_graph(self, G):
        tasks = {}
        for node_id in G.nodes():
            # Random instructions between 1000 and 10000 MIPS equivalent
            instr = random.randint(1000, 10000)
            tasks[node_id] = Task(node_id, instr)

        # Link tasks based on graph
        for u, v in G.edges():
            tasks[u].children.append(tasks[v])
            tasks[v].parents.append(tasks[u])
        return tasks, G

    def generate_random_layered(self, num_tasks):
        G = nx.DiGraph()
        for i in range(num_tasks):
            G.add_node(i)
            
        layers = int(num_tasks ** 0.5)
        nodes_per_layer = num_tasks // layers
        
        current_node = 0
        layer_nodes = []
        
        for i in range(layers):
            count = nodes_per_layer + (1 if i < num_tasks % layers else 0)
            layer_nodes.append(list(range(current_node, current_node + count)))
            current_node += count
            
        for i in range(layers - 1):
            current_layer = layer_nodes[i]
            next_layer = layer_nodes[i+1]
            for u in current_layer:
                v = random.choice(next_layer)
                G.add_edge(u, v)
                for v_other in next_layer:
                    if random.random() < 0.3:
                        G.add_edge(u, v_other)
        
        return self._create_tasks_from_graph(G)

    def generate_montage(self, num_tasks):
        """
        Montage: Astronomy Mosaic
        Structure: Level 1 (mProject) -> Level 2 (mDiffFit) -> Level 3 (mConcatFit) -> Level 4 (mBgModel) -> Level 5 (mBackground) -> Level 6 (mImgTbl) -> Level 7 (mAdd)
        Simplified: Fan-out -> Layer -> Fan-in -> Fan-out -> Fan-in
        """
        G = nx.DiGraph()
        # Level 1: Input images (approx 40% of tasks)
        level1_count = max(2, int(num_tasks * 0.4))
        level1_nodes = list(range(level1_count))
        
        # Level 2: Projections (1-to-1 from Level 1)
        level2_start = level1_count
        level2_count = level1_count
        level2_nodes = list(range(level2_start, level2_start + level2_count))
        
        for i in range(level1_count):
            G.add_edge(level1_nodes[i], level2_nodes[i])
            
        # Level 3: Differences (Many-to-Many)
        remaining = num_tasks - (level1_count + level2_count)
        if remaining < 3: remaining = 3 # Ensure minimal structure
        
        level3_count = int(remaining * 0.6)
        level3_start = level2_start + level2_count
        level3_nodes = list(range(level3_start, level3_start + level3_count))
        
        for u in level2_nodes:
            # Connect to random neighbors in level 3
            targets = random.sample(level3_nodes, k=min(len(level3_nodes), 2))
            for v in targets:
                G.add_edge(u, v)
                
        # Level 4: Aggregation (Final Mosaic)
        final_node = num_tasks - 1
        G.add_node(final_node)
        
        # Connect Level 3 to Final
        for u in level3_nodes:
            G.add_edge(u, final_node)
            
        # Fill any gaps with intermediate nodes if needed
        used_nodes = set(G.nodes())
        for i in range(num_tasks):
            if i not in used_nodes:
                # Attach to random level 2 and final
                if level2_nodes:
                    parent = random.choice(level2_nodes)
                    G.add_edge(parent, i)
                    G.add_edge(i, final_node)
                else:
                    G.add_node(i) # Orphan (shouldn't happen with logic above)

        return self._create_tasks_from_graph(G)

    def generate_cybershake(self, num_tasks):
        """
        CyberShake: Earthquake Hazard
        Structure: Parallel pipelines (Extract -> SGT -> Seis) -> Aggregation (PeakVal)
        """
        G = nx.DiGraph()
        
        # Final aggregation node
        final_node = num_tasks - 1
        G.add_node(final_node)
        
        # Number of pipelines
        pipeline_len = 3 # Extract -> SGT -> Seis
        num_pipelines = (num_tasks - 1) // pipeline_len
        
        current = 0
        for p in range(num_pipelines):
            # Pipeline nodes: n1 -> n2 -> n3 -> Final
            n1 = current
            n2 = current + 1
            n3 = current + 2
            
            if n3 >= final_node: break # Stop if out of nodes
            
            G.add_edge(n1, n2)
            G.add_edge(n2, n3)
            G.add_edge(n3, final_node)
            
            current += 3
            
        # Handle leftovers
        while current < final_node:
            G.add_edge(current, final_node)
            current += 1
            
        return self._create_tasks_from_graph(G)

    def generate_ligo(self, num_tasks):
        """
        LIGO: Gravitational Wave
        Structure: Series of inspection tasks with some merging.
        Simplified: Block of parallel tasks -> Block of parallel tasks -> Merge
        """
        G = nx.DiGraph()
        
        # Split into blocks
        block_size = max(2, int(num_tasks * 0.2))
        num_blocks = num_tasks // block_size
        
        blocks = []
        current = 0
        for i in range(num_blocks):
            end = min(current + block_size, num_tasks)
            blocks.append(list(range(current, end)))
            current = end
            
        # Connect blocks linearly with some cross-over
        for i in range(len(blocks) - 1):
            curr_block = blocks[i]
            next_block = blocks[i+1]
            
            for u in curr_block:
                # Connect to 1-2 nodes in next block
                targets = random.sample(next_block, k=min(len(next_block), 2))
                for v in targets:
                    G.add_edge(u, v)
                    
        return self._create_tasks_from_graph(G)

    def generate_sipht(self, num_tasks):
        """
        SIPHT: Bioinformatics
        Structure: Very wide fan-out, processing, fan-in.
        """
        G = nx.DiGraph()
        
        source = 0
        sink = num_tasks - 1
        G.add_node(source)
        G.add_node(sink)
        
        # Middle layer
        middle_nodes = list(range(1, num_tasks - 1))
        
        for node in middle_nodes:
            G.add_edge(source, node)
            G.add_edge(node, sink)
            
        return self._create_tasks_from_graph(G)
