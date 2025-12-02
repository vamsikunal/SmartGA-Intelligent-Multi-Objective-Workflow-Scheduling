import yaml
import json
import xml.etree.ElementTree as ET
import networkx as nx
from .task import Task

class WorkflowParser:
    def __init__(self):
        pass

    def parse_file(self, file_path):
        if file_path.endswith('.json'):
            return self.parse_json(file_path)
        elif file_path.endswith('.xml') or file_path.endswith('.dax'):
            return self.parse_dax(file_path)
        elif file_path.endswith('.yml') or file_path.endswith('.yaml'):
            return self.parse_yaml(file_path)
        else:
            raise ValueError("Unsupported file format. Use .json, .xml, .dax, or .yml")

    def parse_yaml(self, file_path):
        """
        Parses a Pegasus 5.0 YAML workflow file.
        """
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
            
        tasks = {}
        G = nx.DiGraph()
        
        # 1. Parse Jobs
        for job in data.get('jobs', []):
            tid = job['id']
            # Estimate instructions based on job name or type if available
            # For now, random or default
            instr = 1000 
            tasks[tid] = Task(tid, instr)
            G.add_node(tid)
            
            # Parse inputs/outputs for data transfer modeling
            for use in job.get('uses', []):
                size = 10 # Default 10MB
                if use['type'] == 'input':
                    tasks[tid].data_size_in += size
                elif use['type'] == 'output':
                    tasks[tid].data_size_out += size

        # 2. Parse Dependencies
        for dep in data.get('jobDependencies', []):
            parent_id = dep['id']
            for child_id in dep.get('children', []):
                if parent_id in tasks and child_id in tasks:
                    G.add_edge(parent_id, child_id)
                    tasks[parent_id].children.append(tasks[child_id])
                    tasks[child_id].parents.append(tasks[parent_id])
                    
        return tasks, G

    def parse_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        tasks = {}
        G = nx.DiGraph()

        # 1. Create Tasks
        for task_data in data['tasks']:
            tid = task_data['id']
            instr = task_data.get('instruction_count', 1000)
            tasks[tid] = Task(tid, instr)
            G.add_node(tid)

        # 2. Create Edges (Dependencies)
        for edge in data['dependencies']:
            parent_id = edge['parent']
            child_id = edge['child']
            
            if parent_id in tasks and child_id in tasks:
                G.add_edge(parent_id, child_id)
                tasks[parent_id].children.append(tasks[child_id])
                tasks[child_id].parents.append(tasks[parent_id])
            else:
                print(f"Warning: Dependency {parent_id} -> {child_id} refers to missing tasks.")

        return tasks, G

    def parse_dax(self, file_path):
        """
        Parses a Pegasus DAX (XML) file.
        """
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Namespace handling might be needed for real DAX files
        # For now, assuming simple XML structure or stripping namespaces
        
        tasks = {}
        G = nx.DiGraph()
        
        # 1. Parse Jobs (Tasks)
        # DAX uses <job id="..." runtime="..."> or similar
        for job in root.findall(".//job"):
            tid = job.get('id')
            runtime = float(job.get('runtime', 1.0))
            # Convert runtime to instructions (assuming 1000 MIPS for reference)
            instr = int(runtime * 1000) 
            
            tasks[tid] = Task(tid, instr)
            G.add_node(tid)
            
        # 2. Parse Dependencies
        # DAX uses <child ref="..."><parent ref="..."/></child>
        for child in root.findall(".//child"):
            child_id = child.get('ref')
            for parent in child.findall("parent"):
                parent_id = parent.get('ref')
                
                if parent_id in tasks and child_id in tasks:
                    G.add_edge(parent_id, child_id)
                    tasks[parent_id].children.append(tasks[child_id])
                    tasks[child_id].parents.append(tasks[parent_id])

        return tasks, G
