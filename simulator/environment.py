import simpy
import math
from .resource import ResourceNode

class CloudEdgeEnvironment:
    def __init__(self, cloud_nodes, edge_nodes, bandwidth_cloud_edge=1000):
        self.env = simpy.Environment()
        self.cloud_nodes = cloud_nodes
        self.edge_nodes = edge_nodes
        self.nodes = cloud_nodes + edge_nodes
        self.node_map = {node.node_id: node for node in self.nodes}
        self.bandwidth_cloud_edge = bandwidth_cloud_edge
        
        # Metrics
        self.total_cost = 0.0
        self.total_energy = 0.0
        self.makespan = 0.0
        self.reliability = 1.0 # Start with 100% probability of success

    def calculate_reliability(self, task, node, exec_time):
        # Reliability = e^(-lambda * t)
        # We multiply probabilities: R_total = R1 * R2 * ...
        task_reliability = math.exp(-node.failure_rate * exec_time)
        return task_reliability

    def run_task(self, task, node_id):
        node = self.node_map[node_id]
        
        # Calculation time
        exec_time = task.instruction_count / node.mips
        
        # Data Transfer Time (with proper bandwidth modeling)
        # Transfer data from parents if they were assigned to different nodes
        transfer_time = 0
        for parent in task.parents:
            if hasattr(parent, 'assigned_node') and parent.assigned_node and parent.assigned_node != node_id:
                parent_node = self.node_map[parent.assigned_node]
                
                # Data size from parent (use data_size_out if available, else data_size_in)
                data_size = getattr(parent, 'data_size_out', getattr(task, 'data_size_in', 0))
                
                if data_size > 0:
                    # Check if cross-layer transfer (Cloud <-> Edge)
                    if type(parent_node) != type(node):
                        # Cross-layer: use cloud-edge bandwidth
                        bandwidth_mbps = self.bandwidth_cloud_edge
                    else:
                        # Same layer: use higher bandwidth (simplified)
                        bandwidth_mbps = self.bandwidth_cloud_edge * 10
                    
                    # Convert MB to Mb and calculate transfer time
                    data_in_mb = data_size * 8  # MB to Megabits
                    transfer_time += data_in_mb / bandwidth_mbps  # seconds
        
        total_time = exec_time + transfer_time

        # Wait until node is available
        start_time = max(self.env.now, node.available_time)
        
        # Update node availability
        node.available_time = start_time + total_time
        
        # Wait for execution
        yield self.env.timeout(start_time - self.env.now + total_time)
        
        # Update metrics
        self.total_cost += total_time * node.cost_per_sec
        
        # Energy: Active power during exec, Idle otherwise (simplified)
        self.total_energy += total_time * node.power_active
        
        # Reliability update
        self.reliability *= self.calculate_reliability(task, node, exec_time)
        
        self.makespan = max(self.makespan, self.env.now)
        
        # Mark task as done on this node
        task.assigned_node = node_id
        task.finish_time = self.env.now
        
        return self.env.now

    def reset(self):
        """Reset environment state for a fresh simulation run."""
        self.env = simpy.Environment()
        self.total_cost = 0.0
        self.total_energy = 0.0
        self.makespan = 0.0
        self.reliability = 1.0
        
        # Reset all nodes
        for node in self.nodes:
            node.reset()
    
    def validate_schedule(self, schedule):
        """Validate schedule input before running simulation."""
        if not schedule:
            raise ValueError("Schedule cannot be empty")
        
        for idx, item in enumerate(schedule):
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(f"Schedule item {idx} must be a (task, node_id) tuple")
            
            task, node_id = item
            
            if node_id not in self.node_map:
                raise ValueError(f"Invalid node_id '{node_id}' in schedule item {idx}")
            
            if not hasattr(task, 'task_id'):
                raise ValueError(f"Task in schedule item {idx} missing 'task_id' attribute")
            
            if not hasattr(task, 'instruction_count'):
                raise ValueError(f"Task '{task.task_id}' missing 'instruction_count' attribute")

    def run_task_with_deps(self, task, node_id):
        """Run a task after waiting for all parent tasks to complete."""
        # Wait for all parent tasks to finish
        for parent in task.parents:
            if hasattr(parent, 'finish_time') and parent.finish_time > 0:
                wait_time = parent.finish_time - self.env.now
                if wait_time > 0:
                    yield self.env.timeout(wait_time)
        
        # Now run the task
        yield self.env.process(self.run_task(task, node_id))

    def run(self, schedule):
        """
        Run the simulation with the given schedule.
        Properly handles task dependencies to ensure correctness.
        
        Args:
            schedule: List of (task, node_id) tuples
            
        Returns:
            Dictionary of metrics: makespan, cost, energy, reliability, load_balancing
        """
        # Reset state for fresh run
        self.reset()
        
        # Validate input
        try:
            self.validate_schedule(schedule)
        except ValueError as e:
            # Return worst-case metrics on validation failure
            return {
                "makespan": float('inf'),
                "cost": float('inf'),
                "energy": float('inf'),
                "reliability": 0.0,
                "load_balancing": float('inf')
            }
        
        # Launch all task processes (they will wait for dependencies internally)
        for task, node_id in schedule:
            self.env.process(self.run_task_with_deps(task, node_id))
        
        # Run simulation
        try:
            self.env.run()
        except Exception as e:
            # Handle simulation errors gracefully
            print(f"Simulation error: {e}")
            return {
                "makespan": float('inf'),
                "cost": float('inf'),
                "energy": float('inf'),
                "reliability": 0.0,
                "load_balancing": float('inf')
            }
        
        # Calculate Load Balancing (Standard Deviation of Utilization)
        utilizations = []
        for node in self.nodes:
            # Utilization = Busy Time / Total Makespan
            busy_time = node.available_time  # Assuming start at 0
            if self.makespan > 0:
                util = busy_time / self.makespan
            else:
                util = 0
            utilizations.append(util)
        
        # Standard Deviation (handle edge cases)
        if len(utilizations) == 0:
            std_dev = 0.0
        elif len(utilizations) == 1:
            std_dev = 0.0
        else:
            mean_util = sum(utilizations) / len(utilizations)
            variance = sum((u - mean_util) ** 2 for u in utilizations) / len(utilizations)
            std_dev = math.sqrt(variance)

        return {
            "makespan": self.makespan,
            "cost": self.total_cost,
            "energy": self.total_energy,
            "reliability": self.reliability,
            "load_balancing": std_dev
        }

