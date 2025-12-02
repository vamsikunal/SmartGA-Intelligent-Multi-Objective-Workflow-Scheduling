class ResourceNode:
    def __init__(self, node_id, mips, ram, bandwidth, cost_per_sec, power_active, power_idle, failure_rate):
        self.node_id = node_id
        self.mips = mips
        self.ram = ram
        self.bandwidth = bandwidth
        self.cost_per_sec = cost_per_sec
        self.power_active = power_active
        self.power_idle = power_idle
        self.failure_rate = failure_rate # Lambda (failures per second)
        
        self.current_load = 0
        self.available_time = 0.0

    def reset(self):
        """Reset node state for a fresh simulation."""
        self.current_load = 0
        self.available_time = 0.0

    def __repr__(self):
        return f"Node({self.node_id}, MIPS={self.mips}, Lambda={self.failure_rate})"

class CloudNode(ResourceNode):
    def __init__(self, node_id, config=None):
        # Defaults if config is missing
        mips = config.get('mips', 10000) if config else 10000
        ram = config.get('ram', 16000) if config else 16000
        cost = config.get('cost', 0.5) if config else 0.5
        p_active = config.get('power_active', 100) if config else 100
        p_idle = config.get('power_idle', 10) if config else 10
        fail_rate = config.get('failure_rate', 0.0001) if config else 0.0001
        
        super().__init__(node_id, mips, ram, 1000, cost, p_active, p_idle, fail_rate)

class EdgeNode(ResourceNode):
    def __init__(self, node_id, config=None):
        # Defaults if config is missing
        mips = config.get('mips', 2000) if config else 2000
        ram = config.get('ram', 4000) if config else 4000
        cost = config.get('cost', 0.1) if config else 0.1
        p_active = config.get('power_active', 20) if config else 20
        p_idle = config.get('power_idle', 2) if config else 2
        fail_rate = config.get('failure_rate', 0.001) if config else 0.001
        
        super().__init__(node_id, mips, ram, 100, cost, p_active, p_idle, fail_rate)
