class Task:
    def __init__(self, task_id, instruction_count, data_size_in=0, data_size_out=0):
        self.task_id = task_id
        self.instruction_count = instruction_count
        self.data_size_in = data_size_in   # MB
        self.data_size_out = data_size_out # MB
        self.parents = []
        self.children = []
        self.assigned_node = None
        self.start_time = 0
        self.finish_time = 0

    def __repr__(self):
        return f"Task({self.task_id}, Instr={self.instruction_count})"
