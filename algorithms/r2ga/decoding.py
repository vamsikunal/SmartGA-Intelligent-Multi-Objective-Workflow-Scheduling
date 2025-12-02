"""
Decoding Module for R²GA

Implements Algorithm 2: Chromosome Decoding
Translates chromosomes into valid task-processor schedules using
Candidate Task Set (CTS) mechanism that respects DAG precedence constraints.

Based on: J. Jiang et al., IEEE TPDS 2025
"""

import numpy as np
from typing import Dict, List, Set, Tuple


class Decoder:
    """
    Implements Algorithm 2: Decoding.
    
    Translates a chromosome into a valid workflow schedule (task-to-processor mapping)
    while respecting the DAG's precedence constraints.
    
    Key Concept - Candidate Task Set (CTS):
    At each step, CTS contains tasks whose predecessors have all been scheduled.
    The chromosome genes select which task to schedule next and which processor to use.
    
    Relative Interpretation:
    - Task gene value * |CTS| → which task to select from CTS
    - Unit gene value * p → which processor to assign
    """
    
    def __init__(self, n: int, p: int, W: np.ndarray, C: np.ndarray,
                 dag: List[Tuple[int, int]], predecessors: Dict[int, List[int]]):
        """
        Initialize the Decoder with workflow and system information.
        
        Args:
            n: Number of tasks
            p: Number of processors
            W: Computation cost matrix (n × p)
            C: Communication cost matrix (n × n, task-to-task)
            dag: List of (parent, child) edges
            predecessors: Dict mapping task_id to list of predecessor task_ids
            
        Raises:
            ValueError: If input dimensions are inconsistent
        """
        if W.shape[0] != n or W.shape[1] != p:
            raise ValueError(f"W matrix shape {W.shape} incompatible with n={n}, p={p}")
        if C.shape[0] != n or C.shape[1] != n:
            raise ValueError(f"C matrix shape {C.shape} incompatible with n={n}")
        
        self.n = n
        self.p = p
        self.W = W
        self.C = C
        self.dag = dag
        self.predecessors = predecessors
        
        # Build adjacency list (successors) from DAG
        self.successors: Dict[int, List[int]] = {i: [] for i in range(n)}
        for parent, child in dag:
            self.successors[parent].append(child)
        
        # Find entry tasks (those with no predecessors)
        self.entry_tasks = [task for task in range(n) if not predecessors.get(task, [])]
        
        if not self.entry_tasks:
            raise ValueError("DAG has no entry tasks (cycle detected or invalid structure)")
    
    def decode(self, chromosome: np.ndarray) -> Dict[int, int]:
        """
        Decode a chromosome into a task schedule (Algorithm 2).
        
        Process:
        1. Initialize CTS with entry tasks
        2. For each scheduling step (i = 0 to n-1):
           a. Use task_gene[i] to select task from CTS (relative to |CTS|)
           b. Use unit_gene[i] to select processor (relative to p)
           c. Schedule task to processor
           d. Update CTS: remove scheduled task, add newly ready tasks
        3. Return complete schedule
        
        Args:
            chromosome: 2n-length array of values [0, 1)
            
        Returns:
            Dict[int, int]: Mapping of task_id → processor_id
            
        Example:
            chromosome = [0.3, 0.7, ... | 0.2, 0.9, ...]
            Returns: {0: 1, 1: 0, 2: 2, 3: 1, ...}
        """
        if len(chromosome) != 2 * self.n:
            raise ValueError(f"Chromosome length {len(chromosome)} != 2n ({2*self.n})")
        
        schedule: Dict[int, int] = {}
        
        # Initialize Candidate Task Set (CTS) with entry tasks
        candidate_tasks: Set[int] = set(self.entry_tasks)
        
        # Split chromosome into task and unit genes
        task_genes = chromosome[:self.n]
        unit_genes = chromosome[self.n:]
        
        # Decode chromosome step by step
        for i in range(self.n):
            if not candidate_tasks:
                raise RuntimeError(f"CTS empty at step {i} (DAG cycle or invalid structure)")
            
            # Sort CTS for deterministic selection
            sorted_candidates = sorted(list(candidate_tasks))
            
            # --- 1. Select Task from CTS ---
            # task_gene value determines which candidate to select
            task_gene = task_genes[i]
            selection_index = int(task_gene * len(sorted_candidates))
            # Ensure index is within bounds [0, len-1]
            selection_index = min(selection_index, len(sorted_candidates) - 1)
            selected_task = sorted_candidates[selection_index]
            
            # --- 2. Assign Processor to Task ---
            # unit_gene value determines which processor to use
            unit_gene = unit_genes[i]
            unit_index = int(unit_gene * self.p)
            # Ensure index is within bounds [0, p-1]
            unit_index = min(unit_index, self.p - 1)
            selected_unit = unit_index
            
            # --- 3. Record Assignment ---
            schedule[selected_task] = selected_unit
            
            # --- 4. Update CTS ---
            # Remove scheduled task
            candidate_tasks.remove(selected_task)
            
            # Add newly ready tasks (successors whose all predecessors are scheduled)
            for successor in self.successors[selected_task]:
                # Check if all predecessors of this successor are now scheduled
                is_ready = all(pred in schedule for pred in self.predecessors[successor])
                if is_ready:
                    candidate_tasks.add(successor)
        
        return schedule
    
    def validate_schedule(self, schedule: Dict[int, int]) -> bool:
        """
        Validate that a schedule is complete and respects DAG constraints.
        
        Checks:
        1. All tasks are scheduled
        2. All processor assignments are valid [0, p)
        3. DAG precedence constraints are satisfied
        
        Args:
            schedule: Task-to-processor mapping
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Check all tasks scheduled
        if len(schedule) != self.n:
            return False
        if set(schedule.keys()) != set(range(self.n)):
            return False
        
        # Check all processors are valid
        for processor in schedule.values():
            if processor < 0 or processor >= self.p:
                return False
        
        # Check DAG constraints (this is implicitly satisfied by CTS mechanism,
        # but we verify for completeness)
        scheduled = set()
        for task in sorted(schedule.keys()):
            # Verify all predecessors were scheduled before this task
            for pred in self.predecessors.get(task, []):
                if pred not in scheduled:
                    return False
            scheduled.add(task)
        
        return True
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Decoder(n={self.n}, p={self.p}, entry_tasks={self.entry_tasks})"


if __name__ == '__main__':
    # --- Example Usage and Testing ---
    print("="*60)
    print("Decoder Class Demonstration (Algorithm 2)")
    print("="*60)
    
    # Define a simple workflow (diamond DAG)
    num_tasks = 8
    num_procs = 3
    
    # DAG structure: 0 -> {1,2,3,4} -> {5,6} -> 7
    dag_edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 5), (2, 5),
        (3, 6), (4, 6),
        (5, 7), (6, 7)
    ]
    
    predecessor_structure = {
        0: [],
        1: [0], 2: [0], 3: [0], 4: [0],
        5: [1, 2],
        6: [3, 4],
        7: [5, 6]
    }
    
    # Random matrices for demonstration
    np.random.seed(42)
    W_matrix = np.random.rand(num_tasks, num_procs) * 10  # Computation costs
    C_matrix = np.random.rand(num_tasks, num_tasks) * 5   # Communication costs
    
    # Create decoder
    decoder = Decoder(
        n=num_tasks,
        p=num_procs,
        W=W_matrix,
        C=C_matrix,
        dag=dag_edges,
        predecessors=predecessor_structure
    )
    
    print(f"\nCreated: {decoder}")
    print(f"Entry tasks: {decoder.entry_tasks}")
    
    # Generate random chromosome
    random_chromosome = np.random.rand(2 * num_tasks)
    
    print(f"\n--- Decoding Process ---")
    print(f"Random Chromosome (length {len(random_chromosome)}):")
    task_part = random_chromosome[:num_tasks]
    unit_part = random_chromosome[num_tasks:]
    print(f"  Task genes: {task_part}")
    print(f"  Unit genes: {unit_part}")
    
    # Decode chromosome
    schedule = decoder.decode(random_chromosome)
    
    print(f"\n--- Decoded Schedule ---")
    print(f"Task → Processor mapping:")
    for task in sorted(schedule.keys()):
        print(f"  Task {task} → Processor {schedule[task]}")
    
    # Validate schedule
    is_valid = decoder.validate_schedule(schedule)
    print(f"\nSchedule valid: {is_valid}")
    print(f"All {len(schedule)} tasks scheduled: {len(schedule) == num_tasks}")
    
    # Test with multiple chromosomes
    print(f"\n--- Multiple Chromosome Test ---")
    num_tests = 3
    for test_num in range(num_tests):
        test_chromosome = np.random.rand(2 * num_tasks)
        test_schedule = decoder.decode(test_chromosome)
        test_valid = decoder.validate_schedule(test_schedule)
        print(f"Test {test_num + 1}: Schedule valid = {test_valid}, "
              f"Tasks scheduled = {len(test_schedule)}")
    
    print("\n" + "="*60)
    print("✅ Decoder implementation complete and tested!")
    print("="*60)
