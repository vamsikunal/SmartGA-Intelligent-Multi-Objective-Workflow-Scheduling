# R²GA - Real Relative Encoding Genetic Algorithm

A genetic algorithm-based metaheuristic scheduler for workflow scheduling in heterogeneous distributed computing systems.

## Overview

R²GA uses a novel **real-valued relative encoding** scheme where chromosomes are 2n-length arrays of random real numbers in [0, 1). These values are interpreted **relatively** during decoding - each gene's value determines which available choice to make at each scheduling step.

## Algorithm Components

### 1. Encoding (Algorithm 1)
- **Chromosome Structure**: 2n real values in [0, 1)
  - First n values: Task sequencing genes
  - Second n values: Unit (processor) assignment genes
- **Random Generation**: Each gene uniformly sampled from [0, 1)

### 2. Decoding (Algorithm 2)
- **CTS (Candidate Task Set)**: Tasks whose predecessors are all scheduled
- **Relative Interpretation**: Gene value selects from available options
- **Process**:
  1. Start with entry tasks in CTS
  2. For each scheduling step:
     - Use task gene to select task from CTS (relative to CTS size)
     - Use unit gene to select processor (relative to processor count)
     - Update CTS with newly available tasks
  3. Repeat until all tasks scheduled

### 3. Initial Population (Algorithm 3)
- **Population Size**: Configurable (default: 50)
- **HEFT Seeding**: Optionally include one HEFT solution for quality
- **Random Chromosomes**: Remaining population randomly generated

### 4. Selection (Algorithm 4)
- **Elitism**: Preserve top N solutions (default: 5)
- **Fitness**: Lower makespan = better fitness

### 5. Crossover (Algorithm 5) ⭐ **TA-Approved Specification**
- **Type**: Two-point crossover (NOT single-point or uniform)
- **Process**:
  1. Select random crossover point in task-sequencing part [0, n)
  2. Select random crossover point in unit-assignment part [n, 2n)
  3. Swap segment between these two points with other parent
- **Preserves**: Chromosome structure (task genes + unit genes)
- **Reference**: Algorithm 5 and Figure 3 in paper

### 6. Mutation (Algorithm 6) ⭐ **TA-Approved Specification**
- **Type**: Two-point random replacement (NOT Gaussian or uniform)
- **Process**:
  1. Select ONE random gene in task-sequencing part [0, n)
  2. Select ONE random gene in unit-assignment part [n, 2n)
  3. Replace each with NEW random value [0, 1)
- **Maintains**: Diversity while respecting 2n structure
- **Reference**: Algorithm 6 and Figure 4 in paper

## Usage

### Basic Usage

```python
from src.baselines.r2ga import R2GA
from src.baselines import ResourceConfig

# Create R²GA scheduler with default parameters
r2ga = R2GA()

# Define resources
resources = ResourceConfig(
    num_processors=4,
    processor_speeds=np.array([1.0, 1.2, 1.5, 2.0])
)

# Run R²GA
result = r2ga.schedule(workflow, resources)

# Results
print(f"Makespan: {result.makespan:.2f}")
print(f"Algorithm: {result.algorithm_name}")
```

### Custom Parameters

```python
# Create R²GA with custom GA parameters
r2ga = R2GA(
    pop_size=100,           # Population size
    elite_size=10,          # Number of elites to preserve
    iterations=200,         # Maximum generations
    mutation_rate=0.15,     # Mutation probability
    crossover_rate=0.85,    # Crossover probability
    seed_with_heft=True,    # Include HEFT seed
    convergence_threshold=30,  # Early stopping threshold
    verbose=True            # Show progress
)

result = r2ga.schedule(workflow, resources)
```

### Accessing GA Metadata

```python
result = r2ga.schedule(workflow, resources)

# GA parameters used
ga_params = result.metadata['ga_parameters']
print(f"Population size: {ga_params['pop_size']}")
print(f"Iterations run: {ga_params['actual_iterations']}")

# Convergence data
convergence = result.metadata['convergence_data']
print(f"Best fitness history: {convergence['best_fitness_history']}")
print(f"Converged early: {convergence['converged_early']}")

# Plot convergence
import matplotlib.pyplot as plt
plt.plot(convergence['best_fitness_history'])
plt.xlabel('Iteration')
plt.ylabel('Best Makespan')
plt.title('R²GA Convergence')
plt.show()
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pop_size` | 50 | Population size (number of chromosomes) |
| `elite_size` | 5 | Number of elite solutions to preserve |
| `iterations` | 100 | Maximum number of generations |
| `mutation_rate` | 0.1 | Probability of mutation per chromosome |
| `crossover_rate` | 0.9 | Probability of crossover per parent pair |
| `seed_with_heft` | True | Include HEFT solution in initial population |
| `convergence_threshold` | 20 | Iterations without improvement for early stopping |
| `verbose` | False | Enable verbose logging |
