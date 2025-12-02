# SmartGA: Intelligent Multi-Objective Workflow Scheduling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SmartGA is a novel multi-objective genetic algorithm for cloud-edge workflow scheduling that achieves **22% better makespan** compared to standard GA and **4.2% improvement** over state-of-the-art R2GA.

## Key Features

- **Heuristic Seeding**: 30% faster convergence using HEFT, EFT, MinMin, and MaxMin
- **Adaptive Parameter Control**: Dynamic mutation/crossover rates based on population metrics
- **Multi-Objective Optimization**: Simultaneous optimization of makespan, cost, and load balancing
- **Sub-Linear Complexity**: O(n^0.73) runtime scaling suitable for online scheduling
- **Perfect Hypervolume**: Complete Pareto-optimal coverage (HV = 1.000)

## Performance Highlights

| Metric | SmartGA | R2GA | NSGA-II | GA |
|--------|---------|------|---------|-----|
| Makespan | **6.81s** | 7.09s | 7.35s | 8.31s |
| Hypervolume | **1.000** | 0.980 | 0.960 | 0.000 |
| Convergence | **Gen 25** | Gen 35 | Gen 40 | Gen 45 |
| Runtime | **0.85s** | 0.87s | 0.89s | 0.81s |

### Quick Start

### Installation

This project **requires** a local virtual environment (`venv`) to ensure dependencies are isolated and do not conflict with your system.

> **Note for Ubuntu/Debian users**: If you encounter an error creating the venv, you may need to install the `venv` module:
> `sudo apt install python3.10-venv`

**Option 1: Automated Setup (Recommended)**

Run the setup script to automatically create the `venv`, install dependencies, and get ready:

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

**Option 2: Manual Setup**

```bash
# 1. Create a virtual environment in the project directory
python3 -m venv venv

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run Experiments

The project uses a configuration file to control experiments.

```bash
# Run experiment using default config
python main.py --config config/config.yaml
```

To change experiment parameters (algorithm, workflow size, etc.), edit `config/config.yaml`.

## Project Structure

```
├── algorithms/          # Algorithm implementations
│   ├── smartga_v2/     # SmartGA core components
│   ├── nsga2.py        # NSGA-II baseline
│   ├── mor2ga.py       # R2GA implementation
│   ├── ga.py           # Standard GA
│   ├── psosa.py        # PSO-SA
│   └── woa.py          # WOA
├── simulator/          # Workflow execution simulator
├── workflows/          # Workflow generators (Montage, CyberShake, etc.)
├── FinalResult/        # Research paper sections and figures
│   ├── Section1_Report.md
│   ├── Section2_Report.md
│   ├── Section3_Report.md
│   ├── Section4_Report.md
│   ├── Final_Proposed_Methodology.md
│   └── preamble.md
├── main.py             # Main execution script
└── requirements.txt    # Python dependencies
```

## Research Paper

This repository contains the implementation for the research paper:

**"SmartGA: An Intelligent Multi-Objective Genetic Algorithm for Cloud-Edge Workflow Scheduling"**


## Algorithm Components

### 1. Heuristic Seeding

Initializes 30% of population using classical heuristics:
- HEFT (Heterogeneous Earliest Finish Time)
- EFT (Earliest Finish Time)
- MinMin (Minimum completion time)
- MaxMin (Maximum completion time)

### 2. Adaptive Parameter Control

Dynamically adjusts parameters based on:
- Population diversity (D)
- Hypervolume (HV)
- Pareto front size (|PF|)

### 3. Real-Relative Encoding

Chromosome: `[p1, p2, ..., pn, r1, r2, ..., rn]`
- `pi`: Task priority [0,1]
- `ri`: Resource preference [0,1]


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Dept. of CSE (M.tech)**  
National Institute of Technology Karnataka, Surathkal, Mangalore - 575025, Bharat

## Acknowledgments

- NSGA-II implementation based on Deb et al. (2002)
- HEFT heuristic from Topcuoglu et al. (2002)
- Workflow benchmarks from Pegasus WMS

**Keywords**: Cloud-edge computing, workflow scheduling, multi-objective optimization, genetic algorithms, NSGA-II, heuristic seeding, adaptive parameter control, Pareto optimization
