# AWS Configuration Profiles Guide

## Available Configurations

### 1. `aws_realistic.yaml` - Balanced Real-World Scenario
**Best for:** Understanding typical Cloud-Edge trade-offs

**Resources:**
- 2x AWS c6i.2xlarge Cloud (15K MIPS, $3.40/10hr)
- 8x Edge Gateways (2.5K MIPS, $0.05/10hr)

**Characteristics:**
- Cloud is 6x faster but 68x more expensive
- Moderate complexity
- Good for initial experiments

**Expected Outcome:**
- HEFT will heavily favor Cloud
- Multi-objective will find cost/energy trade-offs

**Run:**
```bash
python3 main.py --config config/aws_realistic.yaml
```

---

### 2. `aws_extreme.yaml` - Maximum Challenge for HEFT
**Best for:** Proving multi-objective algorithms outperform HEFT

**Resources:**
- 1x AWS c6i.4xlarge Cloud (25K MIPS, $10/10hr) ⚠️ BOTTLENECK
- 12x Edge nodes (3K MIPS, $0.02/10hr)

**Characteristics:**
- Cloud is VERY expensive (500x cost)
- Only 1 cloud node = severe bottleneck
- Extreme objective conflicts
- Limited bandwidth to cloud (500 Mbps)

**Expected Outcome:**
- **HEFT will perform POORLY** (greedy → all tasks to 1 cloud node = serialization!)
- **NSGA-II/MOPSO will EXCEL** (distribute across cheap edge nodes)
- Clear demonstration of multi-objective advantages

**Run:**
```bash
python3 main.py --config config/aws_extreme.yaml
```

---

### 3. `aws_multitier.yaml` - Real Enterprise Setup
**Best for:** Realistic multi-tier fog computing

**Resources:**
- 3x AWS Cloud (12K MIPS, $2.5/10hr)
- 6x Fog/Edge Gateways (5K MIPS, $0.2/10hr)

**Characteristics:**
- Three-tier architecture
- Balanced performance/cost ratio
- Fog layer provides middle ground

**Expected Outcome:**
- Complex scheduling decisions
- Multiple Pareto-optimal solutions
- Trade-offs between all 5 objectives visible

**Run:**
```bash
python3 main.py --config config/aws_multitier.yaml
```

---

## Quick Comparison

| Config | Cloud Nodes | Edge Nodes | Best For | HEFT Performance |
|--------|-------------|------------|----------|------------------|
| **Realistic** | 2 (Fast) | 8 (Slow) | Balanced testing | Good |
| **Extreme** | 1 (Bottleneck) | 12 (Cheap) | Breaking HEFT | **Poor** ⭐ |
| **Multitier** | 3 (Fast) | 6 (Medium) | Enterprise scenarios | Medium |

---

## Recommended Experiment Sequence

### Step 1: Baseline (Current Config)
```bash
python3 main.py --config config/config.yaml
```
Small workflow, see HEFT dominate

### Step 2: Realistic AWS
```bash
python3 main.py --config config/aws_realistic.yaml
```
500 tasks, start seeing trade-offs

### Step 3: Extreme Challenge
```bash
python3 main.py --config config/aws_extreme.yaml
```
**800 tasks, HEFT fails, multi-objective wins!** ⭐

### Step 4: Compare Results
```bash
# Check makespan differences
cat results/*/all_results.csv | grep "HEFT\|NSGA"
```

---

## Key Parameters Explained

### Cost (per 10-hour workday)
- **Cloud**: $2.50 - $10.00 (AWS pricing)
- **Edge**: $0.02 - $0.20 (depreciation/power)

### MIPS (Performance)
- **AWS c6i.4xlarge**: 25,000 MIPS
- **AWS c6i.2xlarge**: 15,000 MIPS
- **Edge Gateway**: 2,500 - 5,000 MIPS

### Failure Rate
- **Cloud**: 0.00001 (99.999% uptime)
- **Edge**: 0.0005 - 0.001 (99.9% - 99.95% uptime)

### Power (Watts)
- **Cloud**: 150-250W (data center)
- **Edge**: 15-30W (embedded devices)

---

## Tips for Best Results

1. **Use `aws_extreme.yaml` for publications** - Shows clear multi-objective superiority
2. **Increase workflow size** (500-1000 tasks) for statistical significance
3. **Run multiple times** and average results
4. **Compare Pareto fronts** visually in generated plots

---

## Creating Your Own Config

Copy and modify any config file:

```yaml
resources:
  cloud_nodes:
    count: X           # Number of cloud instances
    mips: Y            # Performance (based on EC2 type)
    cost: Z            # $/10hr (from AWS pricing)
    power_active: W    # Watts (from spec sheets)
    failure_rate: R    # Based on SLA (99.9% = 0.001)
```

**AWS EC2 Reference:**
- t3.medium: ~3,000 MIPS, $0.0416/hr
- c6i.xlarge: ~12,000 MIPS, $0.17/hr
- c6i.2xlarge: ~15,000 MIPS, $0.34/hr
- c6i.4xlarge: ~25,000 MIPS, $0.68/hr
