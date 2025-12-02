"""
PSO-SA Module

Hybrid PSO-SA (Particle Swarm Optimization + Simulated Annealing)
algorithm for workflow scheduling.

Combines swarm intelligence with probabilistic acceptance to
escape local optima.
"""

from .pso_sa import PSOSAScheduler

__all__ = ['PSOSAScheduler']
