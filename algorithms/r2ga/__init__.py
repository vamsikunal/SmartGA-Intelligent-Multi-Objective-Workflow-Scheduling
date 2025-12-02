"""
R²GA (Real Relative Encoding Genetic Algorithm) Package

A genetic algorithm-based metaheuristic for workflow scheduling in 
heterogeneous distributed computing systems.

Based on: J. Jiang et al., IEEE TPDS 2025

Components:
- r2ga.py: Main R2GA scheduler class
- encoding.py: Chromosome encoding (Algorithm 1)
- decoding.py: Chromosome decoding (Algorithm 2)
- evolutionary_operators.py: Population initialization and selection (Algorithms 3-4)
- genetic_operators.py: Crossover and mutation (Algorithms 5-6)
"""

from .r2ga import R2GA

__all__ = ['R2GA']
__version__ = '1.0.0'
__author__ = 'MADGA Development Team'
