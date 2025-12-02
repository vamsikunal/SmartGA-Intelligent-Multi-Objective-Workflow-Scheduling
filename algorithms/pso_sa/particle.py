"""
PSO-SA Particle

Particle class with Simulated Annealing acceptance criterion.
Each particle maintains its own state and applies SA to accept/reject
solutions during PSO evolution.
"""

import numpy as np
import random


class PSO_SA_Particle:
    """
    Particle for PSO-SA hybrid algorithm.
    
    Combines standard PSO particle with Simulated Annealing
    acceptance criterion to escape local optima.
    
    Attributes:
        position: Current position (processor assignments)
        velocity: Current velocity vector
        pBest: Personal best position found
        pBest_fitness: Fitness of personal best
        fitness: Current fitness value
        temperature: Current SA temperature
        E: Current energy difference
        EA: Average energy difference (adaptive)
        n_accepted: Number of solutions accepted
    """
    
    def __init__(self, n, p, initial_temp):
        """
        Initialize particle with random position and velocity.
        
        Args:
            n: Number of tasks (dimension of search space)
            p: Number of processors (bounds)
            initial_temp: Initial SA temperature
        """
        self.n = n
        self.p = p
        
        # Initialize position randomly in [0, p-1]
        self.position = np.random.uniform(0, p - 1, size=n)
        
        # Initialize velocity randomly in [-v_max, v_max]
        v_max = (p - 1) / 2
        self.velocity = np.random.uniform(-v_max, v_max, size=n)
        
        # Personal best (initially same as position)
        self.pBest = self.position.copy()
        self.pBest_fitness = float('inf')  # Minimization
        
        # Current state
        self.fitness = float('inf')
        
        # SA parameters
        self.temperature = initial_temp
        self.E = 0.0  # Energy difference
        self.EA = 0.0  # Average energy
        self.n_accepted = 1  # Count of accepted solutions
    
    def evaluate_with_SA(self, fitness_func, current_temp):
        """
        Evaluate fitness and apply SA acceptance criterion.
        
        This is the key innovation of PSO-SA: instead of always
        accepting better pBest, we use SA probability to sometimes
        accept worse solutions, helping escape local optima.
        
        Args:
            fitness_func: Function to evaluate makespan
            current_temp: Current global temperature
        
        Returns:
            bool: True if solution was accepted
        """
        # Evaluate current position
        self.fitness = fitness_func(self.position)
        
        # Calculate energy difference
        self.E = abs(self.fitness - self.pBest_fitness)
        
        # Acceptance decision
        accept = False
        
        if self.fitness < self.pBest_fitness:
            # Better solution - always accept
            accept = True
        else:
            # Worse solution - SA acceptance probability
            if self.EA > 0 and current_temp > 0:
                P = np.exp(-self.E / (self.EA * current_temp))
                accept = (random.random() < P)
            else:
                accept = False
        
        # Update if accepted
        if accept:
            self.pBest = self.position.copy()
            self.pBest_fitness = self.fitness
            self.n_accepted += 1
            
            # Update average energy (adaptive)
            self.EA = (self.EA * (self.n_accepted - 1) + self.E) / self.n_accepted
        
        return accept
    
    def update_velocity(self, gBest, w, c1, c2):
        """
        Update particle velocity using PSO formula.
        
        v = w*v + c1*r1*(pBest - x) + c2*r2*(gBest - x)
        
        Args:
            gBest: Global best position
            w: Inertia weight
            c1: Cognitive coefficient (personal best attraction)
            c2: Social coefficient (global best attraction)
        """
        r1 = np.random.random(self.n)
        r2 = np.random.random(self.n)
        
        # PSO velocity update
        cognitive = c1 * r1 * (self.pBest - self.position)
        social = c2 * r2 * (gBest - self.position)
        self.velocity = w * self.velocity + cognitive + social
        
        # Clamp velocity to prevent excessive drift
        v_max = (self.p - 1) / 2
        self.velocity = np.clip(self.velocity, -v_max, v_max)
    
    def update_position(self):
        """
        Update particle position and enforce bounds.
        
        Position update: x = x + v
        Bounds: [0, p-1] for each dimension
        """
        # Update position
        self.position = self.position + self.velocity
        
        # Enforce bounds [0, p-1]
        self.position = np.clip(self.position, 0, self.p - 1)
    
    def get_integer_position(self):
        """
        Get position rounded to integers for evaluation.
        
        Returns:
            Integer array of processor assignments
        """
        return np.round(self.position).astype(int)
