"""
Genetic Algorithm for Federated Learning Optimization with Non-IID Data

This module implements a GA to optimize:
1. Federated aggregator weights (translation layer for different robot types)
2. Client selection strategy (which robots to include in each round)
3. Non-IID data handling (Roomba vs Drone different world views)

The problem: Standard federated learning fails with non-IID data (different robot types
have different data distributions). GA finds "good enough" solutions that work across
all robot types without requiring compute-intensive SOTA methods.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import random


class FederatedLearningGA:
    """
    Genetic Algorithm for optimizing federated learning aggregator and client selection.
    
    Each individual represents:
    - Aggregation weights (translation layer) for combining updates from different robot types
    - Client selection strategy (which robots to include)
    - Non-IID handling parameters
    """
    
    def __init__(self,
                 population_size: int = 30,
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.7,
                 num_robot_types: int = 2,  # differential (Roomba) and drone
                 max_clients_per_round: int = None):
        """
        Initialize the GA.
        
        Args:
            population_size: Number of individuals in population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            num_robot_types: Number of different robot types (e.g., 2 for Roomba + Drone)
            max_clients_per_round: Maximum clients to select per round (None = all)
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_robot_types = num_robot_types
        self.max_clients_per_round = max_clients_per_round
        
        # Population: list of individuals
        # Each individual is a dict with:
        #   - 'aggregation_weights': np.array of shape (num_robot_types,) - weights for each type
        #   - 'client_selection': np.array of shape (num_robots,) - binary selection
        #   - 'non_iid_params': np.array - parameters for handling non-IID data
        self.population = []
        self.fitness_scores = []
        
    def initialize_population(self, num_robots: int, robot_types: List[str]):
        """
        Initialize random population.
        
        Args:
            num_robots: Total number of robots
            robot_types: List of robot type strings (e.g., ['differential', 'drone', ...])
        """
        self.num_robots = num_robots
        self.robot_types = robot_types
        unique_types = list(set(robot_types))
        self.num_robot_types = len(unique_types)
        self.type_to_index = {t: i for i, t in enumerate(unique_types)}
        
        self.population = []
        for _ in range(self.population_size):
            individual = {
                # Aggregation weights: one per robot type (sum to 1.0)
                'aggregation_weights': self._random_weights(self.num_robot_types),
                # Client selection: binary vector (which robots to include)
                'client_selection': self._random_client_selection(num_robots),
                # Non-IID parameters: [temperature, diversity_weight, fairness_weight]
                'non_iid_params': np.array([
                    np.random.uniform(0.1, 2.0),  # Temperature for soft aggregation
                    np.random.uniform(0.0, 1.0),  # Diversity weight
                    np.random.uniform(0.0, 1.0)   # Fairness weight
                ])
            }
            self.population.append(individual)
    
    def _random_weights(self, n: int) -> np.ndarray:
        """Generate random normalized weights."""
        weights = np.random.uniform(0.1, 1.0, n)
        return weights / weights.sum()
    
    def _random_client_selection(self, num_robots: int) -> np.ndarray:
        """Generate random client selection (binary vector)."""
        if self.max_clients_per_round is None:
            # Select random subset (at least 50% of clients)
            num_selected = random.randint(max(1, num_robots // 2), num_robots)
        else:
            num_selected = min(self.max_clients_per_round, num_robots)
        
        selection = np.zeros(num_robots, dtype=bool)
        selected_indices = random.sample(range(num_robots), num_selected)
        selection[selected_indices] = True
        return selection
    
    def compute_fitness(self,
                       individual: Dict,
                       robot_updates: List[Dict],
                       robot_types: List[str],
                       current_global_model: Optional[np.ndarray] = None) -> float:
        """
        Compute fitness of an individual.
        
        Fitness considers:
        1. Aggregation quality (how well updates combine)
        2. Client diversity (non-IID handling)
        3. Convergence speed (how fast the model improves)
        4. Fairness (all robot types contribute)
        
        Args:
            individual: Individual to evaluate
            robot_updates: List of updates from each robot
                Each update is a dict with 'weights', 'type', 'data_size', 'loss'
            robot_types: List of robot type strings
            current_global_model: Current global model (optional)
            
        Returns:
            Fitness score (higher is better)
        """
        # Get selected clients
        selected_indices = np.where(individual['client_selection'])[0]
        if len(selected_indices) == 0:
            return 0.0
        
        # Filter updates to selected clients
        selected_updates = [robot_updates[i] for i in selected_indices]
        selected_types = [robot_types[i] for i in selected_indices]
        
        if len(selected_updates) == 0:
            return 0.0
        
        # 1. Aggregation quality: weighted combination of updates
        aggregation_score = self._compute_aggregation_quality(
            individual, selected_updates, selected_types
        )
        
        # 2. Client diversity: ensure different robot types are represented
        diversity_score = self._compute_diversity_score(selected_types)
        
        # 3. Non-IID handling: fairness across robot types
        fairness_score = self._compute_fairness_score(
            individual, selected_updates, selected_types
        )
        
        # 4. Update quality: lower loss is better
        quality_score = self._compute_update_quality(selected_updates)
        
        # Combined fitness (weighted sum)
        fitness = (
            0.4 * aggregation_score +
            0.2 * diversity_score +
            0.2 * fairness_score +
            0.2 * quality_score
        )
        
        return fitness
    
    def _compute_aggregation_quality(self,
                                    individual: Dict,
                                    updates: List[Dict],
                                    types: List[str]) -> float:
        """Compute how well updates aggregate using the translation layer."""
        weights = individual['aggregation_weights']
        type_to_index = self.type_to_index
        
        # Weight updates by robot type
        weighted_sum = 0.0
        total_weight = 0.0
        
        for update, robot_type in zip(updates, types):
            type_idx = type_to_index.get(robot_type, 0)
            type_weight = weights[type_idx]
            
            # Use update quality (inverse of loss) weighted by type
            update_quality = 1.0 / (update.get('loss', 1.0) + 0.1)
            weighted_sum += type_weight * update_quality * update.get('data_size', 1)
            total_weight += type_weight * update.get('data_size', 1)
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.0
    
    def _compute_diversity_score(self, types: List[str]) -> float:
        """Compute diversity: more types represented = better."""
        unique_types = len(set(types))
        max_types = self.num_robot_types
        return unique_types / max_types if max_types > 0 else 0.0
    
    def _compute_fairness_score(self,
                               individual: Dict,
                               updates: List[Dict],
                               types: List[str]) -> float:
        """Compute fairness: ensure all robot types contribute fairly."""
        type_counts = {}
        type_contributions = {}
        
        for update, robot_type in zip(updates, types):
            type_counts[robot_type] = type_counts.get(robot_type, 0) + 1
            type_contributions[robot_type] = type_contributions.get(robot_type, 0.0) + update.get('data_size', 1)
        
        # Fairness: contributions should be balanced across types
        if len(type_counts) == 0:
            return 0.0
        
        contributions = list(type_contributions.values())
        if len(contributions) == 0:
            return 0.0
        
        # Lower variance = more fair
        mean_contrib = np.mean(contributions)
        if mean_contrib == 0:
            return 0.0
        
        variance = np.var(contributions) / (mean_contrib ** 2 + 1e-6)
        fairness = 1.0 / (1.0 + variance)  # Higher when variance is lower
        
        return fairness
    
    def _compute_update_quality(self, updates: List[Dict]) -> float:
        """Compute average quality of updates (lower loss = better)."""
        if len(updates) == 0:
            return 0.0
        
        losses = [update.get('loss', 1.0) for update in updates]
        avg_loss = np.mean(losses)
        quality = 1.0 / (avg_loss + 0.1)  # Inverse of loss
        
        return quality
    
    def select_parents(self, tournament_size: int = 3) -> Tuple[Dict, Dict]:
        """Tournament selection."""
        def tournament():
            candidates = random.sample(list(zip(self.population, self.fitness_scores)), tournament_size)
            return max(candidates, key=lambda x: x[1])[0]
        
        return tournament(), tournament()
    
    def crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Uniform crossover for all components."""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Crossover aggregation weights
        mask = np.random.rand(self.num_robot_types) < 0.5
        child1_weights = np.where(mask, parent1['aggregation_weights'], parent2['aggregation_weights'])
        child2_weights = np.where(~mask, parent1['aggregation_weights'], parent2['aggregation_weights'])
        child1_weights = child1_weights / child1_weights.sum()
        child2_weights = child2_weights / child2_weights.sum()
        
        # Crossover client selection
        mask = np.random.rand(self.num_robots) < 0.5
        child1_selection = np.where(mask, parent1['client_selection'], parent2['client_selection'])
        child2_selection = np.where(~mask, parent1['client_selection'], parent2['client_selection'])
        
        # Crossover non-IID params
        mask = np.random.rand(3) < 0.5
        child1_params = np.where(mask, parent1['non_iid_params'], parent2['non_iid_params'])
        child2_params = np.where(~mask, parent1['non_iid_params'], parent2['non_iid_params'])
        
        child1 = {
            'aggregation_weights': child1_weights,
            'client_selection': child1_selection,
            'non_iid_params': child1_params
        }
        child2 = {
            'aggregation_weights': child2_weights,
            'client_selection': child2_selection,
            'non_iid_params': child2_params
        }
        
        return child1, child2
    
    def mutate(self, individual: Dict):
        """Mutate individual components."""
        # Mutate aggregation weights
        if random.random() < self.mutation_rate:
            noise = np.random.normal(0, 0.1, self.num_robot_types)
            individual['aggregation_weights'] += noise
            individual['aggregation_weights'] = np.clip(individual['aggregation_weights'], 0.01, 1.0)
            individual['aggregation_weights'] = individual['aggregation_weights'] / individual['aggregation_weights'].sum()
        
        # Mutate client selection
        if random.random() < self.mutation_rate:
            # Flip random bits
            num_flips = max(1, int(self.num_robots * 0.1))
            indices = random.sample(range(self.num_robots), min(num_flips, self.num_robots))
            for idx in indices:
                individual['client_selection'][idx] = not individual['client_selection'][idx]
        
        # Mutate non-IID params
        if random.random() < self.mutation_rate:
            noise = np.random.normal(0, 0.1, 3)
            individual['non_iid_params'] += noise
            individual['non_iid_params'] = np.clip(individual['non_iid_params'], [0.1, 0.0, 0.0], [2.0, 1.0, 1.0])
    
    def evolve(self,
              robot_updates: List[Dict],
              robot_types: List[str],
              generations: int = 10,
              track_history: bool = False) -> Tuple[Dict, List[Dict]]:
        """
        Evolve population to find optimal federated learning configuration.
        
        Returns:
            Best individual, fitness history
        """
        fitness_history = []
        
        for gen in range(generations):
            # Evaluate fitness
            self.fitness_scores = []
            for individual in self.population:
                fitness = self.compute_fitness(individual, robot_updates, robot_types)
                self.fitness_scores.append(fitness)
            
            if track_history:
                best_fitness = max(self.fitness_scores)
                avg_fitness = np.mean(self.fitness_scores)
                fitness_history.append({
                    'generation': gen,
                    'best': best_fitness,
                    'avg': avg_fitness
                })
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individual
            best_idx = np.argmax(self.fitness_scores)
            new_population.append(self._deep_copy_individual(self.population[best_idx]))
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents()
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                self.mutate(child2)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            self.population = new_population
        
        # Return best individual
        self.fitness_scores = []
        for individual in self.population:
            fitness = self.compute_fitness(individual, robot_updates, robot_types)
            self.fitness_scores.append(fitness)
        
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx], fitness_history
    
    def _deep_copy_individual(self, individual: Dict) -> Dict:
        """Deep copy an individual."""
        return {
            'aggregation_weights': individual['aggregation_weights'].copy(),
            'client_selection': individual['client_selection'].copy(),
            'non_iid_params': individual['non_iid_params'].copy()
        }
    
    def aggregate_updates(self,
                         individual: Dict,
                         robot_updates: List[Dict],
                         robot_types: List[str]) -> np.ndarray:
        """
        Aggregate robot updates using the optimized configuration.
        
        Returns:
            Aggregated global model update
        """
        selected_indices = np.where(individual['client_selection'])[0]
        if len(selected_indices) == 0:
            return None
        
        selected_updates = [robot_updates[i] for i in selected_indices]
        selected_types = [robot_types[i] for i in selected_indices]
        
        weights = individual['aggregation_weights']
        type_to_index = self.type_to_index
        
        # Weighted aggregation
        aggregated = None
        total_weight = 0.0
        
        for update, robot_type in zip(selected_updates, selected_types):
            type_idx = type_to_index.get(robot_type, 0)
            type_weight = weights[type_idx]
            
            # Get update vector (model weights or gradient)
            update_vector = update.get('weights', np.array([0.0]))
            data_size = update.get('data_size', 1)
            
            # Weight by type and data size
            weighted_update = type_weight * data_size * update_vector
            
            if aggregated is None:
                aggregated = weighted_update
            else:
                aggregated += weighted_update
            
            total_weight += type_weight * data_size
        
        if total_weight > 0 and aggregated is not None:
            aggregated = aggregated / total_weight
        
        return aggregated
