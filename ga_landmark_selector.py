"""
Genetic Algorithm for Optimal Landmark Selection in Non-Convex Environments

This module implements a GA to help robots decide which landmarks to observe
to reduce ambiguity and improve outlier rejection in maze/non-convex environments.
"""

import numpy as np
from typing import List, Tuple, Dict
import random


class LandmarkSelectorGA:
    """
    Genetic Algorithm for selecting optimal landmark observation sets.
    
    Each individual in the population represents a binary vector indicating
    which landmarks to observe. The fitness function evaluates:
    1. Information gain (reduction in position uncertainty)
    2. Geometric diversity (landmarks that reduce ambiguity)
    3. Outlier rejection capability (well-separated landmarks)
    """
    
    def __init__(self, 
                 population_size: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 max_landmarks: int = 10,
                 max_selected: int = 3):
        """
        Initialize the GA.
        
        Args:
            population_size: Number of individuals in population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            max_landmarks: Maximum number of landmarks to consider
            max_selected: Maximum number of landmarks a robot can observe
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_landmarks = max_landmarks
        self.max_selected = max_selected
        
        # Population: list of binary vectors (each vector is a selection)
        self.population = []
        self.fitness_scores = []
        
    def initialize_population(self, num_landmarks: int):
        """Initialize random population of landmark selections."""
        self.population = []
        for _ in range(self.population_size):
            # Random binary vector with at most max_selected ones
            individual = np.zeros(num_landmarks, dtype=bool)
            num_selected = random.randint(1, min(self.max_selected, num_landmarks))
            selected_indices = random.sample(range(num_landmarks), num_selected)
            individual[selected_indices] = True
            self.population.append(individual)
    
    def compute_fitness(self, 
                       individual: np.ndarray,
                       robot_pos: np.ndarray,
                       landmarks: List[np.ndarray],
                       sensor_range: float,
                       walls: List[Tuple[np.ndarray, np.ndarray]] = None) -> float:
        """
        Compute fitness of a landmark selection.
        
        Fitness considers:
        1. Information gain (more landmarks = more info, but with diminishing returns)
        2. Geometric diversity (landmarks that form good triangulation)
        3. Visibility (landmarks must be within range and not blocked by walls)
        4. Ambiguity reduction (landmarks that help disambiguate position)
        
        Args:
            individual: Binary vector indicating selected landmarks
            robot_pos: Current robot position estimate
            landmarks: List of landmark positions
            sensor_range: Maximum sensor range
            walls: List of wall segments (start, end) for line-of-sight checks
            
        Returns:
            Fitness score (higher is better)
        """
        selected_indices = np.where(individual)[0]
        if len(selected_indices) == 0:
            return 0.0
        
        # Get selected landmarks
        selected_landmarks = [landmarks[i] for i in selected_indices]
        
        # 1. Check visibility (within range and not blocked)
        visible_count = 0
        total_info_gain = 0.0
        landmark_positions = []
        
        for lm_pos in selected_landmarks:
            dist = np.linalg.norm(robot_pos - lm_pos)
            
            # Check if within sensor range
            if dist > sensor_range:
                continue
            
            # Check line-of-sight (not blocked by walls)
            if walls is not None:
                if not self._check_line_of_sight(robot_pos, lm_pos, walls):
                    continue
            
            visible_count += 1
            landmark_positions.append(lm_pos)
            
            # Information gain: closer landmarks provide more information
            # But with diminishing returns
            info_gain = 1.0 / (1.0 + dist)  # Inverse distance weighting
            total_info_gain += info_gain
        
        if visible_count == 0:
            return 0.0
        
        # 2. Geometric diversity: landmarks should form good triangulation
        diversity_score = self._compute_geometric_diversity(robot_pos, landmark_positions)
        
        # 3. Ambiguity reduction: landmarks should be well-distributed
        # to reduce position ambiguity (especially in mazes)
        ambiguity_reduction = self._compute_ambiguity_reduction(landmark_positions)
        
        # 4. Penalize too many selections (energy cost, processing overhead)
        selection_penalty = -0.1 * len(selected_indices)
        
        # Combined fitness
        fitness = (total_info_gain * 0.4 + 
                  diversity_score * 0.3 + 
                  ambiguity_reduction * 0.3 + 
                  selection_penalty)
        
        return max(0.0, fitness)  # Ensure non-negative
    
    def _check_line_of_sight(self, 
                            start: np.ndarray, 
                            end: np.ndarray, 
                            walls: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Check if line-of-sight exists between two points (not blocked by walls)."""
        if walls is None or len(walls) == 0:
            return True
        
        # Simple line-segment intersection check
        for wall_start, wall_end in walls:
            if self._line_segments_intersect(start, end, wall_start, wall_end):
                return False
        return True
    
    def _line_segments_intersect(self, 
                                 p1: np.ndarray, p2: np.ndarray,
                                 p3: np.ndarray, p4: np.ndarray) -> bool:
        """Check if two line segments intersect."""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def _compute_geometric_diversity(self, 
                                     robot_pos: np.ndarray,
                                     landmarks: List[np.ndarray]) -> float:
        """Compute how well-distributed landmarks are for triangulation."""
        if len(landmarks) < 2:
            return 0.5  # Single landmark provides some info
        
        # Compute angles from robot to each landmark
        angles = []
        for lm in landmarks:
            dx = lm[0] - robot_pos[0]
            dy = lm[1] - robot_pos[1]
            angle = np.arctan2(dy, dx)
            angles.append(angle)
        
        # Sort angles
        angles = sorted(angles)
        
        # Compute angular spread (want landmarks spread around robot)
        if len(angles) == 2:
            # Two landmarks: want them roughly opposite (180 degrees apart)
            angle_diff = abs(angles[1] - angles[0])
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            # Score: best when close to 180 degrees
            diversity = 1.0 - abs(angle_diff - np.pi) / np.pi
        else:
            # Multiple landmarks: want even distribution
            angle_diffs = []
            for i in range(len(angles)):
                diff = angles[(i+1) % len(angles)] - angles[i]
                if diff < 0:
                    diff += 2 * np.pi
                angle_diffs.append(diff)
            
            # Score based on how uniform the distribution is
            expected_diff = 2 * np.pi / len(angles)
            variance = np.var(angle_diffs)
            diversity = 1.0 / (1.0 + variance / expected_diff**2)
        
        return diversity
    
    def _compute_ambiguity_reduction(self, landmarks: List[np.ndarray]) -> float:
        """Compute how well landmarks reduce position ambiguity."""
        if len(landmarks) < 2:
            return 0.3
        
        # Compute pairwise distances between landmarks
        # Well-separated landmarks reduce ambiguity better
        distances = []
        for i in range(len(landmarks)):
            for j in range(i+1, len(landmarks)):
                dist = np.linalg.norm(landmarks[i] - landmarks[j])
                distances.append(dist)
        
        if len(distances) == 0:
            return 0.3
        
        # Score: want landmarks that are well-separated
        # (helps with outlier rejection and reduces ambiguity)
        avg_distance = np.mean(distances)
        # Normalize: assume max useful distance is ~20 units
        normalized_score = min(1.0, avg_distance / 20.0)
        
        return normalized_score
    
    def evolve(self, 
              robot_pos: np.ndarray,
              landmarks: List[np.ndarray],
              sensor_range: float,
              walls: List[Tuple[np.ndarray, np.ndarray]] = None,
              generations: int = 10,
              track_history: bool = False) -> Tuple[np.ndarray, List[float]]:
        """
        Evolve population to find optimal landmark selection.
        
        Args:
            robot_pos: Current robot position
            landmarks: List of available landmarks
            sensor_range: Sensor range
            walls: Wall segments for line-of-sight
            generations: Number of generations to evolve
            track_history: If True, return fitness history over generations
            
        Returns:
            Tuple of (best individual, fitness history)
            If track_history is False, fitness_history is empty list
        """
        num_landmarks = len(landmarks)
        if num_landmarks == 0:
            return np.array([], dtype=bool), []
        
        # Initialize population if needed
        if len(self.population) == 0 or len(self.population[0]) != num_landmarks:
            self.initialize_population(num_landmarks)
        
        fitness_history = []  # Track best fitness per generation
        
        # Evolve for specified generations
        for generation in range(generations):
            # Evaluate fitness
            self.fitness_scores = []
            for individual in self.population:
                fitness = self.compute_fitness(individual, robot_pos, landmarks, 
                                             sensor_range, walls)
                self.fitness_scores.append(fitness)
            
            # Track best fitness for this generation
            if track_history:
                best_fitness = max(self.fitness_scores)
                avg_fitness = np.mean(self.fitness_scores)
                fitness_history.append({'best': best_fitness, 'avg': avg_fitness})
            
            # Create new population
            new_population = []
            
            # Keep best individual (elitism)
            best_idx = np.argmax(self.fitness_scores)
            new_population.append(self.population[best_idx].copy())
            
            # Generate rest through selection, crossover, mutation
            while len(new_population) < self.population_size:
                # Selection (tournament selection)
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child1 = self._mutate(child1)
                if random.random() < self.mutation_rate:
                    child2 = self._mutate(child2)
                
                # Ensure at least one landmark is selected
                if np.sum(child1) == 0:
                    child1[random.randint(0, num_landmarks-1)] = True
                if np.sum(child2) == 0:
                    child2[random.randint(0, num_landmarks-1)] = True
                
                # Ensure max_selected constraint
                if np.sum(child1) > self.max_selected:
                    # Randomly deselect excess
                    selected = np.where(child1)[0]
                    to_remove = random.sample(list(selected), np.sum(child1) - self.max_selected)
                    child1[to_remove] = False
                
                if np.sum(child2) > self.max_selected:
                    selected = np.where(child2)[0]
                    to_remove = random.sample(list(selected), np.sum(child2) - self.max_selected)
                    child2[to_remove] = False
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            self.population = new_population[:self.population_size]
        
        # Return best individual
        final_fitness = [self.compute_fitness(ind, robot_pos, landmarks, 
                                            sensor_range, walls) 
                        for ind in self.population]
        best_idx = np.argmax(final_fitness)
        return self.population[best_idx], fitness_history
    
    def _tournament_selection(self, tournament_size: int = 3) -> np.ndarray:
        """Tournament selection: pick best from random subset."""
        tournament_indices = random.sample(range(len(self.population)), 
                                         min(tournament_size, len(self.population)))
        tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx].copy()
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover: each bit comes from random parent."""
        mask = np.random.random(len(parent1)) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Flip random bits with mutation rate."""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < 0.1:  # 10% chance per bit
                mutated[i] = not mutated[i]
        return mutated


def select_optimal_landmarks(robot_pos: np.ndarray,
                            landmarks: List[np.ndarray],
                            sensor_range: float,
                            walls: List[Tuple[np.ndarray, np.ndarray]] = None,
                            max_selected: int = 3,
                            ga_generations: int = 10,
                            track_history: bool = False) -> Tuple[List[int], List[Dict[str, float]]]:
    """
    Convenience function to select optimal landmarks using GA.
    
    Args:
        robot_pos: Robot position
        landmarks: Available landmarks
        sensor_range: Sensor range
        walls: Wall segments
        max_selected: Maximum landmarks to select
        ga_generations: GA generations
        track_history: If True, return fitness history
        
    Returns:
        Tuple of (selected landmark indices, fitness history)
        If track_history is False, fitness_history is empty list
    """
    if len(landmarks) == 0:
        return [], []
    
    ga = LandmarkSelectorGA(max_selected=max_selected)
    best_selection, fitness_history = ga.evolve(robot_pos, landmarks, sensor_range, walls, 
                                                ga_generations, track_history=track_history)
    selected_indices = np.where(best_selection)[0].tolist()
    return selected_indices, fitness_history
