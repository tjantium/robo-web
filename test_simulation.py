import sys
import os
from typing import Any, Dict
# Set Qt API before importing matplotlib
os.environ['QT_API'] = 'pyqt6'
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QCheckBox, 
                             QDoubleSpinBox, QSpinBox, QGroupBox, QScrollArea, QTabWidget)
from PyQt6.QtCore import Qt, QTimer
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Circle as MplCircle
from matplotlib import gridspec
from robot_m import RobotAgent
from rvo import RVO
from ga_landmark_selector import LandmarkSelectorGA, select_optimal_landmarks
from federated_ga import FederatedLearningGA

class RobotWebGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Web: Distributed GBP Localisation")
        self.setMinimumSize(1200, 800)

        # 1. Simulation State
        self.time = 0.0
        self.dt = 0.1
        self.iter_count = 0
        self.fps = 0
        self.fps_counter = 0
        self.fps_timer = 0
        
        # Settings
        self.num_robots = 4  # Number of differential drive robots (Roomba-like)
        self.circle_radius = 8.0  # Increased for better spacing
        self.circle_center = np.array([0.0, 0.0])
        self.sensor_range = 10.0  # Increased sensor range
        self.noise_std = [0.1, 0.05]
        self.iter_before_motion = 6
        self.linearize_every = 2
        self.damping = 0.2
        self.use_landmark_only = False
        self.is_robust = False
        self.boundary_radius = 20.0  # Increased boundary for more space
        self.use_boundary_constraint = True  # Keep robots within boundary
        self.use_motion_model = True  # Use circular motion model to help tracking
        self.motion_model_weight = 0.1  # Weight for motion model prediction
        
        # RVO settings
        self.use_rvo = True  # Enable RVO collision avoidance for differential robots
        self.rvo_time_horizon = 3.0  # Increased for better planning
        self.rvo_neighbor_dist = 3.5  # Increased to keep robots more separated
        self.rvo_max_speed = 0.8  # Slightly reduced for smoother motion
        self.min_robot_separation = 1.5  # Minimum distance between robot centers (2 * radius + safety)
        
        # Obstacle settings
        self.num_obstacles = 3  # Reduced for cleaner demo
        self.obstacle_radius = 1.2
        self.obstacles = []  # List of obstacle positions
        self.obstacle_velocities = []  # List of obstacle velocities
        self.obstacle_moving = True  # Whether obstacles move
        
        # Maze/Non-convex environment settings
        # MAZE DISABLED - Commented out for federated learning focus
        # self.use_maze = False  # Enable maze environment
        self.use_maze = False  # Disabled for federated learning
        self.maze_walls = []  # List of wall segments: [(start, end), ...]
        self.maze_complexity = 5  # Number of walls in maze
        
        # GA Landmark Selection settings
        self.use_ga_landmark_selection = False  # Use GA to select optimal landmarks
        self.ga_max_selected = 3  # Maximum landmarks to select via GA
        self.ga_generations = 10  # GA evolution generations
        self.robot_selected_landmarks = {}  # {robot_id: [landmark_indices]}
        self.ga_fitness_history = []  # Track GA evolution: [{robot_id: [{'best': float, 'avg': float}, ...]}, ...]
        self.ga_evolution_data = {}  # {robot_id: [{'best': float, 'avg': float}, ...]} - latest evolution
        
        # Federated Learning tracking
        self.federated_fitness_history = []  # Track FL GA fitness over rounds
        self.federated_aggregation_weights_history = []  # Track aggregation weights over rounds
        self.federated_client_selection_history = []  # Track which clients selected
        self.federated_metrics_history = []  # Track diversity, fairness, quality metrics
        
        # SLAM visualization: Track observation history for factor graph
        self.slam_observations = {}  # {robot_id: [(pose, landmark_idx, iteration), ...]}
        self.show_slam_visualization = True  # Enable SLAM factor graph visualization
        
        # Technical goals settings
        self.use_odometry = True  # Each robot has odometry factors (local fragment)
        self.odometry_noise_std = [0.15, 0.15]  # Odometry uncertainty
        self.async_communication = True  # Simulate asynchronous communication
        self.communication_drop_rate = 0.1  # 10% packet loss
        self.communication_delay = 0.0  # Message delay in iterations
        self.robust_threshold = 2.0  # Threshold for robust factor outlier detection
        self.allow_dynamic_join_leave = True  # Robots can join/leave
        
        # Visualization settings
        self.show_factors = True
        self.show_only_landmark_factors = False
        self.show_path = True
        self.show_samples = False
        self.follow_robot = False
        self.robot_id_to_follow = 0
        self.step_by_step = False
        self.running = True
        
        # Debug/Statistics
        self.total_messages = 0
        self.messages_per_iteration = 0
        self.last_update_stats = {}
        self.debug_console = True  # Print to console
        
        # GPB Performance Tracking
        self.gpb_error_history = []  # Track average position errors over time
        self.gpb_robot_errors = {}  # Track individual robot errors {robot_id: [errors]}
        self.gpb_convergence_history = []  # Track convergence metrics
        self.gpb_message_history = []  # Track messages per iteration
        self.max_history_length = 500  # Keep last 500 data points
        
        # Track active inter-robot connections for visualization
        self.active_connections = set()  # Set of (robot_id1, robot_id2) tuples
        self.message_queue_sizes = {}  # Track message queue sizes per robot
        self.dropped_messages_count = 0  # Count dropped messages
        
        # Create robots in circular formation
        self.robots = []
        self.robot_paths = {}  # Store paths for each robot
        self.robot_gt_paths = {}  # Store ground truth paths
        
        # Coverage map: track which areas have been visited by robots
        self.coverage_grid_resolution = 0.5  # Grid cell size (smaller = finer resolution)
        self.coverage_grid = None  # Will be initialized as 2D array
        self.coverage_grid_bounds = None  # (x_min, x_max, y_min, y_max)
        self.show_coverage_map = True  # Toggle for coverage visualization
        self.coverage_decay_rate = 0.0  # How fast coverage fades (0 = no decay, 1 = instant)
        
        # Multiple landmarks (list of positions)
        self.landmarks = [
            np.array([0.0, 0.0]),  # Default landmark at origin
            np.array([8.0, 0.0]),  # Additional landmarks
            np.array([-8.0, 0.0]),
            np.array([0.0, 8.0]),
            np.array([0.0, -8.0])
        ]
        self.num_landmarks = 1  # Start with 1 landmark
        
        self.init_robots()
        self.init_obstacles()
        # self.init_maze()  # Initialize maze walls if enabled - DISABLED
        
        # Robot sensor view visualization (DISABLED - not working properly)
        self.show_robot_sensor_view = False  # Toggle for sensor view panel
        self.selected_robot_for_view = 0  # Which robot to show sensor view for
        
        # 2. UI Layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # Left panel - Controls
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 0)
        
        # Right panel - Visualization with Tabs
        self.tab_widget = QTabWidget()
        
        # Tab 1: Main Simulation (Simulation + GPB + GA Landmark Selection)
        sim_tab = QWidget()
        sim_layout = QVBoxLayout(sim_tab)
        
        # FPS label
        self.fps_label = QLabel("0 FPS")
        self.fps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sim_layout.addWidget(self.fps_label)
        
        # Main simulation plot - will be initialized in update_plot based on sensor view toggle
        # Initialize with default layout (no sensor view)
        self.fig = plt.figure(figsize=(14, 10))
        gs = self.fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3, 
                                   left=0.05, right=0.70, top=0.95, bottom=0.05)
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.ax_gpb = self.fig.add_subplot(gs[1, 0])
        self.ax_ga = None  # Will be created when GA is enabled
        self.ax_sensor = None  # Will be created when sensor view is enabled
        
        self.canvas = FigureCanvas(self.fig)
        sim_layout.addWidget(self.canvas)
        
        self.tab_widget.addTab(sim_tab, "Simulation")
        
        # Tab 2: Federated Learning
        fl_tab = QWidget()
        fl_layout = QVBoxLayout(fl_tab)
        
        # Title for federated learning tab
        fl_title = QLabel("Federated Learning (GA-Optimized)")
        fl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fl_title.setStyleSheet("font-size: 14pt; font-weight: bold; padding: 10px;")
        fl_layout.addWidget(fl_title)
        
        # Federated learning plot - use subplots for better organization
        self.fig_fl = plt.figure(figsize=(16, 10))
        # Create a 2x2 grid of subplots for different metrics
        gs_fl = self.fig_fl.add_gridspec(2, 2, hspace=0.35, wspace=0.3, 
                                        left=0.08, right=0.95, top=0.93, bottom=0.08)
        
        # Subplot 1: GA Fitness Evolution (top left)
        self.ax_fl_fitness = self.fig_fl.add_subplot(gs_fl[0, 0])
        
        # Subplot 2: Aggregation Weights over time (top right)
        self.ax_fl_weights = self.fig_fl.add_subplot(gs_fl[0, 1])
        
        # Subplot 3: Metrics (Diversity, Fairness, Quality) (bottom left)
        self.ax_fl_metrics = self.fig_fl.add_subplot(gs_fl[1, 0])
        
        # Subplot 4: Client Selection over rounds (bottom right)
        self.ax_fl_clients = self.fig_fl.add_subplot(gs_fl[1, 1])
        
        self.canvas_fl = FigureCanvas(self.fig_fl)
        fl_layout.addWidget(self.canvas_fl)
        
        # Status label for federated learning
        self.fl_status_label = QLabel("Federated Learning: Disabled")
        self.fl_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fl_status_label.setStyleSheet("font-size: 10pt; padding: 5px;")
        fl_layout.addWidget(self.fl_status_label)
        
        self.tab_widget.addTab(fl_tab, "Federated Learning")
        
        main_layout.addWidget(self.tab_widget, 1)
        
        self.setCentralWidget(main_widget)
        
        # Timer for auto-update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(16)  # ~60 FPS
        
        # Flag to track if simulation is shutting down
        self.is_shutting_down = False
        
        # Initial status update
        self.update_status_display()
        self.update_plot()
        
        print("\n" + "="*70)
        print("Robot Web: Distributed GBP Localization")
        print("="*70)
        print("TECHNICAL GOALS:")
        print("1. Distributed MAP Inference on Global Factor Graph")
        print("   - Each robot holds its own fragment (odometry + observations)")
        print("   - GBP performs local matrix math, sends small messages")
        print("   - Converges to same accuracy as centralized solution")
        print("")
        print("2. Key Problems Solved:")
        print("   - Faulty Sensors: Robust Factors ignore outliers")
        print("   - Unreliable Communication: Async, packet loss tolerance")
        print("   - Dynamic Environments: Robots can join/leave anytime")
        print("="*70)
        print(f"Initialized {self.num_robots} differential drive robots, {self.num_landmarks} landmarks")
        print(f"Features: Odometry={self.use_odometry}, Robust={self.is_robust}, Async={self.async_communication}")
        print("Press Ctrl+C or close window to stop simulation safely")
        print("="*70 + "\n")
    
    def init_robots(self):
        """Initialize robots: differential robots (Roomba) and drones for federated learning with non-IID data."""
        self.robots = []
        self.robot_paths = {}
        self.robot_gt_paths = {}
        
        # Federated learning setup (disabled by default - can be enabled via UI)
        self.use_federated_learning = False  # Disabled by default to avoid performance overhead
        self.federated_round = 0
        self.global_model = None  # Global model (e.g., position estimation model)
        self.robot_updates = []  # Store updates from each robot
        self.federated_ga = None  # GA optimizer for federated learning
        
        # Initialize coverage grid
        grid_margin = 2.0  # Extra margin around boundary
        self.coverage_grid_bounds = (
            self.circle_center[0] - self.boundary_radius - grid_margin,
            self.circle_center[0] + self.boundary_radius + grid_margin,
            self.circle_center[1] - self.boundary_radius - grid_margin,
            self.circle_center[1] + self.boundary_radius + grid_margin
        )
        x_size = int((self.coverage_grid_bounds[1] - self.coverage_grid_bounds[0]) / self.coverage_grid_resolution)
        y_size = int((self.coverage_grid_bounds[3] - self.coverage_grid_bounds[2]) / self.coverage_grid_resolution)
        self.coverage_grid = np.zeros((y_size, x_size), dtype=np.float32)  # y, x order for imshow
        # Clear error history when robots are reinitialized (new robots = new IDs)
        self.gpb_robot_errors = {}
        self.gpb_error_history = []
        self.gpb_convergence_history = []
        
        # Initialize robots: mix of differential (Roomba) and drone types for non-IID federated learning
        # Different robot types have different observation models (non-IID data)
        robot_types = []
        for i in range(self.num_robots):
            # Alternate between differential (Roomba) and drone types
            if i % 2 == 0:
                robot_types.append('differential')  # Roomba - ground view
            else:
                robot_types.append('drone')  # Drone - aerial view
        
        for i in range(self.num_robots):
            angle = 2 * np.pi * i / self.num_robots
            # Initial position on circle - use larger radius for better spacing
            gt_pos = self.circle_center + self.circle_radius * np.array([np.cos(angle), np.sin(angle)])
            # Smaller noise for cleaner start
            initial_estimate = gt_pos + np.random.normal(0, 0.2, 2)
            
            robot_type = robot_types[i]
            if robot_type == 'differential':
                robot = RobotAgent(f"Roomba{i+1}", initial_estimate, robot_type='differential')
            else:  # drone
                robot = RobotAgent(f"Drone{i+1}", initial_estimate, robot_type='differential')  # Use differential for now, will add drone kinematics
                # Mark as drone type for federated learning
                robot.robot_type = 'drone'
                robot.observation_model = 'aerial'  # Different observation model
            robot.gt_pos = gt_pos.copy()  # Store ground truth (actual physical position)
            # Initialize precision matrix (inverse covariance) with low precision (high uncertainty)
            # This will be updated by GBP as observations come in
            if not hasattr(robot, 'Lambda') or robot.Lambda is None:
                robot.Lambda = np.eye(2) * 0.1  # Low precision = high initial uncertainty
            robot.gt_angle = angle  # Ground truth angle
            # Alternate direction: even robots clockwise, odd robots counter-clockwise
            robot.direction = 1 if i % 2 == 0 else -1  # 1 for clockwise, -1 for counter-clockwise
            robot.angle = angle + np.pi / 2 * robot.direction  # Face tangent to circle (for visualization)
            robot.estimated_angle = angle  # Initialize estimated angle for motion model
            robot.expected_pos = gt_pos.copy()  # Initialize expected position
            robot.last_position = initial_estimate.copy()  # For odometry
            robot.is_active = True  # Robot is active
            robot.message_queue = []  # For async communication
            robot.last_message_time = {}  # Track message timestamps
            robot.radius = 0.5  # Robot radius for RVO
            robot.preferred_vel = np.array([0.0, 0.0])  # Preferred velocity for RVO
            # Each robot has slightly different circular path to avoid clustering
            robot.path_radius = self.circle_radius + (i % 2) * 1.5  # Alternate radii
            robot.path_center_offset = np.array([
                np.cos(angle) * 1.0, np.sin(angle) * 1.0
            ]) * 0.3  # Slight offset to spread paths
            self.robots.append(robot)
            self.robot_paths[robot.id] = [initial_estimate.copy()]
            self.robot_gt_paths[robot.id] = [gt_pos.copy()]
            
            # Initialize SLAM observations for this robot
            if robot.id not in self.slam_observations:
                self.slam_observations[robot.id] = []
            
            # Mark initial position in coverage grid
            self._update_coverage_grid(gt_pos, robot.radius)
        
        # Initialize federated learning GA optimizer (only if enabled)
        # Note: Federated learning is disabled by default to avoid performance overhead
        # Users can enable it via the UI checkbox if needed
        if self.use_federated_learning:
            robot_types_list = [r.robot_type for r in self.robots]
            self.federated_ga = FederatedLearningGA(
                population_size=20,
                mutation_rate=0.15,
                crossover_rate=0.7,
                num_robot_types=len(set(robot_types_list)),
                max_clients_per_round=None
            )
            self.federated_ga.initialize_population(len(self.robots), robot_types_list)
        else:
            self.federated_ga = None  # Ensure it's None when disabled
        
    
    def init_obstacles(self):
        """Initialize obstacles placed to avoid robot starting positions."""
        self.obstacles = []
        self.obstacle_velocities = []
        
        # Get robot starting positions to avoid placing obstacles too close
        robot_positions = []
        if len(self.robots) > 0:
            for robot in self.robots:
                robot_positions.append(robot.mu.copy())
        
        for i in range(self.num_obstacles):
            # Try to place obstacle away from robots
            max_attempts = 20
            pos = None
            for attempt in range(max_attempts):
                angle = np.random.uniform(0, 2 * np.pi)
                # Place obstacles in middle region, not too close to center or boundary
                radius = np.random.uniform(self.circle_radius * 0.7, self.boundary_radius * 0.7)
                candidate_pos = self.circle_center + radius * np.array([np.cos(angle), np.sin(angle)])
                
                # Check distance to all robots
                min_dist_to_robot = min([np.linalg.norm(candidate_pos - rp) for rp in robot_positions])
                if min_dist_to_robot > 3.0:  # At least 3 units from any robot
                    pos = candidate_pos
                    break
            
            # If we couldn't find a good spot, use the last attempt
            if pos is None:
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(self.circle_radius * 0.7, self.boundary_radius * 0.7)
                pos = self.circle_center + radius * np.array([np.cos(angle), np.sin(angle)])
            
            self.obstacles.append(pos.copy())
            
            # Random velocity for moving obstacles (slower for cleaner demo)
            if self.obstacle_moving:
                vel_angle = np.random.uniform(0, 2 * np.pi)
                speed = np.random.uniform(0.15, 0.3)  # Slower movement
                vel = speed * np.array([np.cos(vel_angle), np.sin(vel_angle)])
            else:
                vel = np.array([0.0, 0.0])
            
            self.obstacle_velocities.append(vel)
    
    def _mahalanobis_distance(self, mu, gt_pos, Lambda):
        """
        Compute Mahalanobis distance between estimated position (mu) and ground truth (gt_pos).
        
        Mahalanobis distance: d_M = sqrt((x - μ)^T Σ^(-1) (x - μ))
        Where Σ^(-1) = Lambda (precision matrix)
        
        This accounts for uncertainty - errors in directions with high uncertainty are less penalized.
        More statistically meaningful than Euclidean distance for probabilistic localization.
        
        Args:
            mu: Estimated position [x, y]
            gt_pos: Ground truth position [x, y]
            Lambda: Precision matrix (2x2) - inverse of covariance matrix
        
        Returns:
            Mahalanobis distance (scalar)
        """
        diff = gt_pos - mu
        
        # Handle case where Lambda might be singular or not properly initialized
        try:
            # Mahalanobis distance: sqrt(diff^T * Lambda * diff)
            mahal_dist_sq = diff.T @ Lambda @ diff
            
            # Ensure non-negative (numerical stability)
            mahal_dist_sq = max(0.0, mahal_dist_sq)
            mahal_dist = np.sqrt(mahal_dist_sq)
            
            # Fallback to Euclidean if precision matrix is invalid
            if not np.isfinite(mahal_dist) or mahal_dist < 0:
                return np.linalg.norm(diff)
            
            return mahal_dist
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to Euclidean distance if precision matrix is invalid
            return np.linalg.norm(diff)
    
    def _closest_point_on_segment(self, point, seg_start, seg_end):
        """
        Find the closest point on a line segment to a given point.
        
        Args:
            point: Point [x, y]
            seg_start: Segment start [x, y]
            seg_end: Segment end [x, y]
            
        Returns:
            Closest point on segment [x, y]
        """
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)
        
        if seg_len_sq < 1e-10:  # Degenerate segment (start == end)
            return seg_start.copy()
        
        point_vec = point - seg_start
        t = np.dot(point_vec, seg_vec) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)  # Clamp to segment
        
        return seg_start + t * seg_vec
    
    def _update_coverage_grid(self, robot_pos, robot_radius):
        """Update coverage grid based on robot's ground truth position."""
        if self.coverage_grid is None:
            return
        
        # Convert robot position to grid coordinates
        x_min, x_max, y_min, y_max = self.coverage_grid_bounds
        grid_x = int((robot_pos[0] - x_min) / self.coverage_grid_resolution)
        grid_y = int((robot_pos[1] - y_min) / self.coverage_grid_resolution)
        
        # Mark cells within robot radius as visited
        radius_cells = int(robot_radius / self.coverage_grid_resolution) + 1
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                cell_x = grid_x + dx
                cell_y = grid_y + dy
                
                # Check bounds
                if (0 <= cell_x < self.coverage_grid.shape[1] and 
                    0 <= cell_y < self.coverage_grid.shape[0]):
                    # Check if within circular radius
                    dist = np.sqrt(dx**2 + dy**2) * self.coverage_grid_resolution
                    if dist <= robot_radius:
                        # Mark as visited (increment coverage value)
                        self.coverage_grid[cell_y, cell_x] = min(
                            self.coverage_grid[cell_y, cell_x] + 0.1, 1.0
                        )
        
        # Apply decay if enabled
        if self.coverage_decay_rate > 0:
            self.coverage_grid *= (1.0 - self.coverage_decay_rate)
    
    def _robot_wall_collision(self, robot_pos, wall_start, wall_end, robot_radius):
        """
        Check if robot collides with a wall segment.
        
        Args:
            robot_pos: Robot position [x, y]
            wall_start: Wall segment start [x, y]
            wall_end: Wall segment end [x, y]
            robot_radius: Robot radius
            
        Returns:
            True if collision detected
        """
        closest = self._closest_point_on_segment(robot_pos, wall_start, wall_end)
        dist = np.linalg.norm(robot_pos - closest)
        return dist < robot_radius
    
    # MAZE DISABLED - Commented out for federated learning
    def init_maze_disabled(self):
        """Initialize maze walls for non-convex environment."""
        self.maze_walls = []
        if not self.use_maze:
            return
        
        # Create a simple, visible maze pattern with prominent walls
        # Walls are line segments: [(start_pos, end_pos), ...]
        # Make walls clearly visible within the boundary
        
        # Use a simpler, more reliable approach
        # Create walls that are guaranteed to be visible
        
        # Always create at least a cross pattern for visibility
        # Make sure walls are clearly within the visible area
        # Horizontal wall through center (guaranteed to be visible)
        wall_length = self.boundary_radius * 0.8  # 80% of boundary radius
        self.maze_walls.append((
            np.array([-wall_length, 0.0]), 
            np.array([wall_length, 0.0])
        ))
        
        # Vertical wall through center (guaranteed to be visible)
        self.maze_walls.append((
            np.array([0.0, -wall_length]), 
            np.array([0.0, wall_length])
        ))
        
        # Add additional walls based on complexity
        num_extra_walls = max(0, self.maze_complexity - 2)
        
        for i in range(num_extra_walls):
            # Alternate between horizontal and vertical
            if i % 2 == 0:
                # Horizontal wall
                y_pos = -self.boundary_radius * 0.5 + (i // 2 + 1) * (self.boundary_radius * 0.3 / (num_extra_walls // 2 + 1))
                # Create wall with gap in middle
                gap = self.boundary_radius * 0.2
                self.maze_walls.append((
                    np.array([-self.boundary_radius * 0.7, y_pos]),
                    np.array([-gap, y_pos])
                ))
                self.maze_walls.append((
                    np.array([gap, y_pos]),
                    np.array([self.boundary_radius * 0.7, y_pos])
                ))
            else:
                # Vertical wall
                x_pos = -self.boundary_radius * 0.5 + ((i-1) // 2 + 1) * (self.boundary_radius * 0.3 / (num_extra_walls // 2 + 1))
                # Create wall with gap in middle
                gap = self.boundary_radius * 0.2
                self.maze_walls.append((
                    np.array([x_pos, -self.boundary_radius * 0.7]),
                    np.array([x_pos, -gap])
                ))
                self.maze_walls.append((
                    np.array([x_pos, gap]),
                    np.array([x_pos, self.boundary_radius * 0.7])
                ))
        
        # Debug output
        if self.debug_console and len(self.maze_walls) > 0:
            print(f"[Maze] Initialized {len(self.maze_walls)} wall segments (complexity={self.maze_complexity})")
    
    def _collect_federated_updates(self):
        """Collect updates from all robots for federated learning."""
        self.robot_updates = []
        active_count = 0
        for robot in self.robots:
            if not robot.is_active:
                continue
            
            active_count += 1
            
            # Simulate robot's local training update
            # In real federated learning, this would be model weights/gradients
            # Use Mahalanobis distance for position error (accounts for uncertainty)
            position_error = self._mahalanobis_distance(robot.mu, robot.gt_pos, robot.Lambda)
            
            # Different robot types have different observation models (non-IID)
            # Check robot type (handle both 'differential' and 'drone' types)
            robot_type_str = getattr(robot, 'robot_type', 'differential')
            if robot_type_str == 'differential' or 'Roomba' in robot.name:  # Roomba - ground view
                # Ground robots see obstacles and landmarks differently
                observation_quality = 0.8  # Good for ground features
                data_size = 100  # More data from ground exploration
            elif robot_type_str == 'drone' or 'Drone' in robot.name:  # Drone - aerial view
                # Aerial robots see topology differently
                observation_quality = 0.6  # Different perspective
                data_size = 50  # Less data, different distribution
            else:
                observation_quality = 0.7
                data_size = 75
            
            # Create update (simplified: using position estimate as "model")
            # Use a 2D vector for weights (position estimate)
            update = {
                'weights': robot.mu.copy(),  # In real FL, this would be model parameters
                'type': robot_type_str,
                'data_size': data_size,
                'loss': max(position_error, 0.01),  # Lower is better, ensure > 0
                'observation_quality': observation_quality
            }
            self.robot_updates.append(update)
        
        if self.debug_console and len(self.robot_updates) == 0:
            print(f"[Federated Learning] Warning: No active robots to collect updates from (active: {active_count}/{len(self.robots)})")
    
    def _apply_global_update(self, global_update: np.ndarray, config: Dict):
        """Apply aggregated global update to robots."""
        # Simplified: adjust robot position estimates based on global consensus
        # In real federated learning, this would update model parameters
        selected_indices = np.where(config['client_selection'])[0]
        
        for idx in selected_indices:
            if idx < len(self.robots):
                robot = self.robots[idx]
                if robot.is_active:
                    # Blend global update with local estimate (weighted average)
                    alpha = 0.1  # Learning rate
                    robot.mu = (1 - alpha) * robot.mu + alpha * global_update
    
    def create_control_panel(self):
        """Create the left control panel with settings."""
        panel = QWidget()
        panel.setMaximumWidth(320)  # Slightly wider for better layout
        panel.setStyleSheet("background-color: #2b2b2b; color: white;")
        
        # Main layout with scroll area
        main_layout = QVBoxLayout(panel)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create scroll area to prevent squeezing
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Content widget for scroll area
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Settings Group - Collapsible
        settings_group = QGroupBox("▼ Settings")
        settings_group.setCheckable(True)
        settings_group.setChecked(True)  # Start expanded
        settings_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(8)
        
        # Sub-group: Robot Configuration
        robot_config_group = QGroupBox("Robot Config")
        robot_config_group.setCheckable(True)
        robot_config_group.setChecked(True)
        robot_config_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                border: 1px solid #555;
                border-radius: 3px;
                margin-top: 5px;
                padding-top: 5px;
                font-size: 9pt;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 5px;
                padding: 0 3px;
            }
        """)
        robot_config_layout = QVBoxLayout()
        robot_config_layout.setSpacing(5)
        
        robots_layout = QHBoxLayout()
        robots_layout.addWidget(QLabel("Num Robots:"))
        self.num_robots_spin = QSpinBox()
        self.num_robots_spin.setRange(2, 10)
        self.num_robots_spin.setValue(self.num_robots)
        self.num_robots_spin.valueChanged.connect(self.on_num_robots_changed)
        robots_layout.addWidget(self.num_robots_spin)
        robot_config_layout.addLayout(robots_layout)
        
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Circle Radius:"))
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(1.0, 20.0)
        self.radius_spin.setSingleStep(0.5)
        self.radius_spin.setValue(self.circle_radius)
        self.radius_spin.valueChanged.connect(self.on_radius_changed)
        radius_layout.addWidget(self.radius_spin)
        robot_config_layout.addLayout(radius_layout)
        
        # RVO Collision Avoidance checkbox
        self.use_rvo_cb = QCheckBox("Enable RVO Collision Avoidance")
        self.use_rvo_cb.setChecked(self.use_rvo)
        self.use_rvo_cb.stateChanged.connect(self.on_rvo_changed)
        robot_config_layout.addWidget(self.use_rvo_cb)
        
        robot_config_group.setLayout(robot_config_layout)
        def update_robot_config_arrow(checked):
            robot_config_group.setTitle("▶ Robot Config" if not checked else "Robot Config")
        robot_config_group.toggled.connect(update_robot_config_arrow)
        settings_layout.addWidget(robot_config_group)
        
        # Sub-group: Sensor & Communication
        sensor_comm_group = QGroupBox("Sensor & Comm")
        sensor_comm_group.setCheckable(True)
        sensor_comm_group.setChecked(True)
        sensor_comm_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                border: 1px solid #555;
                border-radius: 3px;
                margin-top: 5px;
                padding-top: 5px;
                font-size: 9pt;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 5px;
                padding: 0 3px;
            }
        """)
        sensor_comm_layout = QVBoxLayout()
        sensor_comm_layout.setSpacing(5)
        
        sensor_layout = QHBoxLayout()
        sensor_layout.addWidget(QLabel("Sensor Range:"))
        self.sensor_spin = QDoubleSpinBox()
        self.sensor_spin.setRange(1.0, 30.0)
        self.sensor_spin.setSingleStep(0.5)
        self.sensor_spin.setValue(self.sensor_range)
        self.sensor_spin.valueChanged.connect(self.on_sensor_range_changed)
        sensor_layout.addWidget(self.sensor_spin)
        sensor_comm_layout.addLayout(sensor_layout)
        
        landmarks_layout = QHBoxLayout()
        landmarks_layout.addWidget(QLabel("Num Landmarks:"))
        self.num_landmarks_spin = QSpinBox()
        self.num_landmarks_spin.setRange(0, 10)
        self.num_landmarks_spin.setValue(self.num_landmarks)
        self.num_landmarks_spin.valueChanged.connect(self.on_num_landmarks_changed)
        landmarks_layout.addWidget(self.num_landmarks_spin)
        sensor_comm_layout.addLayout(landmarks_layout)
        
        self.landmark_only_cb = QCheckBox("Use Landmark Only")
        self.landmark_only_cb.setChecked(self.use_landmark_only)
        self.landmark_only_cb.stateChanged.connect(self.on_landmark_only_changed)
        sensor_comm_layout.addWidget(self.landmark_only_cb)
        
        self.async_cb = QCheckBox("Async Communication")
        self.async_cb.setChecked(self.async_communication)
        self.async_cb.stateChanged.connect(self.on_async_changed)
        sensor_comm_layout.addWidget(self.async_cb)
        
        drop_layout = QHBoxLayout()
        drop_layout.addWidget(QLabel("Packet Loss:"))
        self.drop_spin = QDoubleSpinBox()
        self.drop_spin.setRange(0.0, 50.0)
        self.drop_spin.setSingleStep(5.0)
        self.drop_spin.setValue(self.communication_drop_rate * 100)
        self.drop_spin.setSuffix("%")
        self.drop_spin.valueChanged.connect(self.on_drop_rate_changed)
        drop_layout.addWidget(self.drop_spin)
        sensor_comm_layout.addLayout(drop_layout)
        
        sensor_comm_group.setLayout(sensor_comm_layout)
        def update_sensor_comm_arrow(checked):
            sensor_comm_group.setTitle("▶ Sensor & Comm" if not checked else "Sensor & Comm")
        sensor_comm_group.toggled.connect(update_sensor_comm_arrow)
        settings_layout.addWidget(sensor_comm_group)
        
        # Sub-group: GBP Parameters
        gbp_params_group = QGroupBox("GBP Parameters")
        gbp_params_group.setCheckable(True)
        gbp_params_group.setChecked(True)  # Enabled by default so controls are selectable
        gbp_params_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                border: 1px solid #555;
                border-radius: 3px;
                margin-top: 5px;
                padding-top: 5px;
                font-size: 9pt;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 5px;
                padding: 0 3px;
            }
        """)
        gbp_params_layout = QVBoxLayout()
        gbp_params_layout.setSpacing(5)
        
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Iter Before Mot:"))
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, 20)
        self.iter_spin.setValue(self.iter_before_motion)
        self.iter_spin.valueChanged.connect(self.on_iter_changed)
        iter_layout.addWidget(self.iter_spin)
        gbp_params_layout.addLayout(iter_layout)
        
        lin_layout = QHBoxLayout()
        lin_layout.addWidget(QLabel("Linearize Every:"))
        self.lin_spin = QSpinBox()
        self.lin_spin.setRange(1, 10)
        self.lin_spin.setValue(self.linearize_every)
        self.lin_spin.valueChanged.connect(self.on_linearize_changed)
        lin_layout.addWidget(self.lin_spin)
        gbp_params_layout.addLayout(lin_layout)
        
        damp_layout = QHBoxLayout()
        damp_layout.addWidget(QLabel("Damping:"))
        self.damp_spin = QDoubleSpinBox()
        self.damp_spin.setRange(0.0, 1.0)
        self.damp_spin.setSingleStep(0.01)
        self.damp_spin.setValue(self.damping)
        self.damp_spin.valueChanged.connect(self.on_damping_changed)
        damp_layout.addWidget(self.damp_spin)
        gbp_params_layout.addLayout(damp_layout)
        
        noise_layout = QHBoxLayout()
        noise_layout.addWidget(QLabel("Noise Fraction:"))
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 0.1)
        self.noise_spin.setSingleStep(0.001)
        self.noise_spin.setValue(self.noise_std[0])
        self.noise_spin.valueChanged.connect(self.on_noise_changed)
        noise_layout.addWidget(self.noise_spin)
        gbp_params_layout.addLayout(noise_layout)
        
        self.robust_cb = QCheckBox("Is Robust")
        self.robust_cb.setChecked(self.is_robust)
        self.robust_cb.stateChanged.connect(self.on_robust_changed)
        gbp_params_layout.addWidget(self.robust_cb)
        
        self.odometry_cb = QCheckBox("Use Odometry Factors")
        self.odometry_cb.setChecked(self.use_odometry)
        self.odometry_cb.stateChanged.connect(self.on_odometry_changed)
        gbp_params_layout.addWidget(self.odometry_cb)
        
        gbp_params_group.setLayout(gbp_params_layout)
        def update_gbp_params_arrow(checked):
            gbp_params_group.setTitle("▶ GBP Parameters" if not checked else "GBP Parameters")
        gbp_params_group.toggled.connect(update_gbp_params_arrow)
        settings_layout.addWidget(gbp_params_group)
        
        # Keep dynamic join/leave in main settings
        self.dynamic_cb = QCheckBox("Allow Dynamic Join/Leave")
        self.dynamic_cb.setChecked(self.allow_dynamic_join_leave)
        self.dynamic_cb.stateChanged.connect(self.on_dynamic_changed)
        settings_layout.addWidget(self.dynamic_cb)
        
        # Sub-group: Maze & GA Settings
        maze_ga_group = QGroupBox("Maze & GA Selection")
        maze_ga_group.setCheckable(True)
        maze_ga_group.setChecked(True)  # Enabled by default so controls are selectable
        maze_ga_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                border: 1px solid #555;
                border-radius: 3px;
                margin-top: 5px;
                padding-top: 5px;
                font-size: 9pt;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 5px;
                padding: 0 3px;
            }
        """)
        maze_ga_layout = QVBoxLayout()
        maze_ga_layout.setSpacing(5)
        
        # MAZE DISABLED - Commented out for federated learning
        # self.use_maze_cb = QCheckBox("Enable Maze Environment")
        # self.use_maze_cb.setChecked(self.use_maze)
        # self.use_maze_cb.stateChanged.connect(self.on_maze_changed)
        # maze_ga_layout.addWidget(self.use_maze_cb)
        
        # Federated Learning toggle
        self.use_federated_cb = QCheckBox("Enable-FL(GA)")
        self.use_federated_cb.setChecked(self.use_federated_learning)
        self.use_federated_cb.stateChanged.connect(self.on_federated_changed)
        maze_ga_layout.addWidget(self.use_federated_cb)
        
        maze_complexity_layout = QHBoxLayout()
        maze_complexity_layout.addWidget(QLabel("Maze Complexity:"))
        self.maze_complexity_spin = QSpinBox()
        self.maze_complexity_spin.setRange(3, 15)
        self.maze_complexity_spin.setValue(self.maze_complexity)
        self.maze_complexity_spin.valueChanged.connect(self.on_maze_complexity_changed)
        maze_complexity_layout.addWidget(self.maze_complexity_spin)
        maze_ga_layout.addLayout(maze_complexity_layout)
        
        self.use_ga_cb = QCheckBox("Use GA Landmark Selection")
        self.use_ga_cb.setChecked(self.use_ga_landmark_selection)
        self.use_ga_cb.stateChanged.connect(self.on_ga_selection_changed)
        maze_ga_layout.addWidget(self.use_ga_cb)
        
        ga_max_layout = QHBoxLayout()
        ga_max_layout.addWidget(QLabel("GA Max Selected:"))
        self.ga_max_spin = QSpinBox()
        self.ga_max_spin.setRange(1, 5)
        self.ga_max_spin.setValue(self.ga_max_selected)
        self.ga_max_spin.valueChanged.connect(self.on_ga_max_changed)
        ga_max_layout.addWidget(self.ga_max_spin)
        maze_ga_layout.addLayout(ga_max_layout)
        
        ga_gen_layout = QHBoxLayout()
        ga_gen_layout.addWidget(QLabel("GA Generations:"))
        self.ga_gen_spin = QSpinBox()
        self.ga_gen_spin.setRange(5, 30)
        self.ga_gen_spin.setValue(self.ga_generations)
        self.ga_gen_spin.valueChanged.connect(self.on_ga_gen_changed)
        ga_gen_layout.addWidget(self.ga_gen_spin)
        maze_ga_layout.addLayout(ga_gen_layout)
        
        maze_ga_group.setLayout(maze_ga_layout)
        def update_maze_ga_arrow(checked):
            maze_ga_group.setTitle("▶ Maze & GA Selection" if not checked else "Maze & GA Selection")
        maze_ga_group.toggled.connect(update_maze_ga_arrow)
        settings_layout.addWidget(maze_ga_group)
        
        settings_group.setLayout(settings_layout)
        # Connect toggle to update arrow indicator
        def update_settings_arrow(checked):
            settings_group.setTitle("▶ Settings" if not checked else "▼ Settings")
        settings_group.toggled.connect(update_settings_arrow)
        layout.addWidget(settings_group)
        
        # Visualization Group - Collapsible
        viz_group = QGroupBox("▼ Visualisation")
        viz_group.setCheckable(True)
        viz_group.setChecked(True)  # Start expanded
        viz_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        viz_layout = QVBoxLayout()
        
        self.show_factors_cb = QCheckBox("Show Inter-robot Communication")
        self.show_factors_cb.setChecked(self.show_factors)
        self.show_factors_cb.stateChanged.connect(self.on_show_factors_changed)
        viz_layout.addWidget(self.show_factors_cb)
        
        self.show_landmark_cb = QCheckBox("Show Only Landmark Factors")
        self.show_landmark_cb.setChecked(self.show_only_landmark_factors)
        self.show_landmark_cb.stateChanged.connect(self.on_show_landmark_changed)
        viz_layout.addWidget(self.show_landmark_cb)
        
        self.show_path_cb = QCheckBox("Show Path")
        self.show_path_cb.setChecked(self.show_path)
        self.show_path_cb.stateChanged.connect(self.on_show_path_changed)
        viz_layout.addWidget(self.show_path_cb)
        
        self.show_coverage_cb = QCheckBox("Show Coverage Map")
        self.show_coverage_cb.setChecked(self.show_coverage_map)
        self.show_coverage_cb.stateChanged.connect(self.on_show_coverage_changed)
        viz_layout.addWidget(self.show_coverage_cb)
        
        self.show_slam_cb = QCheckBox("Show SLAM Factor Graph")
        self.show_slam_cb.setChecked(self.show_slam_visualization)
        self.show_slam_cb.stateChanged.connect(self.on_show_slam_changed)
        viz_layout.addWidget(self.show_slam_cb)
        
        self.show_samples_cb = QCheckBox("Show Samples")
        self.show_samples_cb.setChecked(self.show_samples)
        self.show_samples_cb.stateChanged.connect(self.on_show_samples_changed)
        viz_layout.addWidget(self.show_samples_cb)
        
        # Sensor view disabled - not working properly
        # self.show_sensor_view_cb = QCheckBox("Show Robot Sensor View")
        # self.show_sensor_view_cb.setChecked(self.show_robot_sensor_view)
        # self.show_sensor_view_cb.stateChanged.connect(self.on_show_sensor_view_changed)
        # viz_layout.addWidget(self.show_sensor_view_cb)
        # 
        # sensor_robot_layout = QHBoxLayout()
        # sensor_robot_layout.addWidget(QLabel("Robot ID:"))
        # self.sensor_robot_spin = QSpinBox()
        # self.sensor_robot_spin.setRange(0, max(0, self.num_robots - 1))
        # self.sensor_robot_spin.setValue(self.selected_robot_for_view)
        # self.sensor_robot_spin.valueChanged.connect(self.on_sensor_robot_changed)
        # sensor_robot_layout.addWidget(self.sensor_robot_spin)
        # viz_layout.addLayout(sensor_robot_layout)
        
        self.follow_robot_cb = QCheckBox("Follow Robot")
        self.follow_robot_cb.setChecked(self.follow_robot)
        self.follow_robot_cb.stateChanged.connect(self.on_follow_robot_changed)
        viz_layout.addWidget(self.follow_robot_cb)
        
        robot_id_layout = QHBoxLayout()
        robot_id_layout.addWidget(QLabel("Robot ID to fol:"))
        self.robot_id_spin = QSpinBox()
        self.robot_id_spin.setRange(0, self.num_robots - 1)
        self.robot_id_spin.setValue(self.robot_id_to_follow)
        self.robot_id_spin.valueChanged.connect(self.on_robot_id_changed)
        robot_id_layout.addWidget(self.robot_id_spin)
        viz_layout.addLayout(robot_id_layout)
        
        self.step_by_step_cb = QCheckBox("Step by Step")
        self.step_by_step_cb.setChecked(self.step_by_step)
        self.step_by_step_cb.stateChanged.connect(self.on_step_by_step_changed)
        viz_layout.addWidget(self.step_by_step_cb)
        
        self.run_cb = QCheckBox("Run")
        self.run_cb.setChecked(self.running)
        self.run_cb.stateChanged.connect(self.on_run_changed)
        viz_layout.addWidget(self.run_cb)
        
        viz_group.setLayout(viz_layout)
        # Connect toggle to update arrow indicator
        def update_viz_arrow(checked):
            viz_group.setTitle("▶ Visualisation" if not checked else "▼ Visualisation")
        viz_group.toggled.connect(update_viz_arrow)
        layout.addWidget(viz_group)
        
        # Debug/Status Group - Collapsible
        debug_group = QGroupBox("▼ Status")
        debug_group.setCheckable(True)
        debug_group.setChecked(True)  # Start expanded
        debug_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        debug_layout = QVBoxLayout()
        
        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("background-color: #1e1e1e; padding: 5px; border-radius: 3px;")
        debug_layout.addWidget(self.status_label)
        
        self.messages_label = QLabel("Messages: 0")
        debug_layout.addWidget(self.messages_label)
        
        self.iterations_label = QLabel("Iterations: 0")
        debug_layout.addWidget(self.iterations_label)
        
        self.robots_status_label = QLabel("Robots: 0 active")
        debug_layout.addWidget(self.robots_status_label)
        
        debug_group.setLayout(debug_layout)
        # Connect toggle to update arrow indicator
        def update_debug_arrow(checked):
            debug_group.setTitle("▶ Status" if not checked else "▼ Status")
        debug_group.toggled.connect(update_debug_arrow)
        layout.addWidget(debug_group)
        
        layout.addStretch()
        
        # Set content widget to scroll area
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        
        return panel
    
    def on_num_robots_changed(self, value):
        self.num_robots = value
        self.robot_id_spin.setRange(0, max(0, self.num_robots - 1))
        self.init_robots()
    
    def on_num_landmarks_changed(self, value):
        self.num_landmarks = value
        # Ensure we have enough landmarks in the list
        while len(self.landmarks) < value:
            # Add new landmarks in a circle pattern
            angle = 2 * np.pi * len(self.landmarks) / max(value, 1)
            new_landmark = self.circle_center + (self.circle_radius * 1.5) * np.array([
                np.cos(angle), np.sin(angle)
            ])
            self.landmarks.append(new_landmark.copy())
    
    def on_radius_changed(self, value):
        self.circle_radius = value
    
    def on_rvo_changed(self, state):
        """Handle RVO enable/disable checkbox change."""
        checked_value = 2  # Qt.CheckState.Checked
        self.use_rvo = (int(state) == checked_value)
        if self.debug_console:
            print(f"[RVO] {'Enabled' if self.use_rvo else 'Disabled'}")
        self.init_robots()
    
    def on_landmark_only_changed(self, state):
        self.use_landmark_only = (state == Qt.CheckState.Checked)
    
    def on_iter_changed(self, value):
        self.iter_before_motion = value
    
    def on_linearize_changed(self, value):
        self.linearize_every = value
    
    def on_sensor_range_changed(self, value):
        self.sensor_range = value
    
    def on_damping_changed(self, value):
        self.damping = value
    
    def on_noise_changed(self, value):
        self.noise_std = [value, value * 0.5]
    
    def on_robust_changed(self, state):
        self.is_robust = (state == Qt.CheckState.Checked)
    
    def on_odometry_changed(self, state):
        self.use_odometry = (state == Qt.CheckState.Checked)
    
    def on_async_changed(self, state):
        self.async_communication = (state == Qt.CheckState.Checked)
    
    def on_drop_rate_changed(self, value):
        self.communication_drop_rate = value / 100.0  # Convert percentage to fraction
    
    def on_dynamic_changed(self, state):
        self.allow_dynamic_join_leave = (state == Qt.CheckState.Checked)
    
    def on_odometry_changed(self, state):
        self.use_odometry = (state == Qt.CheckState.Checked)
    
    def on_async_changed(self, state):
        self.async_communication = (state == Qt.CheckState.Checked)
    
    def on_drop_rate_changed(self, value):
        self.communication_drop_rate = value / 100.0  # Convert percentage to fraction
    
    def on_dynamic_changed(self, state):
        self.allow_dynamic_join_leave = (state == Qt.CheckState.Checked)
    
    # MAZE DISABLED
    # def on_maze_changed(self, state):
    #     # Fix: stateChanged signal passes an integer (0=Unchecked, 2=Checked)
    #     # Use explicit integer comparison for reliability
    #     checked_value = 2  # Qt.CheckState.Checked
    #     self.use_maze = (int(state) == checked_value)
    #     self.init_maze()
    #     # Force immediate update to show walls
    #     if self.debug_console:
    #         if self.use_maze:
    #             print(f"[Maze] Enabled: {len(self.maze_walls)} walls created")
    #         else:
    #             print(f"[Maze] Disabled")
    
    def on_federated_changed(self, state):
        self.use_federated_learning = (int(state) == 2)
        if self.use_federated_learning and self.federated_ga is None:
            # Initialize federated learning GA optimizer when enabled
            robot_types_list = [r.robot_type for r in self.robots]
            self.federated_ga = FederatedLearningGA(
                population_size=20,
                mutation_rate=0.15,
                crossover_rate=0.7,
                num_robot_types=len(set(robot_types_list)),
                max_clients_per_round=None
            )
            self.federated_ga.initialize_population(len(self.robots), robot_types_list)
            if self.debug_console:
                print(f"[Federated Learning] Enabled and initialized")
        elif not self.use_federated_learning:
            # Clean up when disabled
            self.federated_ga = None
            self.federated_round = 0
            self.federated_fitness_history = []
            self.federated_aggregation_weights_history = []
            self.federated_client_selection_history = []
            self.federated_metrics_history = []
            if self.debug_console:
                print(f"[Federated Learning] Disabled and cleaned up")
    
    def on_maze_complexity_changed(self, value):
        self.maze_complexity = value
        self.init_maze()
    
    def on_ga_selection_changed(self, state):
        self.use_ga_landmark_selection = (state == Qt.CheckState.Checked)
        # Clear selected landmarks when toggling
        if not self.use_ga_landmark_selection:
            self.robot_selected_landmarks = {}
    
    def on_ga_max_changed(self, value):
        self.ga_max_selected = value
    
    def on_ga_gen_changed(self, value):
        self.ga_generations = value
    
    def on_show_factors_changed(self, state):
        self.show_factors = (state == Qt.CheckState.Checked)
    
    def on_show_landmark_changed(self, state):
        self.show_only_landmark_factors = (state == Qt.CheckState.Checked)
    
    def on_show_path_changed(self, state):
        self.show_path = (state == Qt.CheckState.Checked)
    
    def on_show_coverage_changed(self, state):
        self.show_coverage_map = (int(state) == 2)
    
    def on_show_slam_changed(self, state):
        self.show_slam_visualization = (int(state) == 2)
    
    def on_show_samples_changed(self, state):
        self.show_samples = (state == Qt.CheckState.Checked)
    
    # Sensor view disabled - not working properly
    # def on_show_sensor_view_changed(self, state):
    #     self.show_robot_sensor_view = (state == Qt.CheckState.Checked)
    #     # Force immediate figure recreation
    #     if hasattr(self, 'fig') and self.fig is not None:
    #         # Force recreation by setting ax_sensor to None if disabling, or mark for creation if enabling
    #         if not self.show_robot_sensor_view:
    #             self.ax_sensor = None
    #         # The figure will be recreated in the next update_plot call
    #     if self.debug_console:
    #         print(f"[Sensor View] {'Enabled' if self.show_robot_sensor_view else 'Disabled'}, will recreate figure on next update")
    # 
    # def on_sensor_robot_changed(self, value):
    #     self.selected_robot_for_view = value
    #     self.sensor_robot_spin.setRange(0, max(0, self.num_robots - 1))
    
    def on_follow_robot_changed(self, state):
        self.follow_robot = (state == Qt.CheckState.Checked)
    
    def on_robot_id_changed(self, value):
        self.robot_id_to_follow = value
    
    def on_step_by_step_changed(self, state):
        self.step_by_step = (state == Qt.CheckState.Checked)
        if self.step_by_step:
            self.running = False
            self.run_cb.setChecked(False)
    
    def on_run_changed(self, state):
        self.running = (state == Qt.CheckState.Checked)
        if self.running:
            self.step_by_step = False
            self.step_by_step_cb.setChecked(False)

    def update_obstacles(self):
        """Update obstacle positions."""
        for i, (obs_pos, obs_vel) in enumerate(zip(self.obstacles, self.obstacle_velocities)):
            if self.obstacle_moving:
                # Update position
                new_pos = obs_pos + obs_vel * self.dt
                
                # Bounce off boundary
                dist_from_center = np.linalg.norm(new_pos - self.circle_center)
                if dist_from_center > self.boundary_radius - self.obstacle_radius:
                    # Reflect velocity
                    direction = (new_pos - self.circle_center) / dist_from_center
                    obs_vel = obs_vel - 2 * np.dot(obs_vel, direction) * direction
                    self.obstacle_velocities[i] = obs_vel
                    # Clamp position
                    new_pos = self.circle_center + direction * (self.boundary_radius - self.obstacle_radius - 0.1)
                
                self.obstacles[i] = new_pos
    
    def update_simulation(self):
        """Update simulation state with RVO, obstacles, and mixed robot types."""
        if self.is_shutting_down:
            return
        if not self.running and not self.step_by_step:
            return
        
        # Update FPS
        self.fps_timer += 16
        self.fps_counter += 1
        if self.fps_timer >= 1000:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.fps_timer = 0
            self.fps_label.setText(f"{self.fps} FPS")
        
        self.time += self.dt
        
        # Update obstacles
        self.update_obstacles()
        
        # Phase 1: Compute velocities for all robots (RVO planning)
        robot_velocities = {}  # Store computed velocities before applying movement
        for robot in self.robots:
            if not robot.is_active:
                continue
            
            # Preferred velocity (circular motion with individual paths)
            # Use robot's direction to determine rotation direction
            direction = getattr(robot, 'direction', 1)  # Default to clockwise if not set
            angular_velocity = 0.03 * direction  # Different directions for different robots
            
            # Update ground truth angle for motion planning
            if not hasattr(robot, 'gt_angle'):
                robot.gt_angle = getattr(robot, 'estimated_angle', 0.0)
            robot.gt_angle += angular_velocity * self.dt
            robot.estimated_angle += angular_velocity * self.dt  # Also update estimated for motion model
            
            # Use individual path parameters to avoid clustering
            if hasattr(robot, 'path_radius') and hasattr(robot, 'path_center_offset'):
                path_center = self.circle_center + robot.path_center_offset
                preferred_pos = path_center + robot.path_radius * np.array([
                    np.cos(robot.gt_angle), np.sin(robot.gt_angle)
                ])
            else:
                # Fallback to default circular path
                preferred_pos = self.circle_center + self.circle_radius * np.array([
                    np.cos(robot.gt_angle), np.sin(robot.gt_angle)
                ])
            
            # Preferred velocity based on ground truth position (where robot actually is)
            preferred_vel = (preferred_pos - robot.gt_pos) / self.dt
            # Limit preferred velocity magnitude
            vel_mag = np.linalg.norm(preferred_vel)
            if vel_mag > self.rvo_max_speed:
                preferred_vel = preferred_vel / vel_mag * self.rvo_max_speed
            robot.preferred_vel = preferred_vel
            
            # Apply RVO collision avoidance
            if self.use_rvo:
                neighbors = []
                # Add other robots as neighbors (use GROUND TRUTH positions for actual collision avoidance)
                for neighbor in self.robots:
                    if neighbor.id != robot.id and neighbor.is_active:
                        # Estimate neighbor velocity from ground truth path
                        if len(self.robot_gt_paths[neighbor.id]) > 1:
                            neighbor_vel = (neighbor.gt_pos - self.robot_gt_paths[neighbor.id][-1]) / self.dt
                        else:
                            neighbor_vel = np.array([0.0, 0.0])
                        neighbors.append({
                            'pos': neighbor.gt_pos,  # Use ground truth position
                            'vel': neighbor_vel,
                            'radius': neighbor.radius,
                            'robot_radius': robot.radius
                        })
                
                # Add obstacles as neighbors
                for obs_pos, obs_vel in zip(self.obstacles, self.obstacle_velocities):
                    neighbors.append({
                        'pos': obs_pos,
                        'vel': obs_vel,
                        'radius': self.obstacle_radius,
                        'robot_radius': robot.radius
                    })
                
                # MAZE DISABLED - Commented out
                # Add maze walls as static obstacles (only if RVO is enabled)
                # When RVO is disabled, walls are still handled by collision detection below
                if False and self.use_rvo and self.use_maze and self.maze_walls:  # MAZE DISABLED
                    for wall_start, wall_end in self.maze_walls:
                        # Find closest point on wall segment to robot (use ground truth for actual avoidance)
                        closest_point = self._closest_point_on_segment(robot.gt_pos, wall_start, wall_end)
                        dist_to_wall = np.linalg.norm(robot.gt_pos - closest_point)
                        
                        # Only add as obstacle if robot is close enough to the wall
                        wall_avoidance_dist = robot.radius + 0.5  # Robot radius + safety margin
                        if dist_to_wall < wall_avoidance_dist * 2:  # Check within 2x avoidance distance
                            # Treat wall as a static obstacle at closest point
                            # Use a small radius to represent the wall thickness
                            wall_radius = 0.3  # Effective wall thickness for collision
                            neighbors.append({
                                'pos': closest_point,
                                'vel': np.array([0.0, 0.0]),  # Walls are static
                                'radius': wall_radius,
                                'robot_radius': robot.radius
                            })
                
                # Compute safe velocity using RVO (use ground truth position for actual collision avoidance)
                # Increase neighbor_dist to keep robots more separated
                safe_vel = RVO.compute_rvo_velocity(
                    robot.gt_pos, preferred_vel, neighbors,
                    self.rvo_time_horizon, 
                    max(self.rvo_neighbor_dist, (robot.radius + 0.5) * 2),  # At least 2x robot diameter
                    self.rvo_max_speed
                )
                
                # Convert to differential drive control (use ground truth angle)
                linear_vel, angular_vel = RVO.compute_differential_control(
                    robot.gt_pos, robot.gt_angle, safe_vel,
                    max_linear=self.rvo_max_speed, max_angular=1.0
                )
            else:
                # No RVO, use preferred velocity directly (use ground truth position/angle)
                linear_vel, angular_vel = RVO.compute_differential_control(
                    robot.gt_pos, robot.gt_angle, preferred_vel,
                    max_linear=self.rvo_max_speed, max_angular=1.0
                )
            
            # Store velocities for this robot
            robot_velocities[robot.id] = {
                'linear_vel': linear_vel,
                'angular_vel': angular_vel
            }
        
        # Phase 2: Apply movement for all robots
        robot_old_positions = {}  # Store old positions for collision resolution
        for robot in self.robots:
            if not robot.is_active:
                continue
            
            # Get computed velocities
            vel_data = robot_velocities[robot.id]
            robot.linear_vel = vel_data['linear_vel']
            robot.angular_vel = vel_data['angular_vel']
            
            # Store old position before movement
            robot_old_positions[robot.id] = {
                'pos': robot.gt_pos.copy(),
                'angle': robot.gt_angle
            }
            
            # Update GROUND TRUTH position (actual physical movement)
            # Ground truth moves based on actual kinematics, independent of estimates
            # Update ground truth kinematics
            if abs(robot.angular_vel) > 1e-6:
                # Arc motion
                radius = robot.linear_vel / (robot.angular_vel + 1e-6)
                dtheta = robot.angular_vel * self.dt
                dx = radius * (np.sin(robot.gt_angle + dtheta) - np.sin(robot.gt_angle))
                dy = radius * (-np.cos(robot.gt_angle + dtheta) + np.cos(robot.gt_angle))
            else:
                # Straight motion
                dx = robot.linear_vel * np.cos(robot.gt_angle) * self.dt
                dy = robot.linear_vel * np.sin(robot.gt_angle) * self.dt
            
            robot.gt_pos += np.array([dx, dy])
            robot.gt_angle += robot.angular_vel * self.dt
        
        # Phase 3: Resolve all collisions (process all robots to avoid double-processing)
        # Check for robot-to-robot collisions (prevent robots from overlapping)
        # This is a safety check that works even if RVO is disabled or fails
        processed_pairs = set()
        for i, robot in enumerate(self.robots):
            if not robot.is_active:
                continue
            for j, other_robot in enumerate(self.robots):
                if i >= j or not other_robot.is_active:
                    continue
                
                # Avoid processing same pair twice
                pair_id = tuple(sorted([robot.id, other_robot.id]))
                if pair_id in processed_pairs:
                    continue
                processed_pairs.add(pair_id)
                
                dist_to_other = np.linalg.norm(robot.gt_pos - other_robot.gt_pos)
                min_separation = self.min_robot_separation  # Minimum distance between robot centers
                
                if dist_to_other < min_separation:
                    # Collision detected - push robots apart
                    separation_dir = robot.gt_pos - other_robot.gt_pos
                    if np.linalg.norm(separation_dir) < 1e-6:
                        # Robots are exactly on top of each other - random separation
                        angle = np.random.uniform(0, 2 * np.pi)
                        separation_dir = np.array([np.cos(angle), np.sin(angle)])
                    else:
                        separation_dir = separation_dir / np.linalg.norm(separation_dir)
                    
                    # Push both robots apart to maintain minimum separation
                    overlap = min_separation - dist_to_other
                    push_distance = overlap / 2.0  # Split the push between both robots
                    robot.gt_pos += separation_dir * push_distance
                    other_robot.gt_pos -= separation_dir * push_distance
                    
                    # Also reduce velocities to prevent immediate re-collision
                    robot.linear_vel *= 0.7  # Slow down
                    other_robot.linear_vel *= 0.7  # Slow down
        
        # Phase 4: Check wall collisions for all robots
        for robot in self.robots:
            if not robot.is_active:
                continue
            
            old_data = robot_old_positions.get(robot.id, {'pos': robot.gt_pos.copy(), 'angle': robot.gt_angle})
            old_gt_pos = old_data['pos']
            old_gt_angle = old_data['angle']
            
            # MAZE DISABLED - Commented out
            # Check for wall collisions on GROUND TRUTH position
            if False and self.use_maze and self.maze_walls:  # MAZE DISABLED
                # Check if new ground truth position collides with any wall
                collision_detected = False
                for wall_start, wall_end in self.maze_walls:
                    if self._robot_wall_collision(robot.gt_pos, wall_start, wall_end, robot.radius):
                        collision_detected = True
                        # Revert ground truth to old position
                        robot.gt_pos = old_gt_pos.copy()
                        robot.gt_angle = old_gt_angle
                        # Apply repulsion away from wall to prevent getting stuck
                        closest = self._closest_point_on_segment(robot.gt_pos, wall_start, wall_end)
                        repulsion_dir = robot.gt_pos - closest
                        repulsion_dist = np.linalg.norm(repulsion_dir)
                        if repulsion_dist < robot.radius + 0.1 and repulsion_dist > 1e-6:
                            repulsion_dir = repulsion_dir / repulsion_dist
                            # Push robot away from wall
                            push_distance = (robot.radius + 0.15) - repulsion_dist
                            robot.gt_pos += repulsion_dir * push_distance
                        # Reduce velocity when hitting wall
                        robot.linear_vel *= 0.5
                        break
            
            # Update ESTIMATED position (mu) - this is what GPB localizes
            # The estimated position should track ground truth but with errors that GPB corrects
            # For now, update mu based on kinematics to track motion (GPB will correct errors)
            # Use the same kinematics as ground truth but this represents the robot's belief
            if abs(robot.angular_vel) > 1e-6:
                # Arc motion
                radius = robot.linear_vel / (robot.angular_vel + 1e-6)
                dtheta = robot.angular_vel * self.dt
                dx = radius * (np.sin(robot.angle + dtheta) - np.sin(robot.angle))
                dy = radius * (-np.cos(robot.angle + dtheta) + np.cos(robot.angle))
            else:
                # Straight motion
                dx = robot.linear_vel * np.cos(robot.angle) * self.dt
                dy = robot.linear_vel * np.sin(robot.angle) * self.dt
            
            robot.mu += np.array([dx, dy])
            robot.angle += robot.angular_vel * self.dt
            
            # Update expected position for motion model
            robot.expected_pos = robot.mu.copy()
            
            # Sync visualization angle with ground truth
            robot.angle = robot.gt_angle
            
            # Update paths (limit to 200 for cleaner visualization)
            self.robot_paths[robot.id].append(robot.mu.copy())
            self.robot_gt_paths[robot.id].append(robot.gt_pos.copy())
            # Keep more path history for better coverage visualization
            if len(self.robot_paths[robot.id]) > 1000:
                self.robot_paths[robot.id].pop(0)
            if len(self.robot_gt_paths[robot.id]) > 1000:
                self.robot_gt_paths[robot.id].pop(0)
            
            # Update coverage map based on ground truth position
            self._update_coverage_grid(robot.gt_pos, robot.radius)
        
        # Federated Learning: Collect updates from robots (non-IID data)
        # Only run if enabled and GA optimizer is initialized
        if self.use_federated_learning:
            # Ensure federated_ga is initialized
            if self.federated_ga is None:
                robot_types_list = [r.robot_type for r in self.robots]
                self.federated_ga = FederatedLearningGA(
                    population_size=20,
                    mutation_rate=0.15,
                    crossover_rate=0.7,
                    num_robot_types=len(set(robot_types_list)),
                    max_clients_per_round=None
                )
                self.federated_ga.initialize_population(len(self.robots), robot_types_list)
                if self.debug_console:
                    print(f"[Federated Learning] Auto-initialized GA optimizer")
            
            # Run federated learning more frequently for better visualization (every 20 iterations)
            if self.iter_count % 20 == 0:
                self._collect_federated_updates()
                if len(self.robot_updates) > 0:
                    # Use GA to optimize aggregation
                    robot_types_list = [r.robot_type for r in self.robots]
                    best_config, fitness_history = self.federated_ga.evolve(
                        self.robot_updates, robot_types_list, generations=5, track_history=True
                    )
                    
                    # Store federated learning metrics for visualization
                    if fitness_history:
                        self.federated_fitness_history.append({
                            'round': self.federated_round,
                            'best': fitness_history[-1]['best'],
                            'avg': fitness_history[-1]['avg']
                        })
                    
                    # Store aggregation weights
                    type_names = list(self.federated_ga.type_to_index.keys())
                    weights_dict = {}
                    for type_name, idx in self.federated_ga.type_to_index.items():
                        weights_dict[type_name] = best_config['aggregation_weights'][idx]
                    self.federated_aggregation_weights_history.append({
                        'round': self.federated_round,
                        'weights': weights_dict
                    })
                    
                    # Store client selection
                    selected_clients = np.where(best_config['client_selection'])[0].tolist()
                    self.federated_client_selection_history.append({
                        'round': self.federated_round,
                        'selected': selected_clients
                    })
                    
                    # Calculate and store metrics
                    diversity = self.federated_ga._compute_diversity_score(
                        [robot_types_list[i] for i in selected_clients]
                    )
                    fairness = self.federated_ga._compute_fairness_score(
                        best_config,
                        [self.robot_updates[i] for i in selected_clients],
                        [robot_types_list[i] for i in selected_clients]
                    )
                    quality = self.federated_ga._compute_update_quality(
                        [self.robot_updates[i] for i in selected_clients]
                    )
                    self.federated_metrics_history.append({
                        'round': self.federated_round,
                        'diversity': diversity,
                        'fairness': fairness,
                        'quality': quality
                    })
                    
                    # Aggregate updates using optimized configuration
                    global_update = self.federated_ga.aggregate_updates(
                        best_config, self.robot_updates, robot_types_list
                    )
                    if global_update is not None:
                        # Apply global update to robots (simplified: update position estimates)
                        self._apply_global_update(global_update, best_config)
                    
                    if self.debug_console:
                        print(f"[Federated Learning] Round {self.federated_round}: "
                              f"Fitness={fitness_history[-1]['best']:.3f}, "
                              f"Diversity={diversity:.3f}, Fairness={fairness:.3f}, "
                              f"Selected {len(selected_clients)}/{len(self.robots)} clients")
                    
                    self.federated_round += 1
                else:
                    # No robot updates collected - this shouldn't happen but log it
                    if self.debug_console and self.iter_count % 100 == 0:
                        print(f"[Federated Learning] Warning: No robot updates collected (iter {self.iter_count})")
        
        # Perform GBP iterations (GPB for robot-to-robot observations)
        if self.iter_count % self.iter_before_motion == 0:
            # Reset message counter for this iteration
            self.messages_per_iteration = 0
            message_details = []
            # Clear active connections (will be rebuilt this iteration)
            self.active_connections.clear()
            
            # Exchange messages between robots (Distributed MAP Inference)
            # Phase 1: Collect messages for ALL robots simultaneously (using current estimates)
            # This ensures all robots use the same "snapshot" of positions, making it more parallel
            robot_messages = {}  # Store messages for each robot before applying updates
            
            for i, robot in enumerate(self.robots):
                if not robot.is_active:
                    continue
                    
                robot.inbox.clear()
                messages_received = 0
                
                # 1. ODOMETRY FACTOR (Robot's own motion - part of local fragment)
                # Each robot maintains its own odometry factor
                if self.use_odometry and hasattr(robot, 'expected_pos'):
                    odometry_message = robot.get_odometry_message(
                        robot.expected_pos, self.odometry_noise_std
                    )
                    robot.inbox['odometry'] = odometry_message
                    messages_received += 1
                
                # 2. LANDMARK OBSERVATIONS (if enabled)
                # Use GA to select optimal landmarks if enabled
                landmark_indices_to_observe = None
                if self.use_ga_landmark_selection and self.num_landmarks > 1:
                    # Run GA to select optimal landmarks for this robot
                    # Only recompute every few iterations to save computation
                    if robot.id not in self.robot_selected_landmarks or self.iter_count % 10 == 0:
                        try:
                            selected, fitness_history = select_optimal_landmarks(
                                robot.mu,
                                self.landmarks[:self.num_landmarks],
                                self.sensor_range,
                                None,  # self.maze_walls if self.use_maze else None,  # MAZE DISABLED
                                max_selected=self.ga_max_selected,
                                ga_generations=self.ga_generations,
                                track_history=True
                            )
                            self.robot_selected_landmarks[robot.id] = selected
                            # Store fitness history for plotting
                            if fitness_history:
                                self.ga_evolution_data[robot.id] = fitness_history
                            if self.debug_console and self.iter_count % 50 == 0:
                                print(f"[GA] {robot.name} selected landmarks: {selected}")
                        except Exception as e:
                            if self.debug_console:
                                print(f"[GA Error] {robot.name}: {e}")
                            # Fallback: select all visible landmarks
                            selected = list(range(min(self.ga_max_selected, self.num_landmarks)))
                            self.robot_selected_landmarks[robot.id] = selected
                    landmark_indices_to_observe = self.robot_selected_landmarks.get(robot.id, None)
                
                if self.use_landmark_only:
                    # Observe all landmarks in range (or GA-selected if enabled)
                    for lm_idx, landmark_pos in enumerate(self.landmarks[:self.num_landmarks]):
                        # Skip if GA selection is active and this landmark not selected
                        if landmark_indices_to_observe is not None and lm_idx not in landmark_indices_to_observe:
                            continue
                        
                        # Check if landmark is in range using GROUND TRUTH position (actual physical distance)
                        dist_gt = np.linalg.norm(robot.gt_pos - landmark_pos)
                        if dist_gt < self.sensor_range:
                            # Check line-of-sight if maze is enabled (using ground truth position)
                            if self.use_maze and self.maze_walls:
                                from ga_landmark_selector import LandmarkSelectorGA
                                ga_temp = LandmarkSelectorGA()
                                if not ga_temp._check_line_of_sight(robot.gt_pos, landmark_pos, self.maze_walls):
                                    continue  # Landmark blocked by wall
                            
                            # Generate measurement from ground truth position (actual observation)
                            angle_gt = np.arctan2(robot.gt_pos[1] - landmark_pos[1],
                                                 robot.gt_pos[0] - landmark_pos[0])
                            measurement = np.array([dist_gt, angle_gt]) + np.random.normal(0, self.noise_std[0], 2)
                            message = robot.get_local_message(landmark_pos, measurement, self.noise_std, self.is_robust)
                            robot.inbox[f'landmark_{lm_idx}'] = message
                            messages_received += 1
                            message_details.append(f"  {robot.name}: received from landmark {lm_idx+1} (dist={dist_gt:.2f})")
                            
                            # Track SLAM observation for factor graph visualization
                            if self.show_slam_visualization:
                                if robot.id not in self.slam_observations:
                                    self.slam_observations[robot.id] = []
                                # Store observation: (pose, landmark_idx, iteration)
                                self.slam_observations[robot.id].append((
                                    robot.gt_pos.copy(), lm_idx, self.iter_count
                                ))
                                # Limit history to prevent memory issues
                                if len(self.slam_observations[robot.id]) > 500:
                                    self.slam_observations[robot.id].pop(0)
                else:
                    # Include landmark messages even when not in landmark-only mode
                    for lm_idx, landmark_pos in enumerate(self.landmarks[:self.num_landmarks]):
                        # Skip if GA selection is active and this landmark not selected
                        if landmark_indices_to_observe is not None and lm_idx not in landmark_indices_to_observe:
                            continue
                        
                        # Check if landmark is in range using GROUND TRUTH position (actual physical distance)
                        dist_gt = np.linalg.norm(robot.gt_pos - landmark_pos)
                        if dist_gt < self.sensor_range:
                            # Check line-of-sight if maze is enabled (using ground truth position)
                            if self.use_maze and self.maze_walls:
                                from ga_landmark_selector import LandmarkSelectorGA
                                ga_temp = LandmarkSelectorGA()
                                if not ga_temp._check_line_of_sight(robot.gt_pos, landmark_pos, self.maze_walls):
                                    continue  # Landmark blocked by wall
                            
                            # Generate measurement from ground truth position (actual observation)
                            angle_gt = np.arctan2(robot.gt_pos[1] - landmark_pos[1],
                                                 robot.gt_pos[0] - landmark_pos[0])
                            measurement = np.array([dist_gt, angle_gt]) + np.random.normal(0, self.noise_std[0], 2)
                            message = robot.get_local_message(landmark_pos, measurement, self.noise_std, self.is_robust)
                            robot.inbox[f'landmark_{lm_idx}'] = message
                            messages_received += 1
                
                # 3. PEER OBSERVATIONS (Robot-to-robot messages)
                # Each robot observes ALL other robots within sensor range and receives their state
                # This ensures bidirectional communication: Robot i gets Robot j's state, and vice versa
                if not self.use_landmark_only:
                    for j, neighbor in enumerate(self.robots):
                        if i != j and neighbor.is_active:
                            # Check distance between ESTIMATED positions
                            dist_estimated = np.linalg.norm(robot.mu - neighbor.mu)
                            if dist_estimated < self.sensor_range:
                                # Track this connection for visualization
                                connection = tuple(sorted([robot.id, neighbor.id]))
                                self.active_connections.add(connection)
                                
                                # Simulate asynchronous communication
                                message_dropped = False
                                message_delayed = False
                                
                                if self.async_communication:
                                    # Packet loss simulation (cut off wifi)
                                    if np.random.random() < self.communication_drop_rate:
                                        self.dropped_messages_count += 1
                                        message_dropped = True
                                        # Add to message queue for delayed delivery (simulating retry)
                                        queue_entry = {
                                            'from': neighbor.id,
                                            'iter': self.iter_count,
                                            'delayed': True
                                        }
                                        # Check if already in queue to avoid duplicates
                                        if not any(q['from'] == neighbor.id for q in robot.message_queue):
                                            robot.message_queue.append(queue_entry)
                                    
                                    # Message delay (store in queue for future delivery)
                                    if not message_dropped and self.communication_delay > 0:
                                        if neighbor.id not in robot.last_message_time or \
                                           self.iter_count - robot.last_message_time[neighbor.id] >= self.communication_delay:
                                            robot.last_message_time[neighbor.id] = self.iter_count
                                        else:
                                            message_delayed = True
                                            # Message delayed - add to queue
                                            queue_entry = {
                                                'from': neighbor.id,
                                                'iter': self.iter_count,
                                                'delayed': True
                                            }
                                            if not any(q['from'] == neighbor.id for q in robot.message_queue):
                                                robot.message_queue.append(queue_entry)
                                
                                # Only send message if not dropped or delayed
                                if not message_dropped and not message_delayed:
                                    # Generate measurement from robot's GROUND TRUTH position observing neighbor's GROUND TRUTH position
                                    # This measurement represents: "I see neighbor at distance r and angle" (actual physical observation)
                                    dx_gt = neighbor.gt_pos[0] - robot.gt_pos[0]
                                    dy_gt = neighbor.gt_pos[1] - robot.gt_pos[1]
                                    r_gt = np.sqrt(dx_gt**2 + dy_gt**2)
                                    
                                    # Check if actually in range (using ground truth)
                                    if r_gt < self.sensor_range:
                                        # Generate measurement from ground truth positions (actual observation)
                                        angle_gt = np.arctan2(dy_gt, dx_gt)
                                        
                                        # Add sensor noise (sometimes with outliers for robust testing)
                                        if self.is_robust and np.random.random() < 0.05:  # 5% outlier rate
                                            measurement = np.array([r_gt, angle_gt]) + np.random.normal(0, self.noise_std[0] * 5, 2)
                                        else:
                                            measurement = np.array([r_gt, angle_gt]) + np.random.normal(0, self.noise_std[0], 2)
                                        
                                        # Neighbor creates message containing its state (position information)
                                        # This message says: "Based on your observation of me, here's information about my position"
                                        # Use estimated position for linearization point in message
                                        message = neighbor.get_local_message(robot.mu, measurement, self.noise_std, self.is_robust)
                                        robot.inbox[neighbor.id] = message
                                        messages_received += 1
                                        
                                        # For logging
                                        dist_gt = np.linalg.norm(robot.gt_pos - neighbor.gt_pos)
                                        message_details.append(f"  {robot.name}: received from {neighbor.name} (est_dist={dist_estimated:.2f}, gt_dist={dist_gt:.2f})")
                                elif message_dropped:
                                    message_details.append(f"  {robot.name}: message from {neighbor.name} DROPPED (packet loss)")
                                elif message_delayed:
                                    message_details.append(f"  {robot.name}: message from {neighbor.name} DELAYED (queued)")
                
                # Process queued messages (delayed/dropped messages that are now deliverable)
                if self.async_communication and robot.message_queue:
                    processed_queue = []
                    for queued_msg in robot.message_queue:
                        neighbor_id = queued_msg['from']
                        neighbor = next((r for r in self.robots if r.id == neighbor_id), None)
                        if neighbor and neighbor.is_active:
                            dist_estimated = np.linalg.norm(robot.mu - neighbor.mu)
                            if dist_estimated < self.sensor_range:
                                # Retry delivery after delay
                                if self.iter_count - queued_msg['iter'] >= 2:  # 2 iteration delay
                                    # Generate measurement
                                    dx = neighbor.mu[0] - robot.mu[0]
                                    dy = neighbor.mu[1] - robot.mu[1]
                                    r = np.sqrt(dx**2 + dy**2)
                                    angle = np.arctan2(dy, dx)
                                    measurement = np.array([r, angle]) + np.random.normal(0, self.noise_std[0], 2)
                                    message = neighbor.get_local_message(robot.mu, measurement, self.noise_std, self.is_robust)
                                    robot.inbox[neighbor.id] = message
                                    messages_received += 1
                                    processed_queue.append(queued_msg)
                    # Remove processed messages from queue
                    robot.message_queue = [msg for msg in robot.message_queue if msg not in processed_queue]
                
                # Track message queue size
                self.message_queue_sizes[robot.id] = len(robot.message_queue)
                
                # Store messages and stats for this robot (don't update position yet)
                robot_messages[robot.id] = {
                    'inbox': robot.inbox.copy(),  # Copy inbox for later update
                    'messages_received': messages_received,
                    'old_mu': robot.mu.copy()
                }
            
            # Phase 2: Update ALL robots simultaneously (parallel update)
            # This makes the system more distributed - all robots update at once using the same snapshot
            for i, robot in enumerate(self.robots):
                if not robot.is_active:
                    continue
                
                if robot.id in robot_messages:
                    msg_data = robot_messages[robot.id]
                    robot.inbox = msg_data['inbox']
                    messages_received = msg_data['messages_received']
                    
                    # 4. UPDATE ROBOT POSITION (Distributed GBP)
                    # Combine all messages (odometry + observations) to update estimate
                    if robot.inbox:
                        old_mu = msg_data['old_mu']
                        robot.update_from_web()  # GBP combines all messages
                        # Apply damping
                        robot.mu = (1 - self.damping) * robot.mu + self.damping * old_mu
                        
                        # Apply motion model (weak prior toward expected circular position)
                        if self.use_motion_model and hasattr(robot, 'expected_pos'):
                            # Blend GBP estimate with motion model prediction
                            robot.mu = (1 - self.motion_model_weight) * robot.mu + \
                                      self.motion_model_weight * robot.expected_pos
                        
                        # Apply boundary constraints (only hard boundary, no soft radius constraint)
                        if self.use_boundary_constraint:
                            # Clamp position to boundary radius
                            dist_from_center = np.linalg.norm(robot.mu - self.circle_center)
                            if dist_from_center > self.boundary_radius:
                                direction = (robot.mu - self.circle_center) / dist_from_center
                                robot.mu = self.circle_center + direction * self.boundary_radius
                            
                            # Update estimated angle based on new position
                            direction = robot.mu - self.circle_center
                            robot.estimated_angle = np.arctan2(direction[1], direction[0])
                        
                        self.robot_paths[robot.id].append(robot.mu.copy())
                        
                        # Limit path length
                        if len(self.robot_paths[robot.id]) > 500:
                            self.robot_paths[robot.id].pop(0)
                    
                    self.messages_per_iteration += messages_received
                    # Calculate GPB error using Mahalanobis distance (accounts for uncertainty)
                    gpb_error = self._mahalanobis_distance(robot.mu, robot.gt_pos, robot.Lambda)
                    self.last_update_stats[robot.id] = {
                        'messages': messages_received,
                        'position': robot.mu.copy(),
                        'error': gpb_error
                    }
            
            
            # Update total message count
            self.total_messages += self.messages_per_iteration
            
            # Console logging
            if self.debug_console and self.messages_per_iteration > 0:
                print(f"\n[Iteration {self.iter_count}] Messages exchanged: {self.messages_per_iteration}")
                
                # Show message summary per robot
                for robot in self.robots:
                    if robot.id in self.last_update_stats:
                        stats = self.last_update_stats[robot.id]
                        neighbor_count = len([k for k in robot.inbox.keys() if not k.startswith('landmark_') and not k.startswith('odometry')])
                        print(f"  {robot.name}: received {stats['messages']} messages ({neighbor_count} from neighbors)")
                
                for detail in message_details[:15]:  # Show more details
                    print(detail)
                if len(message_details) > 15:
                    print(f"  ... and {len(message_details) - 15} more messages")
                
                # Print position updates
                active_robots = sum(1 for r in self.robots if len(r.inbox) > 0)
                avg_error = np.mean([stats['error'] for stats in self.last_update_stats.values()])
                print(f"  Active robots: {active_robots}/{len(self.robots)}, Avg position error: {avg_error:.3f}")
                
                # Check for robots not receiving messages
                robots_without_messages = [r for r in self.robots if r.is_active and len([k for k in r.inbox.keys() if not k.startswith('odometry')]) == 0]
                if robots_without_messages:
                    print(f"  WARNING: {len(robots_without_messages)} robots received no neighbor messages: {[r.name for r in robots_without_messages]}")
                
                # Track GPB performance metrics (only during GBP iterations)
                if self.last_update_stats:
                    avg_error = np.mean([stats['error'] for stats in self.last_update_stats.values()])
                    max_error = max([stats['error'] for stats in self.last_update_stats.values()])
                    
                    # Store average in history
                    self.gpb_error_history.append(avg_error)
                    self.gpb_message_history.append(self.messages_per_iteration)
                    
                    # Store individual robot errors
                    for robot_id, stats in self.last_update_stats.items():
                        if robot_id not in self.gpb_robot_errors:
                            self.gpb_robot_errors[robot_id] = []
                        self.gpb_robot_errors[robot_id].append(stats['error'])
                        # Limit individual robot error history
                        if len(self.gpb_robot_errors[robot_id]) > self.max_history_length:
                            self.gpb_robot_errors[robot_id].pop(0)
                    
                    # Calculate convergence metric (rate of error reduction)
                    if len(self.gpb_error_history) > 10:
                        recent_errors = self.gpb_error_history[-10:]
                        convergence_rate = (recent_errors[0] - recent_errors[-1]) / max(recent_errors[0], 0.001)
                        self.gpb_convergence_history.append(convergence_rate)
                    else:
                        self.gpb_convergence_history.append(0.0)
                    
                    # Limit history length
                    if len(self.gpb_error_history) > self.max_history_length:
                        self.gpb_error_history.pop(0)
                        self.gpb_message_history.pop(0)
                        if len(self.gpb_convergence_history) > 0:
                            self.gpb_convergence_history.pop(0)
        
        self.iter_count += 1
        self.update_status_display()
        self.update_plot()
    
    def draw_robot_sensor_view(self):
        """Draw what a selected robot sees from its perspective."""
        if self.ax_sensor is None:
            if self.debug_console:
                print(f"[Sensor View] ax_sensor is None, cannot draw")
            return
        
        if self.selected_robot_for_view >= len(self.robots):
            if self.debug_console:
                print(f"[Sensor View] Invalid robot ID: {self.selected_robot_for_view} (max: {len(self.robots)-1})")
            return
        
        robot = self.robots[self.selected_robot_for_view]
        if not robot.is_active:
            if self.debug_console:
                print(f"[Sensor View] Robot {self.selected_robot_for_view} is not active")
            return
        
        self.ax_sensor.clear()
        self.ax_sensor.set_aspect('equal')
        
        # Center view on robot
        center = robot.mu
        view_range = self.sensor_range * 1.2
        
        # Draw sensor range circle
        sensor_circle = plt.Circle(center, self.sensor_range, fill=False, 
                                  linestyle='--', color='gray', linewidth=2, alpha=0.5)
        self.ax_sensor.add_patch(sensor_circle)
        
        # Draw robot at center
        self.ax_sensor.plot(center[0], center[1], 'ro', markersize=15, 
                          label=f'{robot.name}', zorder=100)
        # Draw robot orientation arrow
        arrow_length = 1.5
        dx = arrow_length * np.cos(robot.angle)
        dy = arrow_length * np.sin(robot.angle)
        self.ax_sensor.arrow(center[0], center[1], dx, dy, 
                           head_width=0.5, head_length=0.4, fc='red', ec='red', zorder=101)
        
        # Draw visible landmarks
        visible_landmarks = []
        for lm_idx, landmark_pos in enumerate(self.landmarks[:self.num_landmarks]):
            dist = np.linalg.norm(robot.mu - landmark_pos)
            if dist < self.sensor_range:
                # Check line-of-sight if maze enabled
                visible = True
                if self.use_maze and self.maze_walls:
                    from ga_landmark_selector import LandmarkSelectorGA
                    ga_temp = LandmarkSelectorGA()
                    visible = ga_temp._check_line_of_sight(robot.mu, landmark_pos, self.maze_walls)
                
                if visible:
                    # Check if GA-selected
                    is_selected = False
                    if self.use_ga_landmark_selection:
                        if robot.id in self.robot_selected_landmarks:
                            is_selected = lm_idx in self.robot_selected_landmarks[robot.id]
                    
                    color = 'gold' if is_selected else 'black'
                    self.ax_sensor.plot(landmark_pos[0], landmark_pos[1], 's', 
                                      markersize=10, color=color, 
                                      markeredgecolor='orange' if is_selected else 'white',
                                      markeredgewidth=2, zorder=50)
                    # Draw line to landmark
                    self.ax_sensor.plot([center[0], landmark_pos[0]], 
                                      [center[1], landmark_pos[1]], 
                                      'r--', alpha=0.5, linewidth=1.5, zorder=40)
                    self.ax_sensor.text(landmark_pos[0] + 0.5, landmark_pos[1] + 0.5,
                                       f'L{lm_idx+1}', fontsize=8, fontweight='bold',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                                       zorder=51)
                    visible_landmarks.append((lm_idx, landmark_pos, is_selected))
        
        # Draw visible robots
        for other_robot in self.robots:
            if other_robot.id == robot.id or not other_robot.is_active:
                continue
            dist = np.linalg.norm(robot.mu - other_robot.mu)
            if dist < self.sensor_range:
                # Check if in inbox (actually communicating)
                is_communicating = other_robot.id in robot.inbox
                color = 'green' if is_communicating else 'blue'
                self.ax_sensor.plot(other_robot.mu[0], other_robot.mu[1], 'o', 
                                  markersize=10, color=color, 
                                  markeredgecolor='darkgreen' if is_communicating else 'darkblue',
                                  markeredgewidth=2, zorder=50)
                # Draw line to other robot
                line_style = 'g-' if is_communicating else 'b--'
                self.ax_sensor.plot([center[0], other_robot.mu[0]], 
                                  [center[1], other_robot.mu[1]], 
                                  line_style, alpha=0.6, linewidth=2, zorder=40)
                self.ax_sensor.text(other_robot.mu[0] + 0.5, other_robot.mu[1] + 0.5,
                                   other_robot.name, fontsize=7, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
                                   zorder=51)
        
        # Draw obstacles in range
        for obs_pos in self.obstacles:
            dist = np.linalg.norm(robot.mu - obs_pos)
            if dist < self.sensor_range:
                obstacle_circle = plt.Circle(obs_pos, self.obstacle_radius, 
                                          fill=True, color='red', alpha=0.4, 
                                          edgecolor='darkred', linewidth=2, zorder=30)
                self.ax_sensor.add_patch(obstacle_circle)
        
        # Draw maze walls in view
        if self.use_maze and self.maze_walls:
            for wall_start, wall_end in self.maze_walls:
                # Check if wall is in view
                wall_center = (wall_start + wall_end) / 2
                if np.linalg.norm(robot.mu - wall_center) < view_range:
                    self.ax_sensor.plot([wall_start[0], wall_end[0]], 
                                      [wall_start[1], wall_end[1]], 
                                      'k-', linewidth=3, alpha=0.8, zorder=20)
        
        # Set view limits
        self.ax_sensor.set_xlim(center[0] - view_range, center[0] + view_range)
        self.ax_sensor.set_ylim(center[1] - view_range, center[1] + view_range)
        self.ax_sensor.set_xlabel('X', fontsize=9)
        self.ax_sensor.set_ylabel('Y', fontsize=9)
        self.ax_sensor.set_title(f'{robot.name} Sensor View\n'
                               f'Landmarks: {len(visible_landmarks)} visible, '
                               f'{sum(1 for _, _, sel in visible_landmarks if sel)} GA-selected',
                               fontsize=9, fontweight='bold')
        self.ax_sensor.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='This Robot'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Communicating Robot'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Visible Robot'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gold', markersize=8, label='GA-Selected Landmark'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=8, label='Visible Landmark'),
        ]
        self.ax_sensor.legend(handles=legend_elements, loc='upper right', fontsize=7)
        
        if self.step_by_step:
            self.running = False
            self.run_cb.setChecked(False)

    def update_plot(self):
        # Initialize figure if needed or recreate if layout changed
        needs_recreate = False
        if self.fig is None:
            needs_recreate = True
        # Sensor view disabled
        # elif self.show_robot_sensor_view and self.ax_sensor is None:
        #     needs_recreate = True
        # elif not self.show_robot_sensor_view and self.ax_sensor is not None:
        #     needs_recreate = True
        elif self.use_ga_landmark_selection and self.ax_ga is None:
            needs_recreate = True
        elif not self.use_ga_landmark_selection and self.ax_ga is not None:
            needs_recreate = True
        # Federated learning plot is in separate tab, no need to recreate main figure
        
        if needs_recreate:
            old_fig = self.fig
            if old_fig is not None:
                plt.close(old_fig)
            
            # Sensor view disabled - simplified layout
            # if self.show_robot_sensor_view:
            #     # With sensor view: 2x2 grid, main plot, GPB, sensor view, and optionally GA
            #     if self.use_ga_landmark_selection:
            #         self.fig = plt.figure(figsize=(20, 10))
            #         gs = self.fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], width_ratios=[2, 1], 
            #                                   hspace=0.3, wspace=0.3, 
            #                                   left=0.05, right=0.95, top=0.95, bottom=0.05)
            #         self.ax = self.fig.add_subplot(gs[0, 0])
            #         self.ax_gpb = self.fig.add_subplot(gs[1, 0])
            #         self.ax_ga = self.fig.add_subplot(gs[2, 0])
            #         self.ax_sensor = self.fig.add_subplot(gs[0:3, 1])
            #     else:
            #         self.fig = plt.figure(figsize=(18, 10))
            #         gs = self.fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[2, 1], 
            #                                   hspace=0.3, wspace=0.3, 
            #                                   left=0.05, right=0.95, top=0.95, bottom=0.05)
            #         self.ax = self.fig.add_subplot(gs[0, 0])
            #         self.ax_gpb = self.fig.add_subplot(gs[1, 0])
            #         self.ax_ga = None
            #         self.ax_sensor = self.fig.add_subplot(gs[0:2, 1])
            #     if self.debug_console:
            #         print(f"[Sensor View] Created sensor view panel")
            # Simplified layout - no sensor view
            # Without sensor view: 2 or 3 rows depending on GA
            if self.use_ga_landmark_selection:
                self.fig = plt.figure(figsize=(14, 12))
                gs = self.fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3,
                                      left=0.05, right=0.70, top=0.95, bottom=0.05)
                self.ax = self.fig.add_subplot(gs[0, 0])
                self.ax_gpb = self.fig.add_subplot(gs[1, 0])
                self.ax_ga = self.fig.add_subplot(gs[2, 0])
                self.ax_sensor = None
                if self.debug_console:
                    print(f"[GA] Created GA evolution plot")
            else:
                self.fig = plt.figure(figsize=(14, 10))
                gs = self.fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3,
                                      left=0.05, right=0.70, top=0.95, bottom=0.05)
                self.ax = self.fig.add_subplot(gs[0, 0])
                self.ax_gpb = self.fig.add_subplot(gs[1, 0])
                self.ax_ga = None
                self.ax_sensor = None
                if self.debug_console:
                    print(f"[Layout] Created standard layout")
            
            # Update canvas - force immediate update
            if hasattr(self, 'canvas'):
                self.canvas.figure = self.fig
                self.canvas.draw()  # Force immediate draw when layout changes
                if self.debug_console:
                    print(f"[Sensor View] Canvas updated")
        
        self.ax.clear()
        
        # Determine view limits
        if self.follow_robot and 0 <= self.robot_id_to_follow < len(self.robots):
            center_robot = self.robots[self.robot_id_to_follow]
            center = center_robot.gt_pos  # Follow ground truth position (where robot actually is)
            view_range = self.boundary_radius + 2
            self.ax.set_xlim(center[0] - view_range, center[0] + view_range)
            self.ax.set_ylim(center[1] - view_range, center[1] + view_range)
        else:
            view_range = self.boundary_radius + 2
            self.ax.set_xlim(-view_range, view_range)
            self.ax.set_ylim(-view_range, view_range)
        
        # Draw coverage map FIRST (background layer) - shows where robots have been
        if self.show_coverage_map and self.coverage_grid is not None:
            try:
                x_min, x_max, y_min, y_max = self.coverage_grid_bounds
                extent = [x_min, x_max, y_min, y_max]
                # Use a colormap that shows visited areas (green/yellow for visited, transparent for unvisited)
                coverage_alpha = 0.25  # Transparency
                # Create a custom colormap: transparent for 0, green for visited
                from matplotlib.colors import LinearSegmentedColormap
                colors_list = [(0, 0, 0, 0), (0.2, 0.8, 0.2, 0.6), (0.4, 1.0, 0.4, 0.8)]  # Transparent -> Light green -> Bright green
                n_bins = 100
                cmap = LinearSegmentedColormap.from_list('coverage', colors_list, N=n_bins)
                im = self.ax.imshow(self.coverage_grid, extent=extent, origin='lower',
                                  cmap=cmap, alpha=coverage_alpha, vmin=0, vmax=1,
                                  interpolation='bilinear', zorder=0)
            except Exception as e:
                if self.debug_console:
                    print(f"[Coverage] Error drawing coverage map: {e}")
        
        # Draw maze walls (immediately after coverage map) so they're always visible
        if self.use_maze:
            # Always ensure maze is initialized when enabled
            if len(self.maze_walls) == 0:
                if self.debug_console:
                    print(f"[Maze] Maze enabled but no walls! Initializing...")
                self.init_maze()  # Reinitialize
                if self.debug_console and len(self.maze_walls) > 0:
                    print(f"[Maze] Initialized {len(self.maze_walls)} walls")
            
            # Draw walls if we have them
            if len(self.maze_walls) > 0:
                maze_wall_label_drawn = False
                for i, (wall_start, wall_end) in enumerate(self.maze_walls):
                    # Draw wall with thick black line
                    # Use zorder=3 to be above background/obstacles but below robots
                    # Use clip_on=False to ensure walls are always visible even at edges
                    try:
                        # Convert numpy arrays to lists for matplotlib compatibility
                        x_coords = [float(wall_start[0]), float(wall_end[0])]
                        y_coords = [float(wall_start[1]), float(wall_end[1])]
                        
                        # Draw wall with appropriate thickness
                        self.ax.plot(x_coords, y_coords, 
                                   'k-', linewidth=8, alpha=1.0, 
                                   label='Maze Wall' if not maze_wall_label_drawn else '',
                                   zorder=3, solid_capstyle='round', clip_on=False)
                        maze_wall_label_drawn = True
                    except Exception as e:
                        if self.debug_console:
                            print(f"[Maze Error] Failed to draw wall {i}: {e}")
                            import traceback
                            traceback.print_exc()
            else:
                # Debug: if still no walls, print warning
                if self.debug_console:
                    print(f"[Maze Error] use_maze=True but maze_walls is empty after init_maze()!")
                    print(f"[Maze Error] boundary_radius={self.boundary_radius}, use_maze={self.use_maze}")
                    print(f"[Maze Error] maze_walls type: {type(self.maze_walls)}, length: {len(self.maze_walls)}")
        
        # Draw boundary circle
        boundary_circle = plt.Circle(self.circle_center, self.boundary_radius, 
                                    fill=False, linestyle='--', color='red', 
                                    alpha=0.3, linewidth=2, label='Boundary', zorder=2)
        self.ax.add_patch(boundary_circle)
        
        # Draw paths (with limited length for cleaner visualization)
        # Draw GROUND TRUTH paths (where robots actually traveled) - these mark where robots have been
        if self.show_path:
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.robots)))
            for i, robot in enumerate(self.robots):
                if len(self.robot_gt_paths[robot.id]) > 1:
                    path = np.array(self.robot_gt_paths[robot.id])
                    # Show more path history for better coverage visualization
                    # Show last 500 points (or all if less than 500)
                    if len(path) > 500:
                        path = path[-500:]
                    self.ax.plot(path[:, 0], path[:, 1], '-', color=colors[i], 
                               alpha=0.6, linewidth=2.0, label=f"{robot.name} GT path", zorder=1)
        
        # Draw sensor ranges (visualize detection range) for all robots
        # Draw at ground truth position (where robot actually is)
        if self.show_factors:
            sensor_range_label_added = False
            for robot in self.robots:
                sensor_circle = plt.Circle(robot.gt_pos, self.sensor_range, 
                                         fill=False, linestyle=':', color='gray', 
                                         alpha=0.25, linewidth=1.5, zorder=2,
                                         label='Sensor Range' if not sensor_range_label_added else '')
                self.ax.add_patch(sensor_circle)
                sensor_range_label_added = True
        
        # Draw inter-robot communication (GPB messages) - GREEN lines like reference
        # Draw based on GROUND TRUTH positions (actual physical locations)
        inter_robot_connections_drawn = False
        if self.show_factors:
            # Draw connections based on current robot GROUND TRUTH positions and sensor range
            # Check all pairs of robots to see if they're within communication range
            drawn_pairs = set()  # Track which pairs we've already drawn to avoid duplicates
            for i, robot1 in enumerate(self.robots):
                if not robot1.is_active:
                    continue
                for j, robot2 in enumerate(self.robots):
                    if i >= j or not robot2.is_active:
                        continue
                    
                    # Check if robots are within sensor range (using ground truth positions)
                    dist_gt = np.linalg.norm(robot1.gt_pos - robot2.gt_pos)
                    if dist_gt < self.sensor_range:
                        # Create unique pair identifier
                        pair_id = tuple(sorted([robot1.id, robot2.id]))
                        if pair_id not in drawn_pairs:
                            drawn_pairs.add(pair_id)
                            
                            # Green solid lines for inter-robot communication (matching reference)
                            # Draw between ground truth positions (actual physical locations)
                            mid_x = (robot1.gt_pos[0] + robot2.gt_pos[0]) / 2
                            mid_y = (robot1.gt_pos[1] + robot2.gt_pos[1]) / 2
                            self.ax.plot([robot1.gt_pos[0], robot2.gt_pos[0]],
                                       [robot1.gt_pos[1], robot2.gt_pos[1]],
                                       'g-', alpha=0.7, linewidth=2.5,
                                       label='Inter-robot Communication' if not inter_robot_connections_drawn else '', zorder=4)
                            # Add small arrow in middle to show communication direction
                            dx = robot2.gt_pos[0] - robot1.gt_pos[0]
                            dy = robot2.gt_pos[1] - robot1.gt_pos[1]
                            if dist_gt > 0.5:
                                arrow_dx = dx / dist_gt * 0.4
                                arrow_dy = dy / dist_gt * 0.4
                                self.ax.arrow(mid_x - arrow_dx/2, mid_y - arrow_dy/2,
                                            arrow_dx, arrow_dy,
                                            head_width=0.25, head_length=0.2,
                                            fc='green', ec='green', alpha=0.8, zorder=5)
                            inter_robot_connections_drawn = True
            
            # Also show connections from inbox (current messages)
            # Draw from GROUND TRUTH positions (where robots actually are)
            for i, robot in enumerate(self.robots):
                if self.show_only_landmark_factors:
                    # Only show landmark connections
                    for neighbor_id in robot.inbox.keys():
                        if neighbor_id.startswith('landmark_'):
                            lm_idx = int(neighbor_id.split('_')[1])
                            if lm_idx < len(self.landmarks):
                                landmark_pos = self.landmarks[lm_idx]
                                self.ax.plot([robot.gt_pos[0], landmark_pos[0]],
                                           [robot.gt_pos[1], landmark_pos[1]],
                                           'r--', alpha=0.3, linewidth=0.8)
                else:
                    # Show range-bearing measurements (red lines) - the actual observations
                    for neighbor_id in robot.inbox.keys():
                        if neighbor_id.startswith('landmark_'):
                            lm_idx = int(neighbor_id.split('_')[1])
                            if lm_idx < len(self.landmarks):
                                landmark_pos = self.landmarks[lm_idx]
                                self.ax.plot([robot.gt_pos[0], landmark_pos[0]],
                                           [robot.gt_pos[1], landmark_pos[1]],
                                           'r--', alpha=0.4, linewidth=1.0,
                                           label='Range-Bearing Measurements' if i == 0 and lm_idx == 0 else '')
            
            # Visualize message queues (show robots with queued messages)
            # Draw at GROUND TRUTH positions (where robots actually are)
            for robot in self.robots:
                queue_size = self.message_queue_sizes.get(robot.id, 0)
                if queue_size > 0:
                    # Draw a small indicator showing queued messages
                    self.ax.plot(robot.gt_pos[0] + 1.2, robot.gt_pos[1] + 1.2, 'ro', 
                               markersize=8 + queue_size * 2, alpha=0.5, zorder=8)
                    self.ax.text(robot.gt_pos[0] + 1.2, robot.gt_pos[1] + 1.2, 
                               f'Q:{queue_size}', fontsize=7, ha='center', va='center',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7), zorder=9)
        
        # Maze walls already drawn above (right after clear)
        
        # SLAM Visualization: Draw factor graph (observation connections)
        if self.show_slam_visualization:
            # Draw grey lines from historical robot poses to landmarks (factor graph edges)
            for robot_id, observations in self.slam_observations.items():
                for pose, landmark_idx, obs_iter in observations:
                    if landmark_idx < len(self.landmarks):
                        landmark_pos = self.landmarks[landmark_idx]
                        # Draw grey line (factor graph edge) - use alpha based on recency
                        # More recent observations are more opaque
                        age = self.iter_count - obs_iter
                        alpha = max(0.1, 1.0 - age / 200.0)  # Fade out old observations
                        self.ax.plot([pose[0], landmark_pos[0]], [pose[1], landmark_pos[1]],
                                   'grey', linewidth=0.5, alpha=alpha, zorder=1)
            
            # Draw ground truth pose history as yellow dots
            for robot in self.robots:
                if robot.id in self.robot_gt_paths and len(self.robot_gt_paths[robot.id]) > 0:
                    path = np.array(self.robot_gt_paths[robot.id])
                    # Show last 200 poses as yellow dots
                    if len(path) > 200:
                        path = path[-200:]
                    self.ax.plot(path[:, 0], path[:, 1], 'yo', markersize=2, alpha=0.6, zorder=2,
                               label='GT Poses' if robot is self.robots[0] else '')
        
        # Draw landmarks (white dots for SLAM visualization, or black squares if SLAM disabled)
        # First, draw connections from robots to their GA-selected landmarks
        # Draw from GROUND TRUTH positions (where robots actually are)
        if self.use_ga_landmark_selection:
            for robot in self.robots:
                if robot.id in self.robot_selected_landmarks:
                    selected_indices = self.robot_selected_landmarks[robot.id]
                    for lm_idx in selected_indices:
                        if lm_idx < len(self.landmarks):
                            landmark_pos = self.landmarks[lm_idx]
                            # Draw thick orange line from robot to GA-selected landmark
                            self.ax.plot([robot.gt_pos[0], landmark_pos[0]], 
                                       [robot.gt_pos[1], landmark_pos[1]], 
                                       'o-', color='orange', linewidth=3, alpha=0.6,
                                       markersize=0, zorder=9,
                                       label='GA-Selected Landmark Connection' if robot is self.robots[0] and lm_idx == selected_indices[0] else '')
        
        for lm_idx, landmark_pos in enumerate(self.landmarks[:self.num_landmarks]):
            # Highlight GA-selected landmarks
            is_ga_selected = False
            selected_by_robots = []
            if self.use_ga_landmark_selection:
                for robot in self.robots:
                    if robot.id in self.robot_selected_landmarks:
                        if lm_idx in self.robot_selected_landmarks[robot.id]:
                            is_ga_selected = True
                            selected_by_robots.append(robot)
                            break
            
            # For SLAM visualization: use white dots for landmarks (ground truth positions)
            # Otherwise use colored squares
            if self.show_slam_visualization:
                # White dots for landmarks (ground truth positions)
                marker_color = 'white'
                marker_edge = 'black'
                marker_size = 8
                marker_style = 'o'  # Circle
            else:
                # Use different color for GA-selected landmarks (original style)
                if is_ga_selected:
                    marker_color = 'gold'
                    marker_edge = 'orange'
                    marker_size = 15  # Larger for GA-selected
                else:
                    marker_color = 'black'
                    marker_edge = 'white'
                    marker_size = 12
                marker_style = 's'  # Square
            
            label = "Landmark" if lm_idx == 0 else None
            self.ax.plot(landmark_pos[0], landmark_pos[1], marker_style, 
                        markersize=marker_size, markeredgewidth=2, 
                        markerfacecolor=marker_color, markeredgecolor=marker_edge,
                        label=label, zorder=10)
            # Add landmark number with GA indicator
            label_text = f'L{lm_idx+1}'
            if is_ga_selected and not self.show_slam_visualization:
                label_text += ' (GA)'
            text_color = 'black' if self.show_slam_visualization else ('darkorange' if is_ga_selected else 'black')
            self.ax.text(landmark_pos[0] + 0.5, landmark_pos[1] + 0.5, 
                        label_text, fontsize=8, fontweight='bold' if is_ga_selected else 'normal',
                        color=text_color,
                        bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white' if self.show_slam_visualization else ('yellow' if is_ga_selected else 'white'), 
                                alpha=0.8, edgecolor='gray'),
                        zorder=11)
        
        # Draw obstacles
        for obs_pos in self.obstacles:
            obstacle_circle = MplCircle(obs_pos, self.obstacle_radius, 
                                      fill=True, facecolor='red', alpha=0.5, 
                                      edgecolor='darkred', linewidth=2,
                                      label='Obstacle' if obs_pos is self.obstacles[0] else None,
                                      zorder=3)
            self.ax.add_patch(obstacle_circle)
        
        # Draw robots (differential drive - Roomba-like, and Drones)
        # Draw at GROUND TRUTH position (where robot actually is)
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.robots)))
        for i, robot in enumerate(self.robots):
            # Determine robot type for visualization
            is_drone = (robot.robot_type == 'drone' or 'Drone' in robot.name)
            
            if is_drone:
                # Drones: triangles (upward-pointing) to distinguish from ground robots
                marker = '^'  # Triangle marker
                marker_size = 12
                edge_color = 'darkblue'
                edge_width = 2
                robot_label = f"Drone {i+1}" if i == 0 or robot is self.robots[0] else None
            else:
                # Roomba/Differential: circles
                marker = 'o'  # Circle marker
                marker_size = 10
                edge_color = 'black'
                edge_width = 1.5
                robot_label = f"Roomba {i+1}" if i == 0 or robot is self.robots[0] else None
            
            # Draw robot at ground truth position (actual physical location)
            self.ax.plot(robot.gt_pos[0], robot.gt_pos[1], marker, color=colors[i], 
                        markersize=marker_size, markeredgecolor=edge_color, markeredgewidth=edge_width,
                        label=robot_label, zorder=5)
            
            # Draw orientation arrow based on ground truth angle
            arrow_length = 0.8
            dx = arrow_length * np.cos(robot.gt_angle)
            dy = arrow_length * np.sin(robot.gt_angle)
            # Use different arrow style for drones
            arrow_style = '->' if is_drone else '->'
            arrow_color = 'darkblue' if is_drone else colors[i]
            self.ax.arrow(robot.gt_pos[0], robot.gt_pos[1], dx, dy,
                        head_width=0.3, head_length=0.2, fc=arrow_color, ec=edge_color, 
                        zorder=6, linestyle='-' if is_drone else '-')
            
            # Optionally draw estimated position to show localization error
            if self.show_factors:
                # Draw estimated position (mu) - use same marker type as robot
                est_marker = '^' if is_drone else 'o'
                est_size = 6 if is_drone else 6
                self.ax.plot(robot.mu[0], robot.mu[1], est_marker, color=colors[i], 
                            markersize=est_size, alpha=0.5, zorder=4)
                # Draw line from estimated to ground truth to show error
                self.ax.plot([robot.mu[0], robot.gt_pos[0]], [robot.mu[1], robot.gt_pos[1]], 
                            '--', color=colors[i], linewidth=1, alpha=0.3, zorder=3)
        
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Update title to reflect robot types
        has_drones = any(r.robot_type == 'drone' or 'Drone' in r.name for r in self.robots)
        has_roomba = any(r.robot_type == 'differential' or 'Roomba' in r.name for r in self.robots)
        if has_drones and has_roomba:
            title = "Multi-Robot System (Roomba + Drones)"
        elif has_drones:
            title = "Drone Swarm"
        else:
            title = "Differential Drive Robots (Roomba)"
        
        if self.use_rvo:
            title += " with RVO"
        title += " & GPB"
        if self.dropped_messages_count > 0:
            title += f" | Dropped: {self.dropped_messages_count}"
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        # Legend showing inter-robot communication and other key elements
        # Position legend outside plot area to the right
        handles, labels = self.ax.get_legend_handles_labels()
        # Remove duplicates while preserving order
        seen = set()
        unique_handles = []
        unique_labels = []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                unique_handles.append(h)
                unique_labels.append(l)
        if unique_handles:
            # Position legend outside plot area to the right
            self.ax.legend(unique_handles, unique_labels, 
                          loc='center left', bbox_to_anchor=(1.02, 0.5),
                          fontsize=9, framealpha=0.9, fancybox=True, shadow=True)
        
        # Add author footer in bottom right corner
        footer_text = "For educational purposes\n"
        footer_text += "Author: Thiwanka Jayasiri\n"
        footer_text += "Ref: Distributed GBP for Multi-Robot SLAM (arXiv:2202.03314)"
        self.ax.text(0.99, 0.01, footer_text, 
                    transform=self.ax.transAxes,
                    fontsize=7, 
                    ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'),
                    zorder=100)
        
        # Update GPB performance plot
        self.ax_gpb.clear()
        if len(self.gpb_error_history) > 1:
            iterations = list(range(len(self.gpb_error_history)))
            
            # Plot individual robot errors (lighter, thinner lines)
            colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(self.robots))))
            individual_lines_drawn = False
            for i, robot in enumerate(self.robots):
                if robot.id in self.gpb_robot_errors and len(self.gpb_robot_errors[robot.id]) > 1:
                    robot_iterations = list(range(len(self.gpb_robot_errors[robot.id])))
                    # Plot individual robot error - don't require exact length match
                    # Just plot what we have (they'll align over time)
                    self.ax_gpb.plot(robot_iterations, self.gpb_robot_errors[robot.id], 
                                   '-', color=colors[i], linewidth=1, alpha=0.5,
                                   label=f'Robot {i+1}', zorder=5)
                    individual_lines_drawn = True
            
            # Plot average error (thicker, more prominent line)
            self.ax_gpb.plot(iterations, self.gpb_error_history, 'b-', linewidth=2.5, 
                           label='Average Error', zorder=10)
            
            self.ax_gpb.set_xlabel("GBP Iteration", fontsize=10)
            self.ax_gpb.set_ylabel("Position Error", fontsize=10)
            self.ax_gpb.set_title("GPB Performance: Individual & Average Position Errors", fontsize=10, fontweight='bold')
            self.ax_gpb.grid(True, alpha=0.3)
            # Position legend outside plot area to the right
            self.ax_gpb.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                              fontsize=7, framealpha=0.9, fancybox=True, shadow=True)
        
        # Add current error as text
        if self.gpb_error_history:
            current_error = self.gpb_error_history[-1]
            # Also show min/max
            if self.last_update_stats:
                current_errors = [stats['error'] for stats in self.last_update_stats.values()]
                min_err = min(current_errors)
                max_err = max(current_errors)
                error_text = f'Avg: {current_error:.3f} | Min: {min_err:.3f} | Max: {max_err:.3f}'
            else:
                error_text = f'Current Error: {current_error:.3f}'
            self.ax_gpb.text(0.02, 0.98, error_text, 
                           transform=self.ax_gpb.transAxes, fontsize=8,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Show convergence trend (improved logic)
        if len(self.gpb_error_history) > 30:
            # Use last 30 points for more stable trend detection
            recent = self.gpb_error_history[-30:]
            # Calculate trend: compare last 10 vs previous 10
            recent_10 = recent[-10:]
            previous_10 = recent[-20:-10] if len(recent) >= 20 else recent[:10]
            
            avg_recent = np.mean(recent_10)
            avg_previous = np.mean(previous_10) if len(previous_10) > 0 else avg_recent
            
            # Consider converging if error decreased by at least 5%
            if avg_recent < avg_previous * 0.95:
                trend = "Converging ✓"
                color = 'green'
            elif avg_recent > avg_previous * 1.05:
                trend = "Diverging ✗"
                color = 'red'
            else:
                # Stable - show as converging if error is low
                if avg_recent < 0.5:
                    trend = "Stable ✓"
                    color = 'blue'
                else:
                    trend = "Stable"
                    color = 'gray'
            
            self.ax_gpb.text(0.98, 0.98, trend, transform=self.ax_gpb.transAxes, 
                           fontsize=9, color=color, fontweight='bold',
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Update GA evolution plot if enabled
        if self.use_ga_landmark_selection and self.ax_ga is not None:
            self.ax_ga.clear()
            has_data = False
            colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(self.robots))))
            
            for i, robot in enumerate(self.robots):
                if robot.id in self.ga_evolution_data and len(self.ga_evolution_data[robot.id]) > 0:
                    has_data = True
                    generations = list(range(len(self.ga_evolution_data[robot.id])))
                    best_fitness = [gen['best'] for gen in self.ga_evolution_data[robot.id]]
                    avg_fitness = [gen['avg'] for gen in self.ga_evolution_data[robot.id]]
                    
                    # Plot best and average fitness
                    self.ax_ga.plot(generations, best_fitness, '-', color=colors[i], 
                                   linewidth=2, label=f'Robot {i+1} Best', alpha=0.8)
                    self.ax_ga.plot(generations, avg_fitness, '--', color=colors[i], 
                                   linewidth=1, label=f'Robot {i+1} Avg', alpha=0.5)
            
            if has_data:
                self.ax_ga.set_xlabel("Generation", fontsize=10)
                self.ax_ga.set_ylabel("Fitness", fontsize=10)
                self.ax_ga.set_title("GA Evolution: Fitness Over Generations", fontsize=10, fontweight='bold')
                self.ax_ga.grid(True, alpha=0.3)
                self.ax_ga.legend(loc='best', fontsize=7, framealpha=0.9, fancybox=True, shadow=True)
            else:
                self.ax_ga.text(0.5, 0.5, 'No GA evolution data yet', 
                               transform=self.ax_ga.transAxes,
                               ha='center', va='center', fontsize=10, alpha=0.5)
                self.ax_ga.set_xlabel("Generation", fontsize=10)
                self.ax_ga.set_ylabel("Fitness", fontsize=10)
                self.ax_ga.set_title("GA Evolution: Fitness Over Generations", fontsize=10, fontweight='bold')
                self.ax_ga.grid(True, alpha=0.3)
        
        # Update Federated Learning plot in separate tab
        self._update_federated_plot()
        
        # Add author footer to GPB plot as well
        footer_text_gpb = "For educational purposes | Author: Thiwanka Jayasiri | Ref: arXiv:2202.03314"
        self.ax_gpb.text(0.99, 0.01, footer_text_gpb, 
                         transform=self.ax_gpb.transAxes,
                         fontsize=6, 
                         ha='right', va='bottom',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'),
                         zorder=100)
        
        self.canvas.draw()
        
        # Sensor view disabled - not working properly
        # if self.show_robot_sensor_view and self.ax_sensor is not None:
        #     try:
        #         self.draw_robot_sensor_view()
        #         # Redraw canvas to show sensor view
        #         self.canvas.draw()
        #     except Exception as e:
        #         if self.debug_console:
        #             print(f"[Sensor View Error] {e}")
        #             import traceback
        #             traceback.print_exc()
    
    def update_status_display(self):
        """Update the status display in the control panel."""
        active_robots = sum(1 for r in self.robots if len(r.inbox) > 0)
        total_connections = len(self.active_connections) * 2  # Bidirectional
        total_queue_size = sum(self.message_queue_sizes.values())
        
        # Count neighbor messages per robot (excluding odometry and landmarks)
        robot_neighbor_counts = {}
        for robot in self.robots:
            if robot.is_active:
                neighbor_msgs = len([k for k in robot.inbox.keys() 
                                    if not k.startswith('landmark_') and not k.startswith('odometry')])
                robot_neighbor_counts[robot.name] = neighbor_msgs
        
        # Calculate average position error
        if self.last_update_stats:
            avg_error = np.mean([stats['error'] for stats in self.last_update_stats.values()])
            max_error = max([stats['error'] for stats in self.last_update_stats.values()])
        else:
            avg_error = 0.0
            max_error = 0.0
        
        # Update status labels
        status_text = f"Status: Running\n"
        status_text += f"Active: {active_robots}/{len(self.robots)} robots\n"
        status_text += f"Connections: {total_connections}\n"
        if total_queue_size > 0:
            status_text += f"Queued: {total_queue_size} msgs\n"
        # Show neighbor message counts
        if robot_neighbor_counts:
            status_text += f"Neighbor msgs:\n"
            for name, count in list(robot_neighbor_counts.items())[:3]:  # Show first 3
                status_text += f"  {name}: {count}\n"
            if len(robot_neighbor_counts) > 3:
                status_text += f"  ..."
        self.status_label.setText(status_text)
        
        msg_text = f"Total Messages: {self.total_messages} (Last: {self.messages_per_iteration})"
        if self.dropped_messages_count > 0:
            msg_text += f"\nDropped: {self.dropped_messages_count}"
        self.messages_label.setText(msg_text)
        self.iterations_label.setText(f"Iterations: {self.iter_count}")
        
        robots_text = f"Robots: {len(self.robots)} active\n"
        robots_text += f"Avg Error: {avg_error:.3f}\n"
        robots_text += f"Max Error: {max_error:.3f}"
        if self.async_communication:
            robots_text += f"\nPacket Loss: {self.communication_drop_rate*100:.1f}%"
        self.robots_status_label.setText(robots_text)
    
    def _update_federated_plot(self):
        """Update the federated learning plot in the separate tab with sleek, professional plots."""
        # Always update the plot (even if disabled, to show status)
        if not hasattr(self, 'ax_fl_fitness') or self.ax_fl_fitness is None:
            # Try to initialize if not exists
            if hasattr(self, 'fig_fl') and self.fig_fl is not None:
                try:
                    gs_fl = self.fig_fl.add_gridspec(2, 2, hspace=0.35, wspace=0.3, 
                                                    left=0.08, right=0.95, top=0.93, bottom=0.08)
                    self.ax_fl_fitness = self.fig_fl.add_subplot(gs_fl[0, 0])
                    self.ax_fl_weights = self.fig_fl.add_subplot(gs_fl[0, 1])
                    self.ax_fl_metrics = self.fig_fl.add_subplot(gs_fl[1, 0])
                    self.ax_fl_clients = self.fig_fl.add_subplot(gs_fl[1, 1])
                except:
                    return
            else:
                return
        
        if not self.use_federated_learning:
            if hasattr(self, 'fl_status_label'):
                self.fl_status_label.setText("Federated Learning: Disabled")
            # Clear all subplots and show disabled message
            self.ax_fl_fitness.clear()
            self.ax_fl_weights.clear()
            self.ax_fl_metrics.clear()
            self.ax_fl_clients.clear()
            
            # Show message in first subplot
            self.ax_fl_fitness.text(0.5, 0.5, 'Federated Learning is Disabled\n\n'
                                         'Enable "Enable-FL(GA)" in settings to start.',
                                     transform=self.ax_fl_fitness.transAxes,
                                     ha='center', va='center', fontsize=12, style='italic',
                                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            self.ax_fl_fitness.set_title("Federated Learning (GA-Optimized)", 
                                        fontsize=14, fontweight='bold', pad=15)
            self.canvas_fl.draw()
            return
        
        # Clear all subplots
        self.ax_fl_fitness.clear()
        self.ax_fl_weights.clear()
        self.ax_fl_metrics.clear()
        self.ax_fl_clients.clear()
        
        # Update status to show it's enabled
        if hasattr(self, 'fl_status_label'):
            if len(self.federated_fitness_history) == 0:
                next_round_iter = ((self.iter_count // 20) + 1) * 20
                self.fl_status_label.setText(
                    f"Enabled | Next round at iteration {next_round_iter} (current: {self.iter_count}) | GA: {'Ready' if self.federated_ga is not None else 'Initializing...'}"
                )
            else:
                latest_round = self.federated_fitness_history[-1]['round']
                latest_fitness = self.federated_fitness_history[-1]['best']
                if len(self.federated_client_selection_history) > 0:
                    latest_selection = self.federated_client_selection_history[-1]
                    selected_count = len(latest_selection['selected'])
                    self.fl_status_label.setText(
                        f"Round {latest_round} | Fitness: {latest_fitness:.4f} | "
                        f"Selected: {selected_count}/{len(self.robots)} clients"
                    )
                else:
                    self.fl_status_label.setText(
                        f"Round {latest_round} | Fitness: {latest_fitness:.4f}"
                    )
        
        if len(self.federated_fitness_history) > 0:
            rounds = [h['round'] for h in self.federated_fitness_history]
            best_fitness = [h['best'] for h in self.federated_fitness_history]
            avg_fitness = [h['avg'] for h in self.federated_fitness_history]
            
            # ===== SUBPLOT 1: GA Fitness Evolution =====
            self.ax_fl_fitness.plot(rounds, best_fitness, 'o-', color='#2E86AB', linewidth=2.5, 
                                   markersize=6, label='Best Fitness', alpha=0.9, zorder=3)
            self.ax_fl_fitness.plot(rounds, avg_fitness, 's--', color='#A23B72', linewidth=2, 
                                   markersize=5, label='Avg Fitness', alpha=0.8, zorder=2)
            self.ax_fl_fitness.set_xlabel("Federated Round", fontsize=11, fontweight='bold')
            self.ax_fl_fitness.set_ylabel("Fitness Score", fontsize=11, fontweight='bold')
            self.ax_fl_fitness.set_title("GA Fitness Evolution", fontsize=12, fontweight='bold', pad=10)
            self.ax_fl_fitness.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            self.ax_fl_fitness.legend(loc='best', fontsize=9, framealpha=0.95, fancybox=True, shadow=True)
            self.ax_fl_fitness.set_facecolor('#FAFAFA')
            
            # ===== SUBPLOT 2: Aggregation Weights Over Time =====
            if len(self.federated_aggregation_weights_history) > 0:
                # Get all unique robot types
                all_types = set()
                for w_dict in self.federated_aggregation_weights_history:
                    all_types.update(w_dict['weights'].keys())
                all_types = sorted(list(all_types))
                
                # Plot weight evolution for each type
                colors_weights = plt.cm.Set2(np.linspace(0, 1, len(all_types)))
                for i, robot_type in enumerate(all_types):
                    weight_history = []
                    weight_rounds = []
                    for w_entry in self.federated_aggregation_weights_history:
                        if robot_type in w_entry['weights']:
                            weight_history.append(w_entry['weights'][robot_type])
                            weight_rounds.append(w_entry['round'])
                    
                    if len(weight_history) > 0:
                        type_label = robot_type.replace('differential', 'Roomba').replace('drone', 'Drone').title()
                        self.ax_fl_weights.plot(weight_rounds, weight_history, 'o-', 
                                               color=colors_weights[i], linewidth=2.5, 
                                               markersize=6, label=type_label, alpha=0.9)
                
                self.ax_fl_weights.set_xlabel("Federated Round", fontsize=11, fontweight='bold')
                self.ax_fl_weights.set_ylabel("Aggregation Weight", fontsize=11, fontweight='bold')
                self.ax_fl_weights.set_title("Aggregation Weights (Translation Layer)", 
                                            fontsize=12, fontweight='bold', pad=10)
                self.ax_fl_weights.set_ylim(0, 1.0)
                self.ax_fl_weights.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
                self.ax_fl_weights.legend(loc='best', fontsize=9, framealpha=0.95, fancybox=True, shadow=True)
                self.ax_fl_weights.set_facecolor('#FAFAFA')
            else:
                self.ax_fl_weights.text(0.5, 0.5, 'No weight data yet', 
                                       transform=self.ax_fl_weights.transAxes,
                                       ha='center', va='center', fontsize=11, style='italic')
                self.ax_fl_weights.set_title("Aggregation Weights", fontsize=12, fontweight='bold')
            
            # ===== SUBPLOT 3: Metrics (Diversity, Fairness, Quality) =====
            if len(self.federated_metrics_history) > 0:
                metric_rounds = [m['round'] for m in self.federated_metrics_history]
                diversity = [m['diversity'] for m in self.federated_metrics_history]
                fairness = [m['fairness'] for m in self.federated_metrics_history]
                quality = [m['quality'] for m in self.federated_metrics_history]
                
                self.ax_fl_metrics.plot(metric_rounds, diversity, 'o-', color='#06A77D', 
                                       linewidth=2.5, markersize=6, label='Diversity', alpha=0.9)
                self.ax_fl_metrics.plot(metric_rounds, fairness, 's-', color='#F18F01', 
                                       linewidth=2.5, markersize=6, label='Fairness', alpha=0.9)
                self.ax_fl_metrics.plot(metric_rounds, quality, '^-', color='#C73E1D', 
                                       linewidth=2.5, markersize=6, label='Quality', alpha=0.9)
                
                self.ax_fl_metrics.set_xlabel("Federated Round", fontsize=11, fontweight='bold')
                self.ax_fl_metrics.set_ylabel("Metric Value", fontsize=11, fontweight='bold')
                self.ax_fl_metrics.set_title("Non-IID Handling Metrics", fontsize=12, fontweight='bold', pad=10)
                self.ax_fl_metrics.set_ylim(0, 1.05)
                self.ax_fl_metrics.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
                self.ax_fl_metrics.legend(loc='best', fontsize=9, framealpha=0.95, fancybox=True, shadow=True)
                self.ax_fl_metrics.set_facecolor('#FAFAFA')
            else:
                self.ax_fl_metrics.text(0.5, 0.5, 'No metrics data yet', 
                                       transform=self.ax_fl_metrics.transAxes,
                                       ha='center', va='center', fontsize=11, style='italic')
                self.ax_fl_metrics.set_title("Non-IID Handling Metrics", fontsize=12, fontweight='bold')
            
            # ===== SUBPLOT 4: Client Selection Over Rounds =====
            if len(self.federated_client_selection_history) > 0:
                # Better visualization: Show selection count and individual robot selection status
                rounds = [s['round'] for s in self.federated_client_selection_history]
                num_selected = [len(s['selected']) for s in self.federated_client_selection_history]
                num_robots = len(self.robots)
                
                # Create two y-axes: one for count, one for individual robot status
                ax_count = self.ax_fl_clients
                
                # Plot 1: Number of selected clients as bars
                colors_bar = ['#06A77D' if n > num_robots/2 else '#F18F01' for n in num_selected]
                bars = ax_count.bar(rounds, num_selected, alpha=0.6, color=colors_bar, 
                                   edgecolor='black', linewidth=1.5, label='Selected Clients', zorder=2)
                ax_count.set_xlabel("Federated Round", fontsize=11, fontweight='bold')
                ax_count.set_ylabel("Number of Selected Clients", fontsize=11, fontweight='bold', color='#06A77D')
                ax_count.set_ylim(0, num_robots + 0.5)
                ax_count.set_yticks(range(num_robots + 1))
                ax_count.tick_params(axis='y', labelcolor='#06A77D')
                ax_count.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
                
                # Add value labels on bars
                for bar, count in zip(bars, num_selected):
                    height = bar.get_height()
                    ax_count.text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(count)}/{num_robots}',
                                ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                # Plot 2: Individual robot selection status as lines (secondary axis)
                ax_robots = ax_count.twinx()
                robot_colors = plt.cm.tab10(np.linspace(0, 1, num_robots))
                
                # For each robot, show selection status (1 = selected, 0 = not selected)
                for robot_idx in range(num_robots):
                    robot_selection = []
                    for sel_entry in self.federated_client_selection_history:
                        robot_selection.append(1.0 if robot_idx in sel_entry['selected'] else 0.0)
                    
                    # Get robot name and type for label
                    robot = self.robots[robot_idx] if robot_idx < len(self.robots) else None
                    robot_name = robot.name if robot else f'R{robot_idx+1}'
                    is_drone = robot and (robot.robot_type == 'drone' or 'Drone' in robot.name)
                    label = f"{robot_name} {'(Drone)' if is_drone else '(Roomba)'}"
                    
                    # Plot as step function for clarity
                    ax_robots.plot(rounds, robot_selection, 'o-', color=robot_colors[robot_idx],
                                 linewidth=2, markersize=5, label=label, alpha=0.7, zorder=1)
                
                ax_robots.set_ylabel("Selection Status", fontsize=11, fontweight='bold', color='#2E86AB')
                ax_robots.set_ylim(-0.1, 1.1)
                ax_robots.set_yticks([0, 1])
                ax_robots.set_yticklabels(['Not Selected', 'Selected'], fontsize=9)
                ax_robots.tick_params(axis='y', labelcolor='#2E86AB')
                
                self.ax_fl_clients.set_title("Client Selection Over Rounds", fontsize=12, fontweight='bold', pad=10)
                self.ax_fl_clients.set_facecolor('#FAFAFA')
                
                # Add legend for robot selection lines (compact)
                lines_robots, labels_robots = ax_robots.get_legend_handles_labels()
                if len(lines_robots) <= 6:  # Only show legend if not too many robots
                    ax_robots.legend(lines_robots, labels_robots, loc='upper right', 
                                    fontsize=7, framealpha=0.9, ncol=1)
            else:
                self.ax_fl_clients.text(0.5, 0.5, 'No client selection data yet', 
                                       transform=self.ax_fl_clients.transAxes,
                                       ha='center', va='center', fontsize=11, style='italic')
                self.ax_fl_clients.set_title("Client Selection Over Rounds", fontsize=12, fontweight='bold')
        else:
            # No data yet - show message in first subplot
            next_round_iter = ((self.iter_count // 20) + 1) * 20
            ga_status = 'Ready' if self.federated_ga is not None else 'Initializing...'
            self.ax_fl_fitness.text(0.5, 0.5, 'Waiting for federated learning data...\n\n'
                                         f'Federated Learning: ENABLED ✓\n'
                                         f'GA Optimizer: {ga_status}\n'
                                         f'Next round: iteration {next_round_iter}\n'
                                         f'Current: {self.iter_count} (runs every 20 iterations)\n\n'
                                         f'The system will collect updates from robots\n'
                                         f'and use GA to optimize aggregation.',
                                     transform=self.ax_fl_fitness.transAxes,
                                     ha='center', va='center', fontsize=11, style='italic',
                                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            self.ax_fl_fitness.set_title("Federated Learning (GA-Optimized)", 
                                        fontsize=14, fontweight='bold', pad=15)
            self.ax_fl_fitness.set_xlim(0, 10)
            self.ax_fl_fitness.set_ylim(0, 1)
            
            # Show empty plots with labels and proper axes
            self.ax_fl_weights.set_title("Aggregation Weights (Translation Layer)", fontsize=12, fontweight='bold')
            self.ax_fl_weights.set_xlabel("Federated Round", fontsize=11, fontweight='bold')
            self.ax_fl_weights.set_ylabel("Aggregation Weight", fontsize=11, fontweight='bold')
            self.ax_fl_weights.set_xlim(0, 10)
            self.ax_fl_weights.set_ylim(0, 1)
            self.ax_fl_weights.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            self.ax_fl_weights.text(0.5, 0.5, 'Waiting for data...', 
                                   transform=self.ax_fl_weights.transAxes,
                                   ha='center', va='center', fontsize=10, style='italic')
            
            self.ax_fl_metrics.set_title("Non-IID Handling Metrics", fontsize=12, fontweight='bold')
            self.ax_fl_metrics.set_xlabel("Federated Round", fontsize=11, fontweight='bold')
            self.ax_fl_metrics.set_ylabel("Metric Value", fontsize=11, fontweight='bold')
            self.ax_fl_metrics.set_xlim(0, 10)
            self.ax_fl_metrics.set_ylim(0, 1)
            self.ax_fl_metrics.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            self.ax_fl_metrics.text(0.5, 0.5, 'Waiting for data...', 
                                   transform=self.ax_fl_metrics.transAxes,
                                   ha='center', va='center', fontsize=10, style='italic')
            
            self.ax_fl_clients.set_title("Client Selection Over Rounds", fontsize=12, fontweight='bold')
            self.ax_fl_clients.set_xlabel("Federated Round", fontsize=11, fontweight='bold')
            self.ax_fl_clients.set_ylabel("Number of Selected Clients", fontsize=11, fontweight='bold')
            self.ax_fl_clients.set_xlim(0, 10)
            self.ax_fl_clients.set_ylim(0, len(self.robots) + 0.5)
            self.ax_fl_clients.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            self.ax_fl_clients.text(0.5, 0.5, 'Waiting for data...', 
                                   transform=self.ax_fl_clients.transAxes,
                                   ha='center', va='center', fontsize=10, style='italic')
        
        # Always update the federated learning canvas
        if hasattr(self, 'canvas_fl'):
            self.canvas_fl.draw()

    def closeEvent(self, event):
        """Handle window close event - safely stop simulation."""
        if self.is_shutting_down:
            if event:
                event.accept()
            return
        
        self.is_shutting_down = True
        print("\n" + "="*60)
        print("[Shutdown] Stopping simulation...")
        self.running = False
        
        if hasattr(self, 'timer'):
            self.timer.stop()
            print("[Shutdown] Timer stopped.")
        
        print("[Shutdown] Cleaning up resources...")
        
        # Close matplotlib figure
        if hasattr(self, 'fig'):
            plt.close(self.fig)
            print("[Shutdown] Matplotlib figure closed.")
        
        print("[Shutdown] Final Statistics:")
        print(f"  - Total iterations: {self.iter_count}")
        print(f"  - Total messages exchanged: {self.total_messages}")
        print(f"  - Average messages per iteration: {self.total_messages / max(self.iter_count, 1):.2f}")
        print("[Shutdown] Simulation closed safely.")
        print("="*60)
        
        if event:
            event.accept()
    
    def shutdown(self):
        """Public method to safely shutdown the simulation."""
        self.closeEvent(None)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RobotWebGUI()
    window.show()
    
    # Handle Ctrl+C gracefully
    import signal
    def signal_handler(sig, frame):
        print("\n[Interrupt] Received interrupt signal, shutting down...")
        window.shutdown()
        app.quit()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("\n[Interrupt] Keyboard interrupt, shutting down...")
        window.shutdown()
        sys.exit(0)