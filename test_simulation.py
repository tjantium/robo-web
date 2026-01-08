import sys
import os
from typing import Any
# Set Qt API before importing matplotlib
os.environ['QT_API'] = 'pyqt6'
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QCheckBox, 
                             QDoubleSpinBox, QSpinBox, QGroupBox, QScrollArea)
from PyQt6.QtCore import Qt, QTimer
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Circle as MplCircle
from robot_m import RobotAgent
from rvo import RVO

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
        
        # Obstacle settings
        self.num_obstacles = 3  # Reduced for cleaner demo
        self.obstacle_radius = 1.2
        self.obstacles = []  # List of obstacle positions
        self.obstacle_velocities = []  # List of obstacle velocities
        self.obstacle_moving = True  # Whether obstacles move
        
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
        self.gpb_error_history = []  # Track position errors over time
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
        
        # 2. UI Layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # Left panel - Controls
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 0)
        
        # Right panel - Visualization
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        
        # FPS label
        self.fps_label = QLabel("0 FPS")
        self.fps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        viz_layout.addWidget(self.fps_label)
        
        # Main simulation plot - leave space on right for legend
        self.fig = plt.figure(figsize=(14, 10))  # Wider figure to accommodate legend
        gs = self.fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3, 
                                   left=0.08, right=0.72, top=0.95, bottom=0.05)
        
        # Top: Main simulation canvas
        self.ax = self.fig.add_subplot(gs[0])
        
        # Bottom: GPB performance plot
        self.ax_gpb = self.fig.add_subplot(gs[1])
        self.ax_gpb.set_xlabel("Iteration")
        self.ax_gpb.set_ylabel("GPB Error")
        self.ax_gpb.set_title("GPB Performance: Average Position Error")
        self.ax_gpb.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvas(self.fig)
        viz_layout.addWidget(self.canvas)
        
        main_layout.addWidget(viz_widget, 1)
        
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
        """Initialize robots: differential robots in a circle, car-like robot away from center."""
        self.robots = []
        self.robot_paths = {}
        self.robot_gt_paths = {}
        
        # Initialize differential robots in a larger circle with better spacing
        for i in range(self.num_robots):
            angle = 2 * np.pi * i / self.num_robots
            # Initial position on circle - use larger radius for better spacing
            gt_pos = self.circle_center + self.circle_radius * np.array([np.cos(angle), np.sin(angle)])
            # Smaller noise for cleaner start
            initial_estimate = gt_pos + np.random.normal(0, 0.2, 2)
            
            robot = RobotAgent(f"DiffRobot{i+1}", initial_estimate, robot_type='differential')
            robot.gt_pos = gt_pos.copy()  # Store ground truth
            # Alternate direction: even robots clockwise, odd robots counter-clockwise
            robot.direction = 1 if i % 2 == 0 else -1  # 1 for clockwise, -1 for counter-clockwise
            robot.angle = angle + np.pi / 2 * robot.direction  # Face tangent to circle
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
    
    def create_control_panel(self):
        """Create the left control panel with settings."""
        panel = QWidget()
        panel.setMaximumWidth(300)
        panel.setStyleSheet("background-color: #2b2b2b; color: white;")
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # Settings Group
        settings_group = QGroupBox("▼ Settings")
        settings_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        settings_layout = QVBoxLayout()
        
        # Number of robots
        robots_layout = QHBoxLayout()
        robots_layout.addWidget(QLabel("Num Robots:"))
        self.num_robots_spin = QSpinBox()
        self.num_robots_spin.setRange(2, 10)  # Reduced max for cleaner demo
        self.num_robots_spin.setValue(self.num_robots)
        self.num_robots_spin.valueChanged.connect(self.on_num_robots_changed)
        robots_layout.addWidget(self.num_robots_spin)
        settings_layout.addLayout(robots_layout)
        
        # Circle radius
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Circle Radius:"))
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(1.0, 20.0)
        self.radius_spin.setSingleStep(0.5)
        self.radius_spin.setValue(self.circle_radius)
        self.radius_spin.valueChanged.connect(self.on_radius_changed)
        radius_layout.addWidget(self.radius_spin)
        settings_layout.addLayout(radius_layout)
        
        # Number of landmarks
        landmarks_layout = QHBoxLayout()
        landmarks_layout.addWidget(QLabel("Num Landmarks:"))
        self.num_landmarks_spin = QSpinBox()
        self.num_landmarks_spin.setRange(0, 10)
        self.num_landmarks_spin.setValue(self.num_landmarks)
        self.num_landmarks_spin.valueChanged.connect(self.on_num_landmarks_changed)
        landmarks_layout.addWidget(self.num_landmarks_spin)
        settings_layout.addLayout(landmarks_layout)
        
        # Use landmark only
        self.landmark_only_cb = QCheckBox("Use Landmark Only")
        self.landmark_only_cb.setChecked(self.use_landmark_only)
        self.landmark_only_cb.stateChanged.connect(self.on_landmark_only_changed)
        settings_layout.addWidget(self.landmark_only_cb)
        
        # Iter before motion
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Iter Before Mot:"))
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, 20)
        self.iter_spin.setValue(self.iter_before_motion)
        self.iter_spin.valueChanged.connect(self.on_iter_changed)
        iter_layout.addWidget(self.iter_spin)
        settings_layout.addLayout(iter_layout)
        
        # Linearize every
        lin_layout = QHBoxLayout()
        lin_layout.addWidget(QLabel("Linearize Every:"))
        self.lin_spin = QSpinBox()
        self.lin_spin.setRange(1, 10)
        self.lin_spin.setValue(self.linearize_every)
        self.lin_spin.valueChanged.connect(self.on_linearize_changed)
        lin_layout.addWidget(self.lin_spin)
        settings_layout.addLayout(lin_layout)
        
        # Sensor range
        sensor_layout = QHBoxLayout()
        sensor_layout.addWidget(QLabel("Sensor Range:"))
        self.sensor_spin = QDoubleSpinBox()
        self.sensor_spin.setRange(1.0, 30.0)
        self.sensor_spin.setSingleStep(0.5)
        self.sensor_spin.setValue(self.sensor_range)
        self.sensor_spin.valueChanged.connect(self.on_sensor_range_changed)
        sensor_layout.addWidget(self.sensor_spin)
        settings_layout.addLayout(sensor_layout)
        
        # Damping
        damp_layout = QHBoxLayout()
        damp_layout.addWidget(QLabel("Damping:"))
        self.damp_spin = QDoubleSpinBox()
        self.damp_spin.setRange(0.0, 1.0)
        self.damp_spin.setSingleStep(0.01)
        self.damp_spin.setValue(self.damping)
        self.damp_spin.valueChanged.connect(self.on_damping_changed)
        damp_layout.addWidget(self.damp_spin)
        settings_layout.addLayout(damp_layout)
        
        # Noise fraction
        noise_layout = QHBoxLayout()
        noise_layout.addWidget(QLabel("Noise Fraction:"))
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 0.1)
        self.noise_spin.setSingleStep(0.001)
        self.noise_spin.setValue(self.noise_std[0])
        self.noise_spin.valueChanged.connect(self.on_noise_changed)
        noise_layout.addWidget(self.noise_spin)
        settings_layout.addLayout(noise_layout)
        
        # Is robust
        self.robust_cb = QCheckBox("Is Robust")
        self.robust_cb.setChecked(self.is_robust)
        self.robust_cb.stateChanged.connect(self.on_robust_changed)
        settings_layout.addWidget(self.robust_cb)
        
        # Use odometry
        self.odometry_cb = QCheckBox("Use Odometry Factors")
        self.odometry_cb.setChecked(self.use_odometry)
        self.odometry_cb.stateChanged.connect(self.on_odometry_changed)
        settings_layout.addWidget(self.odometry_cb)
        
        # Async communication
        self.async_cb = QCheckBox("Async Communication")
        self.async_cb.setChecked(self.async_communication)
        self.async_cb.stateChanged.connect(self.on_async_changed)
        settings_layout.addWidget(self.async_cb)
        
        # Communication drop rate
        drop_layout = QHBoxLayout()
        drop_layout.addWidget(QLabel("Packet Loss:"))
        self.drop_spin = QDoubleSpinBox()
        self.drop_spin.setRange(0.0, 50.0)
        self.drop_spin.setSingleStep(5.0)
        self.drop_spin.setValue(self.communication_drop_rate * 100)
        self.drop_spin.setSuffix("%")
        self.drop_spin.valueChanged.connect(self.on_drop_rate_changed)
        drop_layout.addWidget(self.drop_spin)
        settings_layout.addLayout(drop_layout)
        
        # Allow dynamic join/leave
        self.dynamic_cb = QCheckBox("Allow Dynamic Join/Leave")
        self.dynamic_cb.setChecked(self.allow_dynamic_join_leave)
        self.dynamic_cb.stateChanged.connect(self.on_dynamic_changed)
        settings_layout.addWidget(self.dynamic_cb)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Visualization Group
        viz_group = QGroupBox("▼ Visualisation")
        viz_group.setStyleSheet("QGroupBox { font-weight: bold; }")
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
        
        self.show_samples_cb = QCheckBox("Show Samples")
        self.show_samples_cb.setChecked(self.show_samples)
        self.show_samples_cb.stateChanged.connect(self.on_show_samples_changed)
        viz_layout.addWidget(self.show_samples_cb)
        
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
        layout.addWidget(viz_group)
        
        # Debug/Status Group
        debug_group = QGroupBox("▼ Status")
        debug_group.setStyleSheet("QGroupBox { font-weight: bold; }")
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
        layout.addWidget(debug_group)
        
        layout.addStretch()
        
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
    
    def on_show_factors_changed(self, state):
        self.show_factors = (state == Qt.CheckState.Checked)
    
    def on_show_landmark_changed(self, state):
        self.show_only_landmark_factors = (state == Qt.CheckState.Checked)
    
    def on_show_path_changed(self, state):
        self.show_path = (state == Qt.CheckState.Checked)
    
    def on_show_samples_changed(self, state):
        self.show_samples = (state == Qt.CheckState.Checked)
    
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
        
        # Update differential drive robots with RVO
        for robot in self.robots:
            if not robot.is_active:
                continue
            
            # Preferred velocity (circular motion with individual paths)
            # Use robot's direction to determine rotation direction
            direction = getattr(robot, 'direction', 1)  # Default to clockwise if not set
            angular_velocity = 0.03 * direction  # Different directions for different robots
            robot.estimated_angle += angular_velocity * self.dt
            
            # Use individual path parameters to avoid clustering
            if hasattr(robot, 'path_radius') and hasattr(robot, 'path_center_offset'):
                path_center = self.circle_center + robot.path_center_offset
                preferred_pos = path_center + robot.path_radius * np.array([
                    np.cos(robot.estimated_angle), np.sin(robot.estimated_angle)
                ])
            else:
                # Fallback to default circular path
                preferred_pos = self.circle_center + self.circle_radius * np.array([
                    np.cos(robot.estimated_angle), np.sin(robot.estimated_angle)
                ])
            
            preferred_vel = (preferred_pos - robot.mu) / self.dt
            # Limit preferred velocity magnitude
            vel_mag = np.linalg.norm(preferred_vel)
            if vel_mag > self.rvo_max_speed:
                preferred_vel = preferred_vel / vel_mag * self.rvo_max_speed
            robot.preferred_vel = preferred_vel
            
            # Apply RVO collision avoidance
            if self.use_rvo:
                neighbors = []
                # Add other robots as neighbors
                for neighbor in self.robots:
                    if neighbor.id != robot.id and neighbor.is_active:
                        # Estimate neighbor velocity
                        if len(self.robot_paths[neighbor.id]) > 1:
                            neighbor_vel = (neighbor.mu - self.robot_paths[neighbor.id][-1]) / self.dt
                        else:
                            neighbor_vel = np.array([0.0, 0.0])
                        neighbors.append({
                            'pos': neighbor.mu,
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
                
                # Compute safe velocity using RVO
                safe_vel = RVO.compute_rvo_velocity(
                    robot.mu, preferred_vel, neighbors,
                    self.rvo_time_horizon, self.rvo_neighbor_dist, self.rvo_max_speed
                )
                
                # Convert to differential drive control
                linear_vel, angular_vel = RVO.compute_differential_control(
                    robot.mu, robot.angle, safe_vel,
                    max_linear=self.rvo_max_speed, max_angular=1.0
                )
                
                robot.linear_vel = linear_vel
                robot.angular_vel = angular_vel
            else:
                # No RVO, use preferred velocity directly
                linear_vel, angular_vel = RVO.compute_differential_control(
                    robot.mu, robot.angle, preferred_vel,
                    max_linear=self.rvo_max_speed, max_angular=1.0
                )
                robot.linear_vel = linear_vel
                robot.angular_vel = angular_vel
            
            # Update robot kinematics
            robot.update_kinematics(self.dt)
            robot.gt_pos = robot.mu.copy()  # For visualization
            robot.expected_pos = robot.mu.copy()
            
            # Update paths (limit to 200 for cleaner visualization)
            self.robot_paths[robot.id].append(robot.mu.copy())
            self.robot_gt_paths[robot.id].append(robot.gt_pos.copy())
            if len(self.robot_paths[robot.id]) > 200:
                self.robot_paths[robot.id].pop(0)
            if len(self.robot_gt_paths[robot.id]) > 200:
                self.robot_gt_paths[robot.id].pop(0)
        
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
                if self.use_landmark_only:
                    for lm_idx, landmark_pos in enumerate(self.landmarks[:self.num_landmarks]):
                        dist = np.linalg.norm(robot.mu - landmark_pos)  # Use estimated position
                        if dist < self.sensor_range:
                            angle = np.arctan2(robot.mu[1] - landmark_pos[1],
                                             robot.mu[0] - landmark_pos[0])
                            measurement = np.array([dist, angle]) + np.random.normal(0, self.noise_std[0], 2)
                            message = robot.get_local_message(landmark_pos, measurement, self.noise_std, self.is_robust)
                            robot.inbox[f'landmark_{lm_idx}'] = message
                            messages_received += 1
                            message_details.append(f"  {robot.name}: received from landmark {lm_idx+1} (dist={dist:.2f})")
                else:
                    # Include landmark messages even when not in landmark-only mode
                    for lm_idx, landmark_pos in enumerate(self.landmarks[:self.num_landmarks]):
                        dist = np.linalg.norm(robot.mu - landmark_pos)
                        if dist < self.sensor_range:
                            angle = np.arctan2(robot.mu[1] - landmark_pos[1],
                                             robot.mu[0] - landmark_pos[0])
                            measurement = np.array([dist, angle]) + np.random.normal(0, self.noise_std[0], 2)
                            message = robot.get_local_message(landmark_pos, measurement, self.noise_std, self.is_robust)
                            robot.inbox[f'landmark_{lm_idx}'] = message
                            messages_received += 1
                
                # 3. PEER OBSERVATIONS (Robot-to-robot messages)
                # Robots observe each other's ESTIMATED positions (mu), not ground truth!
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
                                if self.async_communication:
                                    # Packet loss simulation (cut off wifi)
                                    if np.random.random() < self.communication_drop_rate:
                                        self.dropped_messages_count += 1
                                        # Add to message queue for delayed delivery (simulating retry)
                                        if neighbor.id not in robot.message_queue:
                                            robot.message_queue.append({
                                                'from': neighbor.id,
                                                'iter': self.iter_count,
                                                'delayed': True
                                            })
                                        continue  # Message dropped
                                    
                                    # Message delay (store in queue for future delivery)
                                    if self.communication_delay > 0:
                                        if neighbor.id not in robot.last_message_time or \
                                           self.iter_count - robot.last_message_time[neighbor.id] >= self.communication_delay:
                                            robot.last_message_time[neighbor.id] = self.iter_count
                                        else:
                                            # Message delayed - add to queue
                                            if neighbor.id not in robot.message_queue:
                                                robot.message_queue.append({
                                                    'from': neighbor.id,
                                                    'iter': self.iter_count,
                                                    'delayed': True
                                                })
                                            continue  # Message delayed
                                
                                # Generate measurement from neighbor's ESTIMATED position
                                dx = neighbor.mu[0] - robot.mu[0]
                                dy = neighbor.mu[1] - robot.mu[1]
                                r = np.sqrt(dx**2 + dy**2)
                                angle = np.arctan2(dy, dx)
                                
                                # Add sensor noise (sometimes with outliers for robust testing)
                                if self.is_robust and np.random.random() < 0.05:  # 5% outlier rate
                                    measurement = np.array([r, angle]) + np.random.normal(0, self.noise_std[0] * 5, 2)
                                else:
                                    measurement = np.array([r, angle]) + np.random.normal(0, self.noise_std[0], 2)
                                
                                # Neighbor creates message (with robust factor if enabled)
                                message = neighbor.get_local_message(robot.mu, measurement, self.noise_std, self.is_robust)
                                robot.inbox[neighbor.id] = message
                                messages_received += 1
                                
                                # For logging
                                dist_gt = np.linalg.norm(robot.gt_pos - neighbor.gt_pos)
                                message_details.append(f"  {robot.name}: received from {neighbor.name} (est_dist={dist_estimated:.2f}, gt_dist={dist_gt:.2f})")
                
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
                    # Calculate GPB error (difference between estimated and ground truth)
                    gpb_error = np.linalg.norm(robot.mu - robot.gt_pos)
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
                for detail in message_details[:10]:  # Limit to first 10 to avoid spam
                    print(detail)
                if len(message_details) > 10:
                    print(f"  ... and {len(message_details) - 10} more messages")
                
                # Print position updates
                active_robots = sum(1 for r in self.robots if len(r.inbox) > 0)
                avg_error = np.mean([stats['error'] for stats in self.last_update_stats.values()])
                print(f"  Active robots: {active_robots}/{len(self.robots)}, Avg position error: {avg_error:.3f}")
                
                # Track GPB performance metrics (only during GBP iterations)
                if self.last_update_stats:
                    avg_error = np.mean([stats['error'] for stats in self.last_update_stats.values()])
                    max_error = max([stats['error'] for stats in self.last_update_stats.values()])
                    
                    # Store in history
                    self.gpb_error_history.append(avg_error)
                    self.gpb_message_history.append(self.messages_per_iteration)
                    
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
        
        if self.step_by_step:
            self.running = False
            self.run_cb.setChecked(False)

    def update_plot(self):
        self.ax.clear()
        
        # Determine view limits
        if self.follow_robot and 0 <= self.robot_id_to_follow < len(self.robots):
            center_robot = self.robots[self.robot_id_to_follow]
            center = center_robot.mu
            view_range = self.boundary_radius + 2
            self.ax.set_xlim(center[0] - view_range, center[0] + view_range)
            self.ax.set_ylim(center[1] - view_range, center[1] + view_range)
        else:
            view_range = self.boundary_radius + 2
            self.ax.set_xlim(-view_range, view_range)
            self.ax.set_ylim(-view_range, view_range)
        
        # Draw boundary circle
        boundary_circle = plt.Circle(self.circle_center, self.boundary_radius, 
                                    fill=False, linestyle='--', color='red', 
                                    alpha=0.3, linewidth=2, label='Boundary')
        self.ax.add_patch(boundary_circle)
        
        # Draw paths (with limited length for cleaner visualization)
        if self.show_path:
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.robots)))
            for i, robot in enumerate(self.robots):
                if len(self.robot_paths[robot.id]) > 1:
                    path = np.array(self.robot_paths[robot.id])
                    # Only show last 200 points for cleaner visualization
                    if len(path) > 200:
                        path = path[-200:]
                    self.ax.plot(path[:, 0], path[:, 1], '-', color=colors[i], 
                               alpha=0.4, linewidth=1.5, label=f"{robot.name} path")
        
        # Draw sensor ranges (visualize detection range) - only for first robot to reduce clutter
        if self.show_factors and len(self.robots) > 0:
            # Show sensor range for first robot as example
            first_robot = self.robots[0]
            sensor_circle = plt.Circle(first_robot.mu, self.sensor_range, 
                                     fill=False, linestyle=':', color='gray', 
                                     alpha=0.25, linewidth=1.5, zorder=2,
                                     label='Sensor Range' if not self.show_factors else '')
            self.ax.add_patch(sensor_circle)
        
        # Draw inter-robot communication (GPB messages) - GREEN lines like reference
        inter_robot_connections_drawn = False
        if self.show_factors:
            # Draw connections based on current robot positions and sensor range
            # Check all pairs of robots to see if they're within communication range
            drawn_pairs = set()  # Track which pairs we've already drawn to avoid duplicates
            for i, robot1 in enumerate(self.robots):
                if not robot1.is_active:
                    continue
                for j, robot2 in enumerate(self.robots):
                    if i >= j or not robot2.is_active:
                        continue
                    
                    # Check if robots are within sensor range
                    dist = np.linalg.norm(robot1.mu - robot2.mu)
                    if dist < self.sensor_range:
                        # Create unique pair identifier
                        pair_id = tuple(sorted([robot1.id, robot2.id]))
                        if pair_id not in drawn_pairs:
                            drawn_pairs.add(pair_id)
                            
                            # Green solid lines for inter-robot communication (matching reference)
                            mid_x = (robot1.mu[0] + robot2.mu[0]) / 2
                            mid_y = (robot1.mu[1] + robot2.mu[1]) / 2
                            self.ax.plot([robot1.mu[0], robot2.mu[0]],
                                       [robot1.mu[1], robot2.mu[1]],
                                       'g-', alpha=0.7, linewidth=2.5,
                                       label='Inter-robot Communication' if not inter_robot_connections_drawn else '', zorder=4)
                            # Add small arrow in middle to show communication direction
                            dx = robot2.mu[0] - robot1.mu[0]
                            dy = robot2.mu[1] - robot1.mu[1]
                            if dist > 0.5:
                                arrow_dx = dx / dist * 0.4
                                arrow_dy = dy / dist * 0.4
                                self.ax.arrow(mid_x - arrow_dx/2, mid_y - arrow_dy/2,
                                            arrow_dx, arrow_dy,
                                            head_width=0.25, head_length=0.2,
                                            fc='green', ec='green', alpha=0.8, zorder=5)
                            inter_robot_connections_drawn = True
            
            # Also show connections from inbox (current messages)
            for i, robot in enumerate(self.robots):
                if self.show_only_landmark_factors:
                    # Only show landmark connections
                    for neighbor_id in robot.inbox.keys():
                        if neighbor_id.startswith('landmark_'):
                            lm_idx = int(neighbor_id.split('_')[1])
                            if lm_idx < len(self.landmarks):
                                landmark_pos = self.landmarks[lm_idx]
                                self.ax.plot([robot.mu[0], landmark_pos[0]],
                                           [robot.mu[1], landmark_pos[1]],
                                           'r--', alpha=0.3, linewidth=0.8)
                else:
                    # Show range-bearing measurements (red lines) - the actual observations
                    for neighbor_id in robot.inbox.keys():
                        if neighbor_id.startswith('landmark_'):
                            lm_idx = int(neighbor_id.split('_')[1])
                            if lm_idx < len(self.landmarks):
                                landmark_pos = self.landmarks[lm_idx]
                                self.ax.plot([robot.mu[0], landmark_pos[0]],
                                           [robot.mu[1], landmark_pos[1]],
                                           'r--', alpha=0.4, linewidth=1.0,
                                           label='Range-Bearing Measurements' if i == 0 and lm_idx == 0 else '')
            
            # Visualize message queues (show robots with queued messages)
            for robot in self.robots:
                queue_size = self.message_queue_sizes.get(robot.id, 0)
                if queue_size > 0:
                    # Draw a small indicator showing queued messages
                    self.ax.plot(robot.mu[0] + 1.2, robot.mu[1] + 1.2, 'ro', 
                               markersize=8 + queue_size * 2, alpha=0.5, zorder=8)
                    self.ax.text(robot.mu[0] + 1.2, robot.mu[1] + 1.2, 
                               f'Q:{queue_size}', fontsize=7, ha='center', va='center',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7), zorder=9)
        
        # Draw landmarks (black squares, distinct from robot ground truth stars)
        for lm_idx, landmark_pos in enumerate(self.landmarks[:self.num_landmarks]):
            label = "Landmark" if lm_idx == 0 else None
            self.ax.plot(landmark_pos[0], landmark_pos[1], 'ks', 
                        markersize=12, markeredgewidth=2, 
                        markerfacecolor='black', markeredgecolor='white',
                        label=label, zorder=10)
            # Add landmark number
            self.ax.text(landmark_pos[0] + 0.5, landmark_pos[1] + 0.5, 
                        f'L{lm_idx+1}', fontsize=8, color='black',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                        zorder=11)
        
        # Draw obstacles
        for obs_pos in self.obstacles:
            obstacle_circle = MplCircle(obs_pos, self.obstacle_radius, 
                                      fill=True, color='red', alpha=0.5, 
                                      edgecolor='darkred', linewidth=2,
                                      label='Obstacle' if obs_pos is self.obstacles[0] else None,
                                      zorder=3)
            self.ax.add_patch(obstacle_circle)
        
        # Draw robots (differential drive - Roomba-like)
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.robots)))
        for i, robot in enumerate(self.robots):
            # Differential robots: circles with orientation arrows
            self.ax.plot(robot.mu[0], robot.mu[1], 'o', color=colors[i], 
                        markersize=10, markeredgecolor='black', markeredgewidth=1.5,
                        label=f"Robot {i+1}" if i < 3 else None, zorder=5)
            # Draw orientation arrow
            arrow_length = 0.8
            dx = arrow_length * np.cos(robot.angle)
            dy = arrow_length * np.sin(robot.angle)
            self.ax.arrow(robot.mu[0], robot.mu[1], dx, dy,
                        head_width=0.3, head_length=0.2, fc=colors[i], ec='black', zorder=6)
        
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        title = "Differential Drive Robots with RVO & GPB"
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
        
        # Update GPB performance plot
        self.ax_gpb.clear()
        if len(self.gpb_error_history) > 1:
            iterations = list(range(len(self.gpb_error_history)))
            self.ax_gpb.plot(iterations, self.gpb_error_history, 'b-', linewidth=2, label='Avg Error')
            self.ax_gpb.set_xlabel("GBP Iteration", fontsize=10)
            self.ax_gpb.set_ylabel("Position Error", fontsize=10)
            self.ax_gpb.set_title("GPB Performance: Average Position Error Over Time", fontsize=10, fontweight='bold')
            self.ax_gpb.grid(True, alpha=0.3)
            
            # Add current error as text
            if self.gpb_error_history:
                current_error = self.gpb_error_history[-1]
                self.ax_gpb.text(0.02, 0.98, f'Current Error: {current_error:.3f}', 
                               transform=self.ax_gpb.transAxes, fontsize=9,
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Show convergence trend
            if len(self.gpb_error_history) > 20:
                recent = self.gpb_error_history[-20:]
                if recent[-1] < recent[0]:
                    trend = "Converging ✓"
                    color = 'green'
                else:
                    trend = "Diverging ✗"
                    color = 'red'
                self.ax_gpb.text(0.98, 0.98, trend, transform=self.ax_gpb.transAxes, 
                               fontsize=9, color=color, fontweight='bold',
                               verticalalignment='top', horizontalalignment='right',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        self.canvas.draw()
    
    def update_status_display(self):
        """Update the status display in the control panel."""
        active_robots = sum(1 for r in self.robots if len(r.inbox) > 0)
        total_connections = len(self.active_connections) * 2  # Bidirectional
        total_queue_size = sum(self.message_queue_sizes.values())
        
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
            status_text += f"Queued: {total_queue_size} msgs"
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