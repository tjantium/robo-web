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
from robot_m import RobotAgent

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
        self.num_robots = 6
        self.circle_radius = 5.0
        self.circle_center = np.array([0.0, 0.0])
        self.sensor_range = 8.0
        self.noise_std = [0.1, 0.05]
        self.iter_before_motion = 6
        self.linearize_every = 2
        self.damping = 0.2
        self.use_landmark_only = False
        self.is_robust = False
        self.boundary_radius = 15.0  # Maximum distance from center
        self.use_boundary_constraint = True  # Keep robots within boundary
        self.use_motion_model = True  # Use circular motion model to help tracking
        self.motion_model_weight = 0.1  # Weight for motion model prediction
        
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
        
        # Matplotlib Canvas
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
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
        print(f"Initialized {self.num_robots} robots, {self.num_landmarks} landmarks")
        print(f"Features: Odometry={self.use_odometry}, Robust={self.is_robust}, Async={self.async_communication}")
        print("Press Ctrl+C or close window to stop simulation safely")
        print("="*70 + "\n")
    
    def init_robots(self):
        """Initialize robots in a circle."""
        self.robots = []
        self.robot_paths = {}
        self.robot_gt_paths = {}
        
        for i in range(self.num_robots):
            angle = 2 * np.pi * i / self.num_robots
            # Initial position on circle
            gt_pos = self.circle_center + self.circle_radius * np.array([np.cos(angle), np.sin(angle)])
            # Add some noise to initial estimate
            initial_estimate = gt_pos + np.random.normal(0, 0.5, 2)
            
            robot = RobotAgent(f"Robot{i+1}", initial_estimate)
            robot.gt_pos = gt_pos.copy()  # Store ground truth
            robot.angle = angle  # Store angle for circular motion
            robot.estimated_angle = angle  # Initialize estimated angle for motion model
            robot.expected_pos = gt_pos.copy()  # Initialize expected position
            robot.last_position = initial_estimate.copy()  # For odometry
            robot.is_active = True  # Robot is active
            robot.message_queue = []  # For async communication
            robot.last_message_time = {}  # Track message timestamps
            self.robots.append(robot)
            self.robot_paths[robot.id] = [initial_estimate.copy()]
            self.robot_gt_paths[robot.id] = [gt_pos.copy()]
    
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
        self.num_robots_spin.setRange(2, 20)
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
        
        self.show_factors_cb = QCheckBox("Show Factors")
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

    def update_simulation(self):
        """Update simulation state."""
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
        
        # Update ground truth positions (circular motion) - for visualization/comparison only
        angular_velocity = 0.05
        self.time += self.dt
        
        for robot in self.robots:
            # Update ground truth position (circular motion) - this is the "actual" position
            # Used only for visualization and error calculation, NOT for observations
            robot.angle += angular_velocity * self.dt
            robot.gt_pos = self.circle_center + self.circle_radius * np.array([
                np.cos(robot.angle), np.sin(robot.angle)
            ])
            self.robot_gt_paths[robot.id].append(robot.gt_pos.copy())
            
            # MOVE THE ROBOT'S ESTIMATED POSITION (mu) - this is what matters!
            # Robots move their estimated positions based on motion model
            if self.use_motion_model and hasattr(robot, 'estimated_angle'):
                # Update estimated angle based on angular velocity
                robot.estimated_angle += angular_velocity * self.dt
                # Predict where robot thinks it should be (motion model)
                predicted_pos = self.circle_center + self.circle_radius * np.array([
                    np.cos(robot.estimated_angle), np.sin(robot.estimated_angle)
                ])
                # Move the estimated position (with some uncertainty)
                motion_noise = np.random.normal(0, 0.1, 2)
                robot.mu = predicted_pos + motion_noise
                robot.expected_pos = predicted_pos.copy()
            else:
                # Initialize estimated angle based on current position
                if not hasattr(robot, 'estimated_angle'):
                    direction = robot.mu - self.circle_center
                    robot.estimated_angle = np.arctan2(direction[1], direction[0])
                # Still apply motion model
                robot.estimated_angle += angular_velocity * self.dt
                predicted_pos = self.circle_center + self.circle_radius * np.array([
                    np.cos(robot.estimated_angle), np.sin(robot.estimated_angle)
                ])
                motion_noise = np.random.normal(0, 0.1, 2)
                robot.mu = predicted_pos + motion_noise
                robot.expected_pos = predicted_pos.copy()
            
            # Limit path length
            if len(self.robot_gt_paths[robot.id]) > 500:
                self.robot_gt_paths[robot.id].pop(0)
        
        # Perform GBP iterations
        if self.iter_count % self.iter_before_motion == 0:
            # Reset message counter for this iteration
            self.messages_per_iteration = 0
            message_details = []
            
            # Exchange messages between robots (Distributed MAP Inference)
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
                                # Simulate asynchronous communication
                                if self.async_communication:
                                    # Packet loss simulation
                                    if np.random.random() < self.communication_drop_rate:
                                        continue  # Message dropped
                                    
                                    # Message delay (store in queue for future delivery)
                                    if self.communication_delay > 0:
                                        if neighbor.id not in robot.last_message_time or \
                                           self.iter_count - robot.last_message_time[neighbor.id] >= self.communication_delay:
                                            robot.last_message_time[neighbor.id] = self.iter_count
                                        else:
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
                
                # 4. UPDATE ROBOT POSITION (Distributed GBP)
                # Combine all messages (odometry + observations) to update estimate
                if robot.inbox:
                    old_mu = robot.mu.copy()
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
                self.last_update_stats[robot.id] = {
                    'messages': messages_received,
                    'position': robot.mu.copy(),
                    'error': np.linalg.norm(robot.mu - robot.gt_pos)
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
        
        # Draw paths
        if self.show_path:
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.robots)))
            for i, robot in enumerate(self.robots):
                if len(self.robot_paths[robot.id]) > 1:
                    path = np.array(self.robot_paths[robot.id])
                    self.ax.plot(path[:, 0], path[:, 1], '-', color=colors[i], 
                               alpha=0.3, linewidth=1, label=f"{robot.name} path")
        
        # Draw factors (connections between robots)
        if self.show_factors:
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
                    # Show all robot-to-robot connections
                    for neighbor_id, message in robot.inbox.items():
                        if not neighbor_id.startswith('landmark_'):
                            neighbor = next((r for r in self.robots if r.id == neighbor_id), None)
                            if neighbor:
                                self.ax.plot([robot.mu[0], neighbor.mu[0]],
                                           [robot.mu[1], neighbor.mu[1]],
                                           'b--', alpha=0.2, linewidth=0.5)
                    # Show landmark connections
                    for neighbor_id in robot.inbox.keys():
                        if neighbor_id.startswith('landmark_'):
                            lm_idx = int(neighbor_id.split('_')[1])
                            if lm_idx < len(self.landmarks):
                                landmark_pos = self.landmarks[lm_idx]
                                self.ax.plot([robot.mu[0], landmark_pos[0]],
                                           [robot.mu[1], landmark_pos[1]],
                                           'g--', alpha=0.3, linewidth=0.8)
        
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
        
        # Draw robots (estimated positions)
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.robots)))
        for i, robot in enumerate(self.robots):
            self.ax.plot(robot.mu[0], robot.mu[1], 'o', color=colors[i], 
                        markersize=8, label=f"{robot.name}", zorder=5)
            
            # Draw ground truth positions (stars - these are robot true positions, NOT landmarks)
            self.ax.plot(robot.gt_pos[0], robot.gt_pos[1], '*', 
                        color=colors[i], markersize=8, alpha=0.7, 
                        markeredgecolor='black', markeredgewidth=0.5,
                        label=f"{robot.name} GT" if i == 0 else None, zorder=4)
        
        # Draw sensor ranges
        for robot in self.robots:
            circle = plt.Circle(robot.mu, self.sensor_range, 
                              fill=False, linestyle=':', color='gray', alpha=0.3)
            self.ax.add_patch(circle)
        
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right', fontsize=8)
        self.canvas.draw()
    
    def update_status_display(self):
        """Update the status display in the control panel."""
        active_robots = sum(1 for r in self.robots if len(r.inbox) > 0)
        total_connections = sum(len(r.inbox) for r in self.robots)
        
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
        status_text += f"Connections: {total_connections}"
        self.status_label.setText(status_text)
        
        self.messages_label.setText(f"Total Messages: {self.total_messages} (Last: {self.messages_per_iteration})")
        self.iterations_label.setText(f"Iterations: {self.iter_count}")
        
        robots_text = f"Robots: {len(self.robots)} active\n"
        robots_text += f"Avg Error: {avg_error:.3f}\n"
        robots_text += f"Max Error: {max_error:.3f}"
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