import numpy as np
from uuid import uuid4
from gpb_solver import GBPSolver

class RobotMessage:
    """The packet sent between robots in the Robot Web."""
    def __init__(self, sender_id, eta, lambd, linear_point):
        self.sender_id = sender_id
        self.eta = eta
        self.lambd = lambd
        self.linear_point = linear_point # The pose used for linearization

class RobotAgent:
    def __init__(self, name, initial_mu):
        self.id = str(uuid4())[:8]
        self.name = name
        self.mu = initial_mu
        self.inbox = {}  # Stores latest messages from neighbors
        self.odometry_buffer = []  # Store odometry measurements (motion model)
        self.last_position = initial_mu.copy()  # For odometry
        self.is_active = True  # For dynamic join/leave
        self.message_queue = []  # For asynchronous communication
        self.last_message_time = {}  # Track message timestamps for async
        
    def get_local_message(self, neighbor_mu, measurement, noise_std, is_robust=False):
        """
        Creates a message to send to a neighbor based on a measurement.
        This is the 'Factor-to-Variable' message logic.
        Supports robust factors for outlier rejection.
        """
        # Linearization logic (as seen in the previous step)
        dx = self.mu[0] - neighbor_mu[0]
        dy = self.mu[1] - neighbor_mu[1]
        r = np.sqrt(dx**2 + dy**2)
        h = np.array([r, np.arctan2(dy, dx)])
        
        J = np.array([[dx/r, dy/r], [-dy/r**2, dx/r**2]])
        Lambda_s = np.diag([1/noise_std[0]**2, 1/noise_std[1]**2])
        
        # Robust factor: downweight outliers
        if is_robust:
            residual = measurement - h
            residual_norm = np.linalg.norm(residual)
            # Huber-like robust weighting: reduce weight for large residuals
            if residual_norm > 2.0 * noise_std[0]:  # Threshold for outliers
                weight = (2.0 * noise_std[0]) / residual_norm  # Downweight
                Lambda_s = Lambda_s * weight
        
        # Information Form parameters
        msg_L = J.T @ Lambda_s @ J
        msg_e = J.T @ Lambda_s @ (measurement - h + J @ self.mu)
        
        return RobotMessage(self.id, msg_e, msg_L, self.mu)
    
    def get_odometry_message(self, predicted_position, odometry_noise_std):
        """
        Creates an odometry factor message (robot's own motion model).
        This is part of the robot's local fragment.
        """
        # Odometry factor: prior on predicted position
        residual = self.mu - predicted_position
        Lambda_odo = np.diag([1/odometry_noise_std[0]**2, 1/odometry_noise_std[1]**2])
        eta_odo = Lambda_odo @ predicted_position
        
        # Convert to message format
        msg_L = Lambda_odo
        msg_e = eta_odo
        
        return RobotMessage(self.id + "_odo", msg_e, msg_L, self.mu)

    def update_from_web(self):
        """Combines all messages in the inbox using the GBPSolver."""
        if not self.inbox:
            return
        
        etas = [m.eta for m in self.inbox.values()]
        lambdas = [m.lambd for m in self.inbox.values()]
        
        # Call our static solver
        self.mu, _ = GBPSolver.update_variable(etas, lambdas)