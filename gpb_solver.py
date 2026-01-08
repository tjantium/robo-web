import numpy as np

class GBPSolver:
    """A modular engine to handle Information Form updates."""
    @staticmethod
    def information_form(mu, Sigma):
        """Convert mean/covariance to information vector/precision matrix."""
        Lambda = np.linalg.inv(Sigma)
        eta = Lambda @ mu
        return eta, Lambda

    @staticmethod
    def update_variable(eta_messages, lambda_messages, prior_precision=1e-6):
        """Combines all incoming messages to find the new marginal mean."""
        total_eta = np.sum(eta_messages, axis=0)
        total_lambda = np.sum(lambda_messages, axis=0)
        
        # Add a small prior for numerical stability
        total_lambda += np.eye(len(total_eta)) * prior_precision
        
        # Solve Lambda * mu = eta
        new_mu = np.linalg.solve(total_lambda, total_eta)
        return new_mu, total_lambda