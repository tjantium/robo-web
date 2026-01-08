import numpy as np

class RVO:
    """
    Reciprocal Velocity Obstacle (RVO) implementation for collision avoidance.
    Each robot computes a safe velocity that avoids collisions with neighbors.
    """
    
    @staticmethod
    def compute_rvo_velocity(robot_pos, robot_vel, neighbors, time_horizon=2.0, neighbor_dist=2.0, max_speed=1.0):
        """
        Compute RVO velocity for a robot given its neighbors.
        
        Args:
            robot_pos: Current position [x, y]
            robot_vel: Preferred velocity [vx, vy]
            neighbors: List of dicts with 'pos', 'vel', 'radius' for each neighbor
            time_horizon: Time to look ahead for collisions
            neighbor_dist: Preferred distance to neighbors
            max_speed: Maximum speed limit
            
        Returns:
            Safe velocity [vx, vy]
        """
        if len(neighbors) == 0:
            # No neighbors, use preferred velocity (clamped to max_speed)
            vel_norm = np.linalg.norm(robot_vel)
            if vel_norm > max_speed:
                return robot_vel / vel_norm * max_speed
            return robot_vel
        
        # Collect velocity obstacles
        vo_list = []
        
        for neighbor in neighbors:
            neighbor_pos = np.array(neighbor['pos'])
            neighbor_vel = np.array(neighbor['vel'])
            neighbor_radius = neighbor.get('radius', 0.5)
            robot_radius = neighbor.get('robot_radius', 0.5)
            
            # Relative position and velocity
            rel_pos = neighbor_pos - robot_pos
            rel_vel = neighbor_vel - robot_vel
            
            # Distance to neighbor
            dist = np.linalg.norm(rel_pos)
            
            # Combined radius
            combined_radius = robot_radius + neighbor_radius
            
            if dist < 1e-6:
                # Overlapping, use separation vector
                separation = np.random.randn(2)
                separation = separation / np.linalg.norm(separation) * max_speed
                vo_list.append({
                    'type': 'separation',
                    'vector': separation
                })
                continue
            
            # Time to collision
            if np.linalg.norm(rel_vel) < 1e-6:
                # Static neighbor, use repulsion
                direction = -rel_pos / dist
                repulsion_vel = direction * max_speed * 0.5
                vo_list.append({
                    'type': 'repulsion',
                    'vector': repulsion_vel
                })
                continue
            
            # Compute VO cone
            # Angle of relative velocity
            rel_vel_norm = np.linalg.norm(rel_vel)
            rel_vel_dir = rel_vel / rel_vel_norm
            
            # Angle from robot to neighbor
            to_neighbor = rel_pos / dist
            
            # Check if collision is imminent
            time_to_collision = dist / (rel_vel_norm + 1e-6)
            
            if time_to_collision < time_horizon and dist < neighbor_dist * 2:
                # Collision risk detected
                # Compute avoidance direction (perpendicular to relative position)
                perp = np.array([-to_neighbor[1], to_neighbor[0]])
                
                # Choose side that's closer to preferred velocity
                perp1 = perp
                perp2 = -perp
                
                dot1 = np.dot(perp1, robot_vel)
                dot2 = np.dot(perp2, robot_vel)
                
                avoidance_dir = perp1 if dot1 > dot2 else perp2
                
                # Scale avoidance based on urgency
                urgency = 1.0 - (time_to_collision / time_horizon)
                avoidance_vel = avoidance_dir * max_speed * urgency
                
                vo_list.append({
                    'type': 'avoidance',
                    'vector': avoidance_vel,
                    'urgency': urgency
                })
        
        # Combine velocity obstacles
        if len(vo_list) == 0:
            # No obstacles, use preferred velocity
            vel_norm = np.linalg.norm(robot_vel)
            if vel_norm > max_speed:
                return robot_vel / vel_norm * max_speed
            return robot_vel
        
        # Weighted combination of avoidance vectors
        total_vel = np.zeros(2)
        total_weight = 0.0
        
        for vo in vo_list:
            if vo['type'] == 'separation':
                weight = 2.0
                total_vel += vo['vector'] * weight
                total_weight += weight
            elif vo['type'] == 'repulsion':
                weight = 1.5
                total_vel += vo['vector'] * weight
                total_weight += weight
            elif vo['type'] == 'avoidance':
                weight = vo['urgency'] * 1.0
                total_vel += vo['vector'] * weight
                total_weight += weight
        
        # Blend with preferred velocity
        if total_weight > 0:
            avoidance_vel = total_vel / total_weight
            # Blend: 70% avoidance, 30% preferred
            safe_vel = 0.7 * avoidance_vel + 0.3 * robot_vel
        else:
            safe_vel = robot_vel
        
        # Clamp to max speed
        vel_norm = np.linalg.norm(safe_vel)
        if vel_norm > max_speed:
            safe_vel = safe_vel / vel_norm * max_speed
        
        return safe_vel
    
    @staticmethod
    def compute_differential_control(robot_pos, robot_angle, target_vel, max_linear=1.0, max_angular=1.0):
        """
        Convert desired velocity to differential drive control (v, omega).
        
        Args:
            robot_pos: Current position [x, y]
            robot_angle: Current orientation (radians)
            target_vel: Desired velocity [vx, vy]
            max_linear: Maximum linear velocity
            max_angular: Maximum angular velocity
            
        Returns:
            (linear_vel, angular_vel)
        """
        # Desired direction
        vel_norm = np.linalg.norm(target_vel)
        if vel_norm < 1e-6:
            return (0.0, 0.0)
        
        desired_angle = np.arctan2(target_vel[1], target_vel[0])
        
        # Angle difference
        angle_diff = desired_angle - robot_angle
        # Normalize to [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        # Linear velocity (forward speed)
        linear_vel = min(vel_norm, max_linear) * np.cos(angle_diff)
        linear_vel = max(0.0, linear_vel)  # Don't go backwards
        
        # Angular velocity (turning speed)
        angular_vel = angle_diff * max_angular / np.pi
        angular_vel = np.clip(angular_vel, -max_angular, max_angular)
        
        return (linear_vel, angular_vel)
