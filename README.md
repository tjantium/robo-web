# Robot Web: Distributed GBP Localization

A real-time simulation of distributed Gaussian Belief Propagation (GBP) for multi-robot localization, implementing the "Robot Web" concept from distributed SLAM research.

**Author**: Thiwanka Jayasiri (for educational purposes)

**Inspired by**: 
- Paper: [Distributed Gaussian Belief Propagation for Multi-Robot SLAM](https://arxiv.org/pdf/2202.03314)
- Talk: [From SLAM to Spatial AI - Andrew Davison](https://www.youtube.com/watch?v=Z7QPR3UYIjo&list=LL&index=5&t=2477s)

## Overview

This implementation demonstrates distributed Maximum A Posteriori (MAP) inference on a global factor graph using Gaussian Belief Propagation. Each robot maintains its own local fragment of the factor graph (odometry and observations) and exchanges small information-form messages with neighbors to achieve distributed localization.

## Technical Goals

- **Distributed MAP Inference**: Each robot maintains its own factor graph fragment
- **GBP Engine**: Local matrix operations with small message passing
- **Robust Factors**: Outlier rejection for faulty sensors
- **Async Communication**: Packet loss and delay tolerance
- **Dynamic Environments**: Robots can join/leave the network

## Features

- Real-time visualization with PyQt6
- Multiple landmarks support
- Configurable sensor noise and communication reliability
- Odometry factors for motion models
- Robust outlier detection
- Message passing statistics and convergence tracking

## Installation

```bash
pip install -r requirements.txt
python test_simulation.py
```

## Requirements

- Python 3.10+
- PyQt6
- NumPy
- Matplotlib

## Usage

### Settings Panel

- **Num Robots**: Number of robots in simulation
- **Num Landmarks**: Fixed landmarks for localization
- **Sensor Range**: Maximum observation distance
- **Use Odometry Factors**: Enable robot motion models
- **Is Robust**: Enable outlier rejection
- **Async Communication**: Simulate unreliable networks
- **Packet Loss**: Communication drop rate (0-50%)

### Visualization Options

- **Show Factors**: Display robot-to-robot connections
- **Show Path**: Display robot trajectory trails
- **Follow Robot**: Camera follows selected robot

### Status Panel

Real-time statistics including:
- Active robots and connections
- Total messages exchanged
- Position estimation errors
- Convergence metrics

## What This Demo Shows

This simulation demonstrates a multi-robot system where **differential drive robots** (like Roomba vacuums) navigate together using:
1. **RVO (Reciprocal Velocity Obstacles)** for collision avoidance
2. **GPB (Gaussian Belief Propagation)** for distributed localization

### The Scenario

**Robot Type: Differential Drive Robots (Roomba-like)**

The simulation uses **differential drive robots**, which are the most common type of mobile robots (like Roomba vacuum cleaners, TurtleBot, etc.). These robots have:

- **Two independently driven wheels** (left and right)
- **Can rotate in place** (zero turning radius) by spinning wheels in opposite directions
- **Holonomic-like motion** - can move in any direction by combining forward/backward motion with rotation
- **Simple controls**: Linear velocity (forward/backward speed) and angular velocity (rotation rate)

**Why Differential Drive?**
- Most common and practical robot design
- Simple to control and model
- Good for indoor navigation
- Allows for precise positioning and turning

**The Setup:**

- **4 differential drive robots** start positioned in a circle formation
- Robots move in **circular patterns** (some clockwise, some counter-clockwise) to create interesting interactions
- **3 moving obstacles** are randomly placed - robots must navigate around them
- **1 fixed landmark** serves as a reference point for localization
- All within a **circular boundary** (radius ~20 units)

**What Happens:**

1. **Motion**: Each robot follows a circular path while using RVO to avoid collisions
2. **Observation**: Robots observe each other when within sensor range (detect distance and angle)
3. **Communication**: Robots exchange GPB messages to share position information
4. **Localization**: Each robot estimates its own position using:
   - Its own motion (odometry)
   - Observations of nearby robots
   - Observations of the landmark (if in range)
5. **Robustness**: System handles communication failures (packet loss, delays) gracefully

- **4 differential drive robots** move in circular patterns within a bounded area
- Robots use **RVO** to avoid collisions with each other and obstacles
- Robots **observe each other** and exchange **GPB messages** to improve their position estimates
- The system handles **unreliable communication** (packet loss, delays) through message queues
- A **GPB performance plot** shows how well the localization algorithm is working

### Visual Legend - What You See on Screen

**Main Simulation Plot (Top):**

- **Colored Circles with Arrows**: The robots themselves
  - Each robot has a unique color (Robot 1, Robot 2, etc.)
  - The arrow shows the robot's current orientation/direction
  
- **Colored Lines (Trails)**: Robot paths showing where they've been
  - Each robot's path is shown in its color
  - Helps visualize movement patterns

- **Green Solid Lines**: **Inter-robot Communication**
  - Connects robots that are within sensor range of each other
  - Shows which robots can exchange GPB messages
  - Arrows indicate communication direction
  - These lines appear/disappear as robots move in/out of range

- **Red Dashed Lines**: **Range-Bearing Measurements**
  - Connect robots to landmarks
  - Show sensor observations (distance and angle measurements)
  - Only visible when robots are within sensor range of landmarks

- **Gray Dashed Circle**: **Sensor Range**
  - Shows the detection/communication range around a robot
  - Robots can only communicate if they're within each other's sensor range

- **Black Square with "L1"**: **Landmark**
  - Fixed reference point that robots can observe
  - Helps with localization

- **Red Filled Circles**: **Obstacles**
  - Moving obstacles that robots must avoid using RVO
  - Robots navigate around them

- **Pink Dashed Circle**: **Boundary**
  - The outer limit of the simulation area
  - Robots stay within this boundary

- **Yellow "Q:X" Indicators**: **Message Queue**
  - Shows robots with queued messages (delayed/dropped messages waiting to be delivered)
  - Appears when communication is unreliable (packet loss, delays)
  - The number shows how many messages are queued

**GPB Performance Plot (Bottom):**

- **Blue Line**: Average position error over time
  - Shows how accurate the robots' position estimates are
  - Lower is better
  - Should decrease over time as GPB converges

- **"Current Error"**: The most recent average position error value

- **"Converging ✓" or "Diverging ✗"**: Status indicator
  - Shows whether the GPB algorithm is improving (converging) or getting worse (diverging)

### What's Being Demonstrated

1. **Distributed Localization**: Each robot estimates its own position using:
   - Its own motion (odometry)
   - Observations of nearby robots (inter-robot communication)
   - Observations of landmarks (if available)
   - All combined using Gaussian Belief Propagation

2. **Collision Avoidance**: Robots use RVO to:
   - Avoid collisions with other robots
   - Navigate around obstacles
   - Maintain safe distances while moving

3. **Robust Communication**: The system handles:
   - **Packet loss** (simulated WiFi cutoffs) - messages get dropped
   - **Message delays** - messages arrive late
   - **Message queues** - dropped/delayed messages are queued and retried
   - Robots continue to function even when communication is unreliable

4. **Real-time Performance Monitoring**: The GPB plot shows:
   - How well the localization is working
   - Whether the algorithm is converging (getting better) or diverging (getting worse)
   - Error trends over time

### Key Settings to Try

- **Increase Packet Loss** (10% → 50%): See how robots handle communication failures
- **Adjust Sensor Range**: See how communication changes as robots move in/out of range
- **Watch the GPB Plot**: Observe how position errors change as robots exchange messages
- **Check Message Queues**: Look for yellow "Q:X" indicators when packet loss is high

## Technical Details

### Architecture

- **RobotAgent**: Individual robot with local state and message handling
- **GBPSolver**: Information form belief updates
- **RobotMessage**: Small message format (eta, Lambda) for distributed communication

### Algorithm

1. Each robot maintains local fragment (odometry + observations)
2. Robots exchange information-form messages with neighbors
3. GBP combines messages to update beliefs
4. Converges to distributed MAP solution

### Factor Graph Partitioning

Instead of one computer holding the whole map, each robot only holds its own fragment:
- Internal movements (odometry factors)
- Observations of nearby peers (observation factors)
- Landmark observations (if within sensor range)

## References

- **Paper**: [Distributed Gaussian Belief Propagation for Multi-Robot SLAM](https://arxiv.org/pdf/2202.03314)
  
  Implements the distributed GBP framework for multi-robot localization, demonstrating equivalence to centralized solutions while handling robust factors, async communication, and dynamic environments.

- **Talk**: [From SLAM to Spatial AI - Andrew Davison](https://www.youtube.com/watch?v=Z7QPR3UYIjo&list=LL&index=5&t=2477s)
  
  YouTube presentation discussing distributed SLAM and Robot Web concepts with visual demonstrations of the algorithm in action.

## Educational Value

This simulation demonstrates:
- Distributed state estimation
- Factor graph partitioning
- Gaussian Belief Propagation
- Robust estimation techniques
- Asynchronous communication protocols

## Future Enhancements

- Convergence verification and quantitative analysis
- Centralized solution comparison
- Advanced robust kernels (Huber, Tukey, Cauchy)
- Factor graph structure visualization
- Performance metrics dashboard
- Real-world data integration

## Citation

If you use this implementation in your research or educational work, please cite:

**BibTeX:**
```bibtex
@software{jayasiri2024robotweb,
  author = {Jayasiri, Thiwanka},
  title = {Robot Web: Distributed GBP Localization},
  year = {2024},
  url = {https://github.com/thiwankajayasiri/robo-web},
  note = {Educational implementation of distributed Gaussian Belief Propagation for multi-robot localization}
}
```

**Plain Text:**
```
Jayasiri, T. (2024). Robot Web: Distributed GBP Localization. 
Educational implementation of distributed Gaussian Belief Propagation 
for multi-robot localization. 
https://github.com/thiwankajayasiri/robo-web
```

**Note**: This implementation is inspired by the work presented in:
- Distributed Gaussian Belief Propagation for Multi-Robot SLAM (arXiv:2202.03314)
- Andrew Davison's work on distributed SLAM and Spatial AI

## Author

Thiwanka Jayasiri

This implementation is created for educational purposes to demonstrate and understand distributed state estimation algorithms, particularly Gaussian Belief Propagation in multi-robot systems.

## License

MIT License
