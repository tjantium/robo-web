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
