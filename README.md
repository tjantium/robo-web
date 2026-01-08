# Robot Web: Distributed GBP Localization

A real-time simulation of distributed Gaussian Belief Propagation (GBP) for multi-robot localization, implementing the "Robot Web" concept from distributed SLAM research.

**Author**: Thiwanka Jayasiri (for educational purposes)

**Inspired by**: 
- Paper: [Distributed Gaussian Belief Propagation for Multi-Robot SLAM](https://arxiv.org/pdf/2202.03314)
- Talk: [From SLAM to Spatial AI - Andrew Davison](https://www.youtube.com/watch?v=Z7QPR3UYIjo&list=LL&index=5&t=2477s)

## 🎯 Goals Achieved

### Technical Goals
- ✅ **Distributed MAP Inference**: Each robot maintains its own factor graph fragment
- ✅ **GBP Engine**: Local matrix operations with small message passing
- ✅ **Robust Factors**: Outlier rejection for faulty sensors
- ✅ **Async Communication**: Packet loss and delay tolerance
- ✅ **Dynamic Environments**: Robots can join/leave the network

### Key Features
- Real-time visualization with PyQt6
- Multiple landmarks support
- Configurable sensor noise and communication reliability
- Odometry factors for motion models
- Robust outlier detection
- Message passing statistics

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python test_simulation.py
```

## 📋 Requirements

- Python 3.10+
- PyQt6
- NumPy
- Matplotlib

## 🎮 Controls

### Settings Panel
- **Num Robots**: Number of robots in simulation
- **Num Landmarks**: Fixed landmarks for localization
- **Sensor Range**: Maximum observation distance
- **Use Odometry Factors**: Enable robot motion models
- **Is Robust**: Enable outlier rejection
- **Async Communication**: Simulate unreliable networks
- **Packet Loss**: Communication drop rate (0-50%)

### Visualization
- **Show Factors**: Display robot-to-robot connections
- **Show Path**: Display robot trajectory trails
- **Follow Robot**: Camera follows selected robot

## 📊 Status Panel

Real-time statistics:
- Active robots and connections
- Total messages exchanged
- Position estimation errors
- Convergence metrics

## 🔬 Technical Details

### Architecture
- **RobotAgent**: Individual robot with local state
- **GBPSolver**: Information form belief updates
- **RobotMessage**: Small message format (eta, Lambda)

### Algorithm
1. Each robot maintains local fragment (odometry + observations)
2. Robots exchange information-form messages
3. GBP combines messages to update beliefs
4. Converges to distributed MAP solution

## 📈 Assessment

See [ASSESSMENT.md](ASSESSMENT.md) for detailed goals assessment and enhancement roadmap.

## 🎓 Educational Value

This simulation demonstrates:
- Distributed state estimation
- Factor graph partitioning
- Gaussian Belief Propagation
- Robust estimation
- Asynchronous communication

## 🔮 Future Enhancements

- Convergence verification
- Centralized comparison
- Advanced robust kernels
- Factor graph visualization
- Performance metrics dashboard
- Real-world data integration

## 📚 References

- **Paper**: [Distributed Gaussian Belief Propagation for Multi-Robot SLAM](https://arxiv.org/pdf/2202.03314)
  - Implements the distributed GBP framework for multi-robot localization
  - Demonstrates equivalence to centralized solutions
  - Handles robust factors, async communication, and dynamic environments

- **Talk**: [From SLAM to Spatial AI - Andrew Davison](https://www.youtube.com/watch?v=Z7QPR3UYIjo&list=LL&index=5&t=2477s)
  - YouTube presentation discussing distributed SLAM and Robot Web concepts
  - Visual demonstrations of the algorithm in action

## 👤 Author

**Thiwanka Jayasiri**

This implementation is created for educational purposes to demonstrate and understand distributed state estimation algorithms, particularly Gaussian Belief Propagation in multi-robot systems.

## 📝 License

MIT License
