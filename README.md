# Predicting Chaos in the Three-Body Problem 🌌

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/SparshKapoor-CODER/Predicting-Chaos-in-the-Three-Body-Problem-Using-Simulation---Machine-Learning?style=social)](https://github.com/SparshKapoor-CODER/Predicting-Chaos-in-the-Three-Body-Problem-Using-Simulation---Machine-Learning)

A machine learning approach to predict the chaotic behavior of three-body gravitational systems. This project combines physics simulations with deep learning to forecast whether a system will remain stable, experience collisions, or result in escape events.

🔗 **Repository Link**: [https://github.com/SparshKapoor-CODER/Predicting-Chaos-in-the-Three-Body-Problem-Using-Simulation---Machine-Learning](https://github.com/SparshKapoor-CODER/Predicting-Chaos-in-the-Three-Body-Problem-Using-Simulation---Machine-Learning)

![Three-Body Simulation Demo](https://raw.githubusercontent.com/SparshKapoor-CODER/Predicting-Chaos-in-the-Three-Body-Problem-Using-Simulation---Machine-Learning/main/screenshots/simulation_demo.gif)

## Features ✨

- **Physics Engine**: RK4 integrator for accurate gravitational simulations
- **Dataset Generation**: Creates 20,000+ simulated three-body scenarios
- **Machine Learning**: Reinforcement Learning predicts system outcomes
- **Web Interface**: Interactive 3D visualization with Three.js
- **Multi-Approach**: Includes both classification and sequence prediction models

## Project Structure 🗂️

```txt
Predicting Chaos in the Three-Body Problem/
├── app.py        # Flask web application
├── predict.py    # Prediction module
├── RK4.py        # Physics simulation core
├── generate_dataset.py # Dataset generation
├── ml_rl_classification.py # Machine learning model
├── sequence_modeling.py # LSTM time-series prediction
├── analize.py    # To run data analizis om the genrated dataset
├── requirements.txt # Python dependencies
├── static/
│ ├── script.js # Frontend logic
│ └── style.css # Styling
└── templates/
└── index.html # Web interface
```


## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/SparshKapoor-CODER/Predicting-Chaos-in-the-Three-Body-Problem-Using-Simulation---Machine-Learning.git
cd Predicting-Chaos-in-the-Three-Body-Problem-Using-Simulation---Machine-Learning
```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) Generate new training data:
   ```
   python generate_dataset.py
   ```

# Usage 🚀


