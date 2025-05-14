# TORCS AI Racing Controller

## 📋 Project Overview

This project implements an intelligent racing controller for The Open Racing Car Simulator (TORCS) as part of an AI course assignment. The goal is to design a controller that can effectively race against other cars on different tracks while maintaining optimal performance metrics.

### Project Objectives
- Design a controller that can compete and win races against other cars
- Implement effective telemetry data processing
- Maintain optimal performance metrics:
  - Speed optimization
  - Obstacle avoidance
  - Track following
  - Gear management
- Develop a non-rule-based solution using machine learning

## 🎯 Our Solution

We implemented a hybrid approach combining manual control capabilities with machine learning-based autonomous driving:

### 1. Manual Control Mode
- Real-time keyboard-based control interface for data collection
- Intelligent gear management with automatic shifting
- Telemetry data collection for training
- Safety features and collision detection

### 2. Learning-Based Autonomous Mode
- Machine learning model trained on manual driving data
- Feature engineering pipeline for optimal performance
- Real-time action prediction based on car state
- Safety checks and gear management

## 🛠️ Technical Implementation

### Architecture
- Client-Server Architecture:
  - TORCS server (src_server) handles race simulation
  - Python client implements our controller
  - UDP connection for real-time communication
  - 20ms game ticks with 10ms action window

### Key Components

#### Manual Driver (`driver.py`)
- RPM-based automatic gear shifting
- Manual override capabilities
- Telemetry logging system
- Real-time control interface

#### Learning Driver (`learning_driver.py`)
- Random Forest Classifier for action prediction
- Feature engineering pipeline:
  - Standard scaling
  - Polynomial features
  - PCA dimensionality reduction
  - Interaction features
- Comprehensive feature set:
  - Speed (X, Y, Z)
  - Track position
  - Angle
  - RPM
  - Lap time
  - Distance metrics

#### Model Training (`train_models.py`)
- Data collection and preprocessing
- Feature engineering
- Model training and evaluation
- Performance visualization

#### XGBoost Model Training (`xgboostTrain.py`)
- Advanced gradient boosting implementation
- Features used:
  - Speed (X, Y, Z)
  - Track position
  - Angle
  - RPM
  - Gear state
- Model configuration:
  - 100 estimators
  - Maximum depth of 6
  - Learning rate of 0.1
- Multi-output regression for simultaneous prediction of:
  - Steering
  - Acceleration
  - Braking
- Standard scaling for feature normalization
- Performance evaluation using MSE metrics
- Model persistence using joblib

## 📋 Prerequisites

- Python 3.x
- TORCS racing simulator
- Required Python packages:
  - numpy
  - pandas
  - scikit-learn
  - joblib
  - matplotlib
  - seaborn
  - xgboost

## 🧠 AI Concepts and Libraries Used

### Machine Learning Models
1. Random Forest Regressor
   - Ensemble learning method
   - Multiple decision trees for robust predictions
   - Handles non-linear relationships
   - Resistant to overfitting

2. XGBoost Regressor
   - Gradient boosting framework
   - Sequential tree building
   - Advanced optimization techniques
   - Handles missing values automatically

### Data Processing Libraries
- **pandas**: Data manipulation and analysis
- **scikit-learn**:
  - StandardScaler for feature normalization
  - train_test_split for data partitioning
  - MultiOutputRegressor for multi-target prediction
  - mean_squared_error for model evaluation

### Model Persistence
- **joblib**: Efficient model serialization and storage

### Visualization
- **matplotlib**: Performance metrics visualization
- **seaborn**: Statistical data visualization

### Key AI Concepts Implemented
1. Feature Engineering
   - Standard scaling for feature normalization
   - Multi-dimensional feature space
   - Real-time feature processing

2. Multi-output Regression
   - Simultaneous prediction of multiple targets
   - Handles correlated outputs
   - Maintains output relationships

3. Model Evaluation
   - Mean Squared Error (MSE) metrics
   - Actual vs Predicted visualization
   - Performance analysis per target variable

4. Data Preprocessing
   - Missing value handling
   - Feature scaling
   - Train-test splitting
   - Data validation

## 🚀 Installation

1. Install TORCS:
   - Follow instructions at http://cs.adelaide.edu.au/~optlog/SCR2015/index.html
   - Watch tutorial video: https://youtu.be/EqR0y6xhX1U

2. Clone the repository:
```bash
git clone https://github.com/yourusername/torcs-ai-racing.git
cd torcs-ai-racing
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Manual Control Mode (Data Collection)
Run the manual driver:
```bash
python driver.py
```

Controls:
- Arrow keys: Steering and acceleration/braking
- Z: Gear up
- X: Gear down
- S: Start logging
- E: Stop logging
- Q: Quit

### Learning-Based Mode (Autonomous Racing)
Run the learning driver:
```bash
python learning_driver.py
```

## 🎯 Training Your Own Model

1. Collect training data using the manual driver
2. Run the training script:
```bash
python train_models.py
```

The training script will:
- Load and clean racing log data
- Perform data analysis and visualization
- Create feature engineering pipeline
- Train the action classification model
- Save the trained model and pipeline

## 📊 Performance Metrics

The learning driver optimizes:
- Speed and acceleration
- Track position and centering
- Gear management
- Obstacle avoidance
- Lap time

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TORCS development team
- Python community
- Machine learning community 
