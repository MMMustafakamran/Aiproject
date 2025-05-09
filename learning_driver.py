"""
LearningDriver - Machine Learning Based Autonomous Racing Driver

This module implements an autonomous racing driver that uses machine learning models
to control a car in the TORCS racing simulator. The driver loads pre-trained models
for steering, acceleration, and braking decisions, and applies them in real-time
to control the car.

Key Features:
- Real-time feature engineering from sensor data
- Model-based control for steering, acceleration, and braking
- Automatic gear shifting with RPM-based logic
- Safety constraints and fallback behaviors
- Comprehensive logging and error handling

Usage:
    Run with: python pyclient.py --mode learning
    Requires trained models in trained_models/Model-01/ directory
"""

import os
import numpy as np
import pandas as pd
import joblib
import msgParser
import carState
import carControl
import threading
import time
import sys
import logging

class LearningDriver:
    def __init__(self, stage):
        """Initialize the learning driver"""
        # Set up logging
        self._setup_logging()
        
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        
        # Initialize parser and state objects
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        
        # Set model directory
        self.model_dir = os.path.join('trained_models', 'Model-01')
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        # Load trained models
        self.models = self._load_models()
        
        # Initialize feature history for moving averages
        self.history = {
            'SpeedX': [],
            'SpeedY': [],
            'Angle': []
        }
        self.window_size = 5
        
        # Feature scaler
        self.scaler = None
        scaler_path = os.path.join(self.model_dir, 'scaler.joblib')
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                self.logger.info(f"Loaded scaler from {scaler_path}")
            except Exception as e:
                self.logger.error(f"Error loading scaler: {str(e)}")
        else:
            self.logger.warning(f"Scaler not found at {scaler_path}")
            
        # RPM thresholds for automatic gear shifting
        self.upshift_rpm = {
            1: 7000,
            2: 7000,
            3: 7000,
            4: 7500,
            5: 7500,
            6: 7500
        }
        self.downshift_rpm = {
            2: 2000,
            3: 2500,
            4: 3000,
            5: 3500,
            6: 4000
        }
        
        # Initialize previous RPM for change calculation
        self._prev_rpm = 0
        
        # Load feature configuration if available
        self.feature_config = self._load_feature_config()
        
        self.logger.info("LearningDriver initialized successfully")
    
    def _setup_logging(self):
        """Set up logging configuration"""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger('LearningDriver')
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = os.path.join('logs', f'learning_driver_{time.strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _load_feature_config(self):
        """Load feature configuration from Model-01 directory"""
        config_path = os.path.join(self.model_dir, 'feature_config.json')
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.logger.info(f"Loaded feature configuration from {config_path}")
                return config
            except Exception as e:
                self.logger.error(f"Error loading feature configuration: {str(e)}")
        else:
            self.logger.warning(f"Feature configuration not found at {config_path}")
        return None
    
    def _load_models(self):
        """Load the trained models from Model-01 directory"""
        models = {}
        model_files = {
            'steer': 'steer_model.joblib',
            'accel': 'accel_model.joblib',
            'brake': 'brake_model.joblib'
        }
        
        for control, model_file in model_files.items():
            model_path = os.path.join(self.model_dir, model_file)
            if os.path.exists(model_path):
                try:
                    models[control] = joblib.load(model_path)
                    self.logger.info(f"Loaded {control} model from {model_path}")
                except Exception as e:
                    self.logger.error(f"Error loading {control} model: {str(e)}")
                    raise
            else:
                error_msg = f"Model not found: {model_path}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        
        return models
    
    def init(self):
        """Return init string with rangefinder angles"""
        self.angles = [0 for x in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})
    
    def _update_history(self, state):
        """Update feature history for moving averages"""
        for feature in self.history:
            value = getattr(state, f'get{feature}')()
            self.history[feature].append(value)
            if len(self.history[feature]) > self.window_size:
                self.history[feature].pop(0)
    
    def _get_moving_averages(self):
        """Calculate moving averages from history"""
        return {
            'SpeedX_MA': np.mean(self.history['SpeedX']) if self.history['SpeedX'] else 0,
            'SpeedY_MA': np.mean(self.history['SpeedY']) if self.history['SpeedY'] else 0,
            'Angle_MA': np.mean(self.history['Angle']) if self.history['Angle'] else 0
        }
    
    def _prepare_features(self, state):
        """Prepare features for model prediction"""
        try:
            # Get basic state features
            speedX = state.getSpeedX()
            speedY = state.getSpeedY()
            speedZ = state.getSpeedZ()
            
            # Calculate speed magnitude
            speed_magnitude = np.sqrt(speedX**2 + speedY**2 + speedZ**2)
            
            # Get other state features
            angle = state.getAngle()
            rpm = state.getRpm()
            track_pos = state.getTrackPos()
            
            # Calculate changes (if history exists)
            angle_change = angle - self.history['Angle'][-1] if self.history['Angle'] else 0
            rpm_change = rpm - self._prev_rpm if hasattr(self, '_prev_rpm') else 0
            self._prev_rpm = rpm
            
            # Get moving averages
            ma_features = self._get_moving_averages()
            
            # Calculate interaction features
            speed_angle_interaction = speed_magnitude * angle
            speed_position_interaction = speed_magnitude * track_pos
            
            # Create feature dictionary with only input features (no target variables)
            features = {
                'Speed_Magnitude': speed_magnitude,
                'SpeedX': speedX,
                'SpeedY': speedY,
                'SpeedZ': speedZ,
                'Dist_From_Center': abs(track_pos),
                'Angle': angle,
                'Angle_Change': angle_change,
                'RPM': rpm,
                'RPM_Change': rpm_change,
                'TrackPos': track_pos,
                'Speed_Angle_Interaction': speed_angle_interaction,
                'Speed_Position_Interaction': speed_position_interaction,
                'SpeedX_MA': ma_features['SpeedX_MA'],
                'SpeedY_MA': ma_features['SpeedY_MA'],
                'Angle_MA': ma_features['Angle_MA']
            }
            
            # Convert to DataFrame
            df = pd.DataFrame([features])
            
            # Scale features if scaler exists
            if self.scaler:
                try:
                    # Ensure columns match those used during training
                    df = df[self.scaler.feature_names_in_]
                    df = pd.DataFrame(self.scaler.transform(df), columns=df.columns)
                except Exception as e:
                    self.logger.error(f"Error scaling features: {str(e)}")
                    # Fallback to unscaled features
                    pass
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=[
                'Speed_Magnitude', 'SpeedX', 'SpeedY', 'SpeedZ',
                'Dist_From_Center', 'Angle', 'Angle_Change', 'RPM',
                'RPM_Change', 'TrackPos', 'Speed_Angle_Interaction',
                'Speed_Position_Interaction', 'SpeedX_MA', 'SpeedY_MA', 'Angle_MA'
            ])
    
    def _apply_safety_constraints(self, predictions):
        """Apply safety constraints to model predictions"""
        try:
            # Ensure steering is within bounds
            predictions['steer'] = np.clip(predictions['steer'], -1.0, 1.0)
            
            # Ensure acceleration and brake are within bounds
            predictions['accel'] = np.clip(predictions['accel'], 0.0, 1.0)
            predictions['brake'] = np.clip(predictions['brake'], 0.0, 1.0)
            
            # Don't accelerate and brake simultaneously
            if predictions['brake'] > 0.1:
                predictions['accel'] = 0.0
            
            return predictions
        except Exception as e:
            self.logger.error(f"Error applying safety constraints: {str(e)}")
            # Return safe default values
            return {'steer': 0.0, 'accel': 0.0, 'brake': 0.0}
    
    def _auto_shift(self):
        """Handle automatic gear shifting based on RPM thresholds"""
        try:
            current_gear = self.control.gear
            current_rpm = self.state.getRpm()
            current_speed = self.state.getSpeedX()
            
            # Handle reverse gear separately
            if current_gear == -1:
                if current_speed > 0.1:  # Moving forward
                    self.control.gear = 1
                    self.logger.info("Shifting from reverse to first gear")
                return
                
            # Upshift logic
            if current_gear in self.upshift_rpm and current_gear < 6:
                if current_rpm > self.upshift_rpm[current_gear] and current_speed > 0:
                    self.control.gear += 1
                    self.logger.info(f"Auto upshift to gear {self.control.gear}")
                    
            # Downshift logic
            if current_gear in self.downshift_rpm and current_gear > 1:
                if current_rpm < self.downshift_rpm[current_gear]:
                    # Smooth downshift with rev matching
                    if self.control.accel > 0:
                        self.control.accel = max(self.control.accel - 0.2, 0)
                    self.control.gear -= 1
                    self.logger.info(f"Auto downshift to gear {self.control.gear}")
        except Exception as e:
            self.logger.error(f"Error in auto shift: {str(e)}")
    
    def drive(self, msg):
        """Process the current game state and return control commands"""
        try:
            # Update state
            self.state.setFromMsg(msg)
            
            # Update feature history
            self._update_history(self.state)
            
            # Skip prediction if we don't have enough history
            if not all(len(hist) == self.window_size for hist in self.history.values()):
                self.logger.info("Not enough history for prediction, using basic control")
                # Use basic control if not enough history
                self.control.setSteer(0.0)
                self.control.setAccel(0.0)
                self.control.setBrake(0.0)
                self._auto_shift()
                return self.control.toMsg()
            
            # Prepare features
            features = self._prepare_features(self.state)
            
            # Make predictions
            predictions = {}
            for control, model in self.models.items():
                try:
                    predictions[control] = float(model.predict(features)[0])
                except Exception as e:
                    self.logger.error(f"Error predicting {control}: {str(e)}")
                    predictions[control] = 0.0
            
            # Apply safety constraints
            predictions = self._apply_safety_constraints(predictions)
            
            # Update control commands
            self.control.setSteer(predictions['steer'])
            self.control.setAccel(predictions['accel'])
            self.control.setBrake(predictions['brake'])
            
            # Handle automatic gear shifting
            self._auto_shift()
            
            return self.control.toMsg()
            
        except Exception as e:
            self.logger.error(f"Error in drive method: {str(e)}")
            # Return safe default control
            self.control.setSteer(0.0)
            self.control.setAccel(0.0)
            self.control.setBrake(0.0)
            return self.control.toMsg()
    
    def onShutDown(self):
        """Clean up when the driver is shut down"""
        self.logger.info("Driver shutting down")
    
    def onRestart(self):
        """Reset when the driver is restarted"""
        self.logger.info("Driver restarting")
        # Clear history
        for key in self.history:
            self.history[key] = [] 