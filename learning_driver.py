#!/usr/bin/env python3
"""
learning_driver.py

A learning-based driver for SCRC that loads
pre-trained action classification model and feature pipeline,
then at each step predicts the next action using only
features present in carState.py.
"""

import os
import joblib
import numpy as np

import msgParser
import carState
import carControl

class LearningDriver(object):
    """
    A driver object for the SCRC that uses scikit-learn model
    to predict discrete actions (left, right, brake, throttle, etc.).
    """
    def __init__(self, stage,
                 model_dir="trained_models/Model-01",
                 feature_cols=None):
        # stage constants
        self.WARM_UP    = 0
        self.QUALIFYING = 1
        self.RACE       = 2
        self.UNKNOWN    = 3
        self.stage      = stage
        
        # parser / state / control
        self.parser  = msgParser.MsgParser()
        self.state   = carState.CarState()
        self.control = carControl.CarControl()
        
        # ensure model_dir exists
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # load feature pipeline, model, and action mapping
        try:
            self.feature_pipeline = joblib.load(os.path.join(model_dir, "feature_pipeline.joblib"))
            self.model = joblib.load(os.path.join(model_dir, "action_model.joblib"))
            self.action_mapping = joblib.load(os.path.join(model_dir, "action_mapping.joblib"))
            # Create reverse mapping for easier lookup
            self.reverse_mapping = {v: k for k, v in self.action_mapping.items()}
        except Exception as e:
            raise RuntimeError(f"Error loading models: {e}")

        # only features with getters in CarState
        # MUST match exactly what you used during training!
        self.feature_cols = feature_cols or [
            "SpeedX",        # getSpeedX()
            "SpeedY",        # getSpeedY()
            "SpeedZ",        # getSpeedZ()
            "Angle",         # getAngle()
            "TrackPos",      # getTrackPos()
            "Rpm",           # getRpm()
            "CurLapTime",    # getCurLapTime()
            "DistFromStart", # getDistFromStart()
            "DistRaced",     # getDistRaced()
            "Fuel",          # getFuel()
            "Damage"         # getDamage()
        ]
        
        # validate that each feature exists
        for col in self.feature_cols:
            if not (hasattr(self.state, f"get{col}") or hasattr(self.state, col)):
                raise ValueError(f"Required feature '{col}' not available in CarState")

        # track previous gear/speed for safety
        self._prev_gear  = 1
        self._prev_speed = 0.0
        
        # Control thresholds
        self.STEER_THRESHOLD = 0.1
        self.ACCEL_THRESHOLD = 0.1
        self.BRAKE_THRESHOLD = 0.1
        
        # RPM thresholds for gear changes
        self.UPSHIFT_RPM = 6500
        self.DOWNSHIFT_RPM = 3000
    
    def init(self):
        """Return the init string (rangefinder angles)."""
        angles = [0.0]*19
        for i in range(5):
            angles[i]      = -90 + i * 15
            angles[18 - i] =  90 - i * 15
        for i in range(5, 9):
            angles[i]      = -20 + (i-5) * 5
            angles[18 - i] =  20 - (i-5) * 5
        return self.parser.stringify({"init": angles})
    
    def _get(self, col):
        """Helper: call getter or attribute, default 0.0 on error."""
        try:
            getter = f"get{col}"
            if hasattr(self.state, getter):
                return getattr(self.state, getter)()
            return getattr(self.state, col)
        except Exception:
            return 0.0

    def _validate_gear(self, gear, speed):
        """Keep gear in [-1..6], no skips, handle reverse."""
        gear = int(np.clip(gear, -1, 6))
        # no skips
        if abs(gear - self._prev_gear) > 1:
            gear = self._prev_gear + np.sign(gear - self._prev_gear)
        # if asked for reverse but going forward, force 1
        if gear == -1 and speed > 0.1:
            gear = 1
        return gear
    
    def _check_gear_change(self):
        """Check if gear change is needed based on RPM."""
        current_rpm = self._get("Rpm")
        current_gear = self.control.getGear()
        current_speed = self._get("SpeedX")
        
        # Upshift logic
        if current_rpm > self.UPSHIFT_RPM and current_gear < 6 and current_speed > 0:
            return "gear_up"
        
        # Downshift logic
        if current_rpm < self.DOWNSHIFT_RPM and current_gear > 1:
            return "gear_down"
        
        return None
    
    def _apply_action(self, action):
        """Apply the predicted action to the car controls."""
        # Reset all controls
        self.control.setAccel(0.0)
        self.control.setBrake(0.0)
        self.control.setSteer(0.0)
        self.control.setClutch(0.0)
        
        # Check if gear change is needed based on RPM
        gear_action = self._check_gear_change()
        if gear_action:
            action = gear_action  # Override with gear change if needed
        
        # Apply action
        if action == "left":
            self.control.setSteer(-1.0)
        elif action == "right":
            self.control.setSteer(1.0)
        elif action == "brake":
            self.control.setBrake(1.0)
        elif action == "throttle":
            self.control.setAccel(1.0)
        elif action == "gear_up":
            new_gear = self._validate_gear(self._prev_gear + 1, self._get("SpeedX"))
            self.control.setGear(new_gear)
        elif action == "gear_down":
            new_gear = self._validate_gear(self._prev_gear - 1, self._get("SpeedX"))
            self.control.setGear(new_gear)
        elif action == "coast":
            pass  # All controls already at 0
        
        # Add centering logic
        track_pos = self._get("TrackPos")
        if abs(track_pos) > 0.1:  # If car is significantly off center
            current_steer = self.control.getSteer()
            self.control.setSteer(current_steer - (track_pos * 0.5))  # Adjust steering to move towards center
    
    def drive(self, msg):
        """Called each step: parse state, predict action, and apply controls."""
        # parse sensors
        self.state.setFromMsg(msg)
        
        # 1) build feature vector
        feats = [self._get(col) for col in self.feature_cols]
        x = np.array(feats).reshape(1, -1)

        # 2) apply feature engineering pipeline
        x_processed = self.feature_pipeline.transform(x)

        # 3) predict action
        action_idx = self.model.predict(x_processed)[0]
        action = self.reverse_mapping[action_idx]
        
        # 4) apply action
        self._apply_action(action)
        
        # 5) update history
        self._prev_gear = self.control.getGear()
        self._prev_speed = self._get("SpeedX")

        # 6) return message
        return self.control.toMsg()
    
    def onShutDown(self):
        pass
    
    def onRestart(self):
        self._prev_gear  = 1
        self._prev_speed = 0.0
