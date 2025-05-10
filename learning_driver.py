#!/usr/bin/env python3
"""
learning_driver.py

A learning-based driver for SCRC that loads
pre-trained models (cont_model, gear_model) and a scaler,
then at each step predicts the controls using only
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
    A driver object for the SCRC that uses scikit-learn models
    to predict Accel, Brake, Steer, Clutch (continuous) and Gear (discrete).
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
        
        # load scaler and models
        try:
            self.scaler     = joblib.load(os.path.join(model_dir, "scaler.joblib"))
            self.cont_model = joblib.load(os.path.join(model_dir, "cont_model.joblib"))
            self.gear_model = joblib.load(os.path.join(model_dir, "gear_model.joblib"))
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
    
    def drive(self, msg):
        """Called each step: parse state, predict, clamp, and apply controls."""
        # parse sensors
        self.state.setFromMsg(msg)
            
        # 1) build feature vector
        feats = [self._get(col) for col in self.feature_cols]
        x = np.array(feats).reshape(1, -1)

        # 2) scale
        x_s = self.scaler.transform(x)

        # 3) predict continuous outputs
        cont = self.cont_model.predict(x_s)[0]
        if len(cont) != 4:
            raise RuntimeError(f"Expected 4 continuous outputs, got {len(cont)}")
        accel_p, brake_p, steer_p, clutch_p = cont

        # 4) predict discrete gear
        gear_p = int(self.gear_model.predict(x_s)[0])
            
        # 5) clamp
        accel  = float(np.clip(accel_p,  0.0, 1.0))
        brake  = float(np.clip(brake_p,  0.0, 1.0))
        steer  = float(np.clip(steer_p, -1.0, 1.0))
        clutch = float(np.clip(clutch_p,0.0, 1.0))
        speed  = self._get("SpeedX")

        # 6) safety: no accel+brake together
        if brake > 0.1:
            accel = 0.0
            
        # 7) validate gear transitions
        gear = self._validate_gear(gear_p, speed)
            
        # update history
        self._prev_gear  = gear
        self._prev_speed = speed

        # 8) apply to control
        self.control.setAccel(accel)
        self.control.setBrake(brake)
        self.control.setSteer(steer)
        self.control.setClutch(clutch)
        self.control.setGear(gear)

        # Add centering logic
        track_pos = self._get("TrackPos")
        if abs(track_pos) > 0.1:  # If car is significantly off center
            steer = steer - (track_pos * 0.5)  # Adjust steering to move towards center

        # 9) return message
            return self.control.toMsg()
    
    def onShutDown(self):
        pass
    
    def onRestart(self):
        self._prev_gear  = 1
        self._prev_speed = 0.0
