import msgParser
import carState
import carControl

class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self, stage):
        '''Constructor'''
        self.WARM_UP     = 0
        self.QUALIFYING  = 1
        self.RACE        = 2
        self.UNKNOWN     = 3
        self.stage       = stage
        
        self.parser      = msgParser.MsgParser()
        self.state       = carState.CarState()
        self.control     = carControl.CarControl()
        
        self.steer_lock  = 0.785398
        self.max_speed   = 200       # km/h
        # gear thresholds
        self.MIN_GEAR       = 1
        self.MAX_GEAR       = 6
        self.UPSHIFT_RPM    = 7000
        self.DOWNSHIFT_RPM  = 3000
        
        # Curvature sensitivity parameters
        self.CURVATURE_SENSITIVITY = 0.6  # Higher = more sensitive to turns
        self.MIN_TURN_THRESHOLD = 0.4     # Minimum turn sharpness to consider
        self.MAX_TURN_THRESHOLD = 0.7     # Maximum turn sharpness for full braking
        self.BRAKE_THRESHOLD = 0.6        # How curved the path needs to be to apply brakes

    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for _ in range(19)]
        
        # -90 to +90 in 19 beams, finer near center
        for i in range(5):
            self.angles[i]         = -90 + i * 15
            self.angles[18 - i]    =  90 - i * 15
        for i in range(5, 9):
            self.angles[i]         = -20 + (i-5) * 5
            self.angles[18 - i]    =  20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})
    
    def drive(self, msg):
        '''Main loop called every simulation step'''
        # parse incoming message into our CarState
        self.state.setFromMsg(msg)
        
        # compute controls
        self.steer()
        self.gear()
        self.speed()
        
        # send back
        return self.control.toMsg()
    
    def steer(self):
        '''Simple rule-based steering'''
        angle    = self.state.angle     # car's yaw relative to track axis
        trackPos = self.state.trackPos  # lateral displacement (-1..+1)
        steer_cmd = (angle - trackPos * 0.5) / self.steer_lock
        self.control.setSteer(steer_cmd)
    
    def gear(self):
        '''Threshold-based up/down shifts between 1 and 6'''
        rpm  = self.state.getRpm()
        gear = self.state.getGear()
        
        if rpm > self.UPSHIFT_RPM and gear < self.MAX_GEAR:
            gear += 1
        elif rpm < self.DOWNSHIFT_RPM and gear > self.MIN_GEAR:
            gear -= 1
        
        self.control.setGear(gear)
    
    def speed(self):
        speed = self.state.getSpeedX()
        accel = self.control.getAccel()
        steer = self.control.getSteer()
        track_pos = self.state.trackPos
        track_sensors = self.state.getTrack()

        # Calculate how sharp the turn is based on steering, track position, and upcoming track
        turn_sharpness = (abs(steer) + abs(track_pos) * 0.3) * self.CURVATURE_SENSITIVITY
        
        # Check upcoming track conditions (using front sensors)
        if track_sensors and len(track_sensors) >= 5:
            upcoming_track = min(track_sensors[0:5])  # Look at front sensors
            if upcoming_track < 30:  # Only consider very narrow sections
                turn_sharpness += (30 - upcoming_track) / 150.0  # Reduced impact

        # Base acceleration adjustment
        if speed < self.max_speed:
            target_accel = 1.0
        else:
            target_accel = 0.0

        # Adjust acceleration based on turn sharpness
        if turn_sharpness > self.MIN_TURN_THRESHOLD:
            # Gradual reduction based on turn sharpness
            reduction = min(turn_sharpness, 0.6)  # Reduced maximum reduction
            target_accel = target_accel * (1 - reduction)
            
            # Ensure minimum acceleration for recovery
            target_accel = max(target_accel, 0.4)  # Increased minimum acceleration

        # More responsive acceleration changes
        if accel < target_accel:
            accel += 0.3  # Faster acceleration increase
        elif accel > target_accel:
            accel -= 0.2  # Slightly slower deceleration for stability

        # Ensure acceleration stays within bounds
        accel = max(0.0, min(1.0, accel))

        # Add braking logic - only for very curved paths
        brake = 0.0
        if turn_sharpness > self.BRAKE_THRESHOLD and speed > 80:
            # Calculate brake intensity based on how much we exceed the threshold
            brake_intensity = (turn_sharpness - self.BRAKE_THRESHOLD) / (1.0 - self.BRAKE_THRESHOLD)
            brake = min(brake_intensity * 0.8, 0.8)  # Cap at 80% brake

        # Apply controls
        self.control.setAccel(accel)
        self.control.setBrake(brake)

    def onShutDown(self):
        pass
    
    def onRestart(self):
        pass
