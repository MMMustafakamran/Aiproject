#LEARNING BASED DRIVER
'''
manual driving mode only.
run using: python pyclient.py --manual
'''

import msgParser
import carState
import carControl
import threading
import time
import sys

class Driver(object):
    '''
    A driver object for the SCRC that operates in manual mode only.
    '''

    def __init__(self, stage):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        
        # Initialize control values
        self.control.accel = 0.0
        self.control.brake = 0.0
        self.control.steer = 0.0
        self.control.gear = 1
        
        # RPM thresholds for automatic gear shifting
        self.upshift_rpm = {
            1: 6500,
            2: 6600,
            3: 6700,
            4: 7500,
            5: 7500,
            6: 7500
        }
        self.downshift_rpm = {
            2: 2000,
            3: 2500,
            4: 2000,
            5: 2500,
            6: 3000
        }
        
        # Variables for manual override tracking
        self.last_manual_shift_time = time.time()
        self.manual_override_timeout = 2.0  # seconds
        self.is_auto_shifting = True
        self.last_steer_direction = None
        self.is_stopped = True  # Initialize as True
        
        # Start the manual input thread immediately
        self.manual_thread = threading.Thread(target=self._manual_input_loop, daemon=True)
        self.manual_thread.start()

    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})
    
    def _auto_shift(self):
        '''Handle automatic gear shifting based on RPM thresholds'''
        if not self.is_auto_shifting:
            return
            
        current_gear = self.control.gear
        current_rpm = self.state.getRpm()
        current_speed = self.state.getSpeedX()
        
        # Safety check for None values
        if current_rpm is None or current_speed is None:
            return
            
        # Handle reverse gear separately
        if current_gear == -1:
            if current_speed > 0.1:  # Moving forward
                self.control.gear = 1
            return
            
        # Upshift logic
        if current_gear in self.upshift_rpm and current_gear < 6:
            if current_rpm > self.upshift_rpm[current_gear] and current_speed > 0:
                self.control.gear += 1
                print(f"Auto upshift to gear {self.control.gear}")
                
        # Downshift logic
        if current_gear in self.downshift_rpm and current_gear > 1:
            if current_rpm < self.downshift_rpm[current_gear]:
                # Smooth downshift with rev matching
                if self.control.accel > 0:
                    self.control.accel = max(self.control.accel - 0.2, 0)
                self.control.gear -= 1
                print(f"Auto downshift to gear {self.control.gear}")

    def drive(self, msg):
        '''Process telemetry and return control commands.'''
        try:
            self.state.setFromMsg(msg)
            # Update is_stopped based on current speed
            current_speed = self.state.getSpeedX()
            if current_speed is not None:
                self.is_stopped = abs(current_speed) < 0.1
            self._auto_shift()  # Check for automatic gear shifts
            return self.control.toMsg()
        except Exception as e:
            print(f"Error in drive method: {e}")
            return self.control.toMsg()  # Return last known good control state
    
    def onShutDown(self):
        pass
    
    def onRestart(self):
        # Reset control values
        self.control.accel = 0.0
        self.control.brake = 0.0
        self.control.steer = 0.0
        self.control.gear = 1
        self.is_stopped = True
        self.last_steer_direction = None

    def _manual_input_loop(self):
        '''Continuously listen for keyboard input and update control values.'''
        try:
            # For Windows systems
            if sys.platform.startswith("win"):
                import msvcrt
                while True:
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        if key == b'\xe0':  # Arrow key prefix
                            key = msvcrt.getch()
                            
                            if key == b'H':  # Up arrow
                                if self.control.gear == -1:  # In reverse
                                    if self.is_stopped:
                                        self.control.gear = 1  # Shift to first gear
                                        print("Shifting from reverse to first gear")
                                    else:
                                        self.control.brake = min(self.control.brake + 0.1, 1.0)
                                        self.control.accel = 0.0
                                        print("Braking in reverse:", self.control.brake)
                                else:
                                    self.control.accel = min(self.control.accel + 0.1, 1.0)
                                    self.control.brake = 0.0
                                    print("Accelerate:", self.control.accel)
                                    
                            elif key == b'P':  # Down arrow
                                if self.control.gear == -1:  # In reverse
                                    self.control.accel = min(self.control.accel + 0.1, 1.0)
                                    self.control.brake = 0.0
                                    print("Accelerate in reverse:", self.control.accel)
                                elif self.is_stopped and self.control.gear > 0:
                                    self.control.gear = -1  # Shift to reverse
                                    print("Shifting to reverse gear")
                                else:
                                    self.control.brake = min(self.control.brake + 0.1, 1.0)
                                    self.control.accel = 0.0
                                    print("Brake:", self.control.brake)
                                    
                            elif key == b'M':  # Right arrow (inverted: behaves as left)
                                if self.last_steer_direction != "left" or self.control.steer > 0:
                                    self.control.steer = 0.0
                                    self.last_steer_direction = "left"
                                else:
                                    self.control.steer -= 0.1
                                    self.control.steer = max(self.control.steer, -1.0)
                                print("Steer left:", self.control.steer)
                                
                            elif key == b'K':  # Left arrow (inverted: behaves as right)
                                if self.last_steer_direction != "right" or self.control.steer < 0:
                                    self.control.steer = 0.0
                                    self.last_steer_direction = "right"
                                else:
                                    self.control.steer += 0.1
                                    self.control.steer = min(self.control.steer, 1.0)
                                print("Steer right:", self.control.steer)
                        else:
                            # Gear controls: "z" for gear up, "x" for gear down
                            if key.lower() == b'z':
                                if self.control.gear < 6:
                                    self.control.gear += 1
                                    self.last_manual_shift_time = time.time()
                                    self.is_auto_shifting = False
                                    print("Manual gear up:", self.control.gear)
                            elif key.lower() == b'x':
                                if self.control.gear > -1:
                                    if self.control.accel > 0:
                                        self.control.accel = max(self.control.accel - 0.1, 0)
                                    self.control.gear -= 1
                                    self.last_manual_shift_time = time.time()
                                    self.is_auto_shifting = False
                                    print("Manual gear down:", self.control.gear)
                            
                    # Check if manual override timeout has expired
                    if not self.is_auto_shifting and time.time() - self.last_manual_shift_time > self.manual_override_timeout:
                        self.is_auto_shifting = True
                        print("Returning to automatic gear shifting")
                        
                    time.sleep(0.05)
            else:
                # For Unix-like systems
                import select, tty, termios
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                tty.setcbreak(fd)
                try:
                    while True:
                        dr, _, _ = select.select([sys.stdin], [], [], 0)
                        if dr:
                            key = sys.stdin.read(1)
                            if key == '\x1b':  # Escape sequence for arrow keys
                                seq = sys.stdin.read(2)
                                if seq == '[A':  # Up arrow: accelerate
                                    self.control.accel = min(self.control.accel + 0.1, 1.0)
                                    self.control.brake = 0.0
                                    print("Accelerate:", self.control.accel)
                                elif seq == '[B':  # Down arrow: brake
                                    self.control.brake = min(self.control.brake + 0.1, 1.0)
                                    self.control.accel = 0.0
                                    print("Brake:", self.control.brake)
                                elif seq == '[C':  # Right arrow (inverted: behaves as left)
                                    if self.last_steer_direction != "left" or self.control.steer > 0:
                                        self.control.steer = 0.0
                                        self.last_steer_direction = "left"
                                    else:
                                        self.control.steer -= 0.1
                                        self.control.steer = max(self.control.steer, -1.0)
                                    print("Steer left:", self.control.steer)
                                elif seq == '[D':  # Left arrow (inverted: behaves as right)
                                    if self.last_steer_direction != "right" or self.control.steer < 0:
                                        self.control.steer = 0.0
                                        self.last_steer_direction = "right"
                                    else:
                                        self.control.steer += 0.1
                                        self.control.steer = min(self.control.steer, 1.0)
                                    print("Steer right:", self.control.steer)
                            elif key.lower() == 'z':
                                if self.control.gear < 6:
                                    self.control.gear += 1
                                    print("Gear up:", self.control.gear)
                            elif key.lower() == 'x':
                                if self.control.gear > -1:
                                    if self.control.accel > 0:
                                        self.control.accel = max(self.control.accel - 0.1, 0)
                                    self.control.gear -= 1
                                    print("Gear down:", self.control.gear)
                        time.sleep(0.05)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception as e:
            print(f"Error in manual input loop: {e}")
            # Try to restore terminal settings if possible
            if not sys.platform.startswith("win"):
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except:
                    pass
