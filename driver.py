'''
Created on Apr 4, 2012

Modified for manual driving mode only.
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
        
        # Remove automatic control parameters
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
    
    def drive(self, msg):
        '''Process telemetry and return control commands.
           In manual mode only, control values are updated by the manual input thread.'''
        self.state.setFromMsg(msg)
        # No automatic adjustments are performed.
        return self.control.toMsg()
    
    def onShutDown(self):
        pass
    
    def onRestart(self):
        pass

    def _manual_input_loop(self):
        '''Continuously listen for keyboard input and update control values.
           When a steering key (left/right) is pressed, if it is in the opposite direction
           of the last press, reset steer to zero before applying the new adjustment.
           Left and right arrow key functions are inverted:
             - Right arrow behaves as left (steer decreases by 0.1).
             - Left arrow behaves as right (steer increases by 0.1).
           "z" is used for gear up and "x" for gear down.
           For a smooth gear down, acceleration is reduced before shifting.
        '''
        # Initialize the last steer direction
        self.last_steer_direction = None

        # For Windows systems
        if sys.platform.startswith("win"):
            import msvcrt
            while True:
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b'\xe0':  # Arrow key prefix
                        key = msvcrt.getch()
                        if key == b'H':  # Up arrow: accelerate
                            self.control.accel = min(self.control.accel + 0.1, 1.0)
                            self.control.brake = 0.0
                            print("Accelerate:", self.control.accel)
                        elif key == b'P':  # Down arrow: brake
                            self.control.brake = min(self.control.brake + 0.1, 1.0)
                            self.control.accel = 0.0
                            print("Brake:", self.control.brake)
                        elif key == b'M':  # Right arrow (inverted: behaves as left)
                            if self.last_steer_direction != "left" or self.control.steer > 0:
                                self.control.steer = 0.0
                                self.last_steer_direction = "left"
                                print("Right arrow pressed (inverted to left): steering reset to 0")
                            else:
                                self.control.steer -= 0.1
                                self.control.steer = max(self.control.steer, -1.0)
                                print("Right arrow pressed (inverted to left): steer =", self.control.steer)
                        elif key == b'K':  # Left arrow (inverted: behaves as right)
                            if self.last_steer_direction != "right" or self.control.steer < 0:
                                self.control.steer = 0.0
                                self.last_steer_direction = "right"
                                print("Left arrow pressed (inverted to right): steering reset to 0")
                            else:
                                self.control.steer += 0.1
                                self.control.steer = min(self.control.steer, 1.0)
                                print("Left arrow pressed (inverted to right): steer =", self.control.steer)
                    else:
                        # Gear controls: "z" for gear up, "x" for gear down.
                        if key.lower() == b'z':
                            self.control.gear += 1
                            print("Gear up:", self.control.gear)
                        elif key.lower() == b'x':
                            # For smooth gear down, reduce acceleration first if needed.
                            if self.control.accel > 0:
                                self.control.accel = max(self.control.accel - 0.1, 0)
                                print("Reducing acceleration for smooth gear down:", self.control.accel)
                            self.control.gear -= 1
                            print("Gear down:", self.control.gear)
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
                                    print("Right arrow pressed (inverted to left): steering reset to 0")
                                else:
                                    self.control.steer -= 0.1
                                    self.control.steer = max(self.control.steer, -1.0)
                                    print("Right arrow pressed (inverted to left): steer =", self.control.steer)
                            elif seq == '[D':  # Left arrow (inverted: behaves as right)
                                if self.last_steer_direction != "right" or self.control.steer < 0:
                                    self.control.steer = 0.0
                                    self.last_steer_direction = "right"
                                    print("Left arrow pressed (inverted to right): steering reset to 0")
                                else:
                                    self.control.steer += 0.1
                                    self.control.steer = min(self.control.steer, 1.0)
                                    print("Left arrow pressed (inverted to right): steer =", self.control.steer)
                        elif key.lower() == 'z':
                            self.control.gear += 1
                            print("Gear up:", self.control.gear)
                        elif key.lower() == 'x':
                            if self.control.accel > 0:
                                self.control.accel = max(self.control.accel - 0.1, 0)
                                print("Reducing acceleration for smooth gear down:", self.control.accel)
                            self.control.gear -= 1
                            print("Gear down:", self.control.gear)
                    time.sleep(0.05)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
