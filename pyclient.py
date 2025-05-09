'''
pyclient.py
Main client program for connecting a Python driver to the TORCS/SCRC racing simulator
- Establishes UDP communication with the simulator using configurable host and port
- Parses command-line arguments for bot ID, track, stage, and episode/step limits
- Initializes and manages a Driver object for driving logic
- Handles simulator messages (identification, sensor data, shutdown, restart)
- Sends control commands from the driver to the simulator
- Supports multiple episodes and step limits with timeout handling
- Prints key input (speedX, trackPos, angle, rpm, gear) and output (accel, brake, steer, gear) fields periodically
- Logs input (speedX, trackPos, angle, rpm, gear) and output (accel, brake, steer, gear) fields to a CSV file
- Logs car state and control data to CSV files for training purposes
'''
# example python pyclient --mode manual , python pyclient --mode ruleai , python pyclient --mode learning
import sys
import argparse
import socket
import driver  # Default learning-based AI driver
import manual_driver  # Manual control driver
import rule_driver  # Rule-based AI driver
import time  # Added for delay
import csv  # Added for CSV logging
import os  # Added for file path handling


if __name__ == '__main__':
    pass

# Configure argument parser
parser = argparse.ArgumentParser(description='Python client to connect to the TORCS SCRC server.')
parser.add_argument('--host', action='store', dest='host_ip', default='localhost',
                    help='Host IP address (default: localhost)')
parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001,
                    help='Host port number (default: 3001)')
parser.add_argument('--id', action='store', dest='id', default='SCR',
                    help='Bot ID (default: SCR)')
parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=1,
                    help='Maximum number of learning episodes (default: 1)')
parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0,
                    help='Maximum number of steps (default: 0)')
parser.add_argument('--track', action='store', dest='track', default=None,
                    help='Name of the track')
parser.add_argument('--stage', action='store', dest='stage', type=int, default=3,
                    help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')
parser.add_argument('--mode', action='store', dest='mode', default='learning',
                    choices=['learning', 'manual', 'ruleai'],
                    help='Driver mode: learning (default), manual, or ruleai')

arguments = parser.parse_args()

# Print summary
print('Connecting to server host ip:', arguments.host_ip, '@ port:', arguments.host_port)
print('Bot ID:', arguments.id)
print('Maximum episodes:', arguments.max_episodes)
print('Maximum steps:', arguments.max_steps)
print('Track:', arguments.track)
print('Stage:', arguments.stage)
print('Mode:', arguments.mode)
print('*********************************************')

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except socket.error as msg:
    print('Could not make a socket.')
    sys.exit(-1)

# Two second timeout for more reliable server response
sock.settimeout(2.0)

shutdownClient = False
curEpisode = 0

verbose = False

# Initialize appropriate driver based on mode
if arguments.mode == 'manual':
    d = manual_driver.Driver(arguments.stage)
elif arguments.mode == 'ruleai':
    d = rule_driver.Driver(arguments.stage)
else:  # default to learning mode
    d = driver.Driver(arguments.stage)

while not shutdownClient:
    retry_count = 0
    print(f'Attempting to connect with bot ID: {arguments.id}...')
    
    while True:
        buf = arguments.id + d.init()
        
        try:
            sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
        except socket.error as msg:
            print("Failed to send data...Exiting...")
            sys.exit(-1)
            
        try:
            buf, addr = sock.recvfrom(1000)
            buf = buf.decode()
        except socket.error as msg:
            retry_count += 1
            if retry_count % 5 == 0:  # Print every 5th retry
                print(f'Waiting for server response... (Retry {retry_count})')
            time.sleep(1.0)  # Add 1-second delay between retries
            continue
    
        if '***identified***' in buf:
            print('Connected successfully:', buf)
            break

    currentStep = 0
    consecutive_timeouts = 0  # Track consecutive timeouts
    
    # Initialize CSV file for car state and control data
    os.makedirs('results', exist_ok=True)  # Create results directory if it doesn't exist
    
    # Create mode-specific directory
    mode_dir = os.path.join('results', arguments.mode)
    os.makedirs(mode_dir, exist_ok=True)
    
    csv_filename = os.path.join(mode_dir, f'car_data_{time.strftime("%d_%H%M%S")}.csv')
    csv_headers = [
        'Step', 'Time',
        # Car State Fields
        'SpeedX', 'SpeedY', 'SpeedZ', 'TrackPos', 'Angle', 'RPM', 'Gear_State',
        'CurLapTime', 'DistFromStart', 'DistRaced', 'Fuel',
        'Damage', 'RacePos',
        # Car Control Fields
        'Accel', 'Brake', 'Steer', 'Gear_Control', 'Clutch', 'Meta'
    ]
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_headers)
        
        while True:
            # Wait for an answer from server
            buf = None
            try:
                buf, addr = sock.recvfrom(1000)
                buf = buf.decode()
                consecutive_timeouts = 0  # Reset on successful receive
            except socket.error as msg:
                consecutive_timeouts += 1
                if consecutive_timeouts >= 3:  # Only print after 3 consecutive timeouts
                    print(f"No server response after {consecutive_timeouts} attempts...")
                time.sleep(0.1)  # Short delay to avoid busy-waiting
                continue
            
            if verbose:
                print('Received:', buf)
            
            if buf and '***shutdown***' in buf:
                d.onShutDown()
                shutdownClient = True
                print('Client Shutdown')
                break
            
            if buf and '***restart***' in buf:
                d.onRestart()
                print('Client Restart')
                break
            
            currentStep += 1
            if currentStep != arguments.max_steps:
                if buf:
                    # Process sensor data and get control commands
                    buf = d.drive(buf)
                    
                    # Log all car state and control data to CSV
                    csv_writer.writerow([
                        currentStep,
                        time.strftime('%H:%M:%S'),
                        # Car State Values
                        f'{d.state.getSpeedX() or 0:.2f}',
                        f'{d.state.getSpeedY() or 0:.2f}',
                        f'{d.state.getSpeedZ() or 0:.2f}',
                        f'{d.state.getTrackPos() or 0:.2f}',
                        f'{d.state.getAngle() or 0:.2f}',
                        f'{d.state.getRpm() or 0:.0f}',
                        d.state.getGear() or 0,
                        f'{d.state.getCurLapTime() or 0:.2f}',
                        f'{d.state.getDistFromStart() or 0:.2f}',
                        f'{d.state.getDistRaced() or 0:.2f}',
                        f'{d.state.getFuel() or 0:.2f}',
                        f'{d.state.getDamage() or 0:.2f}',
                        d.state.getRacePos() or 0,
                        # Car Control Values
                        f'{d.control.getAccel():.2f}',
                        f'{d.control.getBrake():.2f}',
                        f'{d.control.getSteer():.2f}',
                        d.control.getGear(),
                        f'{d.control.getClutch():.2f}',
                        d.control.getMeta()
                    ])
                    
                    # Print important input and output fields every 10 steps
                    if currentStep % 10 == 0 or verbose:
                        #print(f"Step {currentStep}:")
                        #print(f"Input:  SpeedX: {d.state.getSpeedX():.2f} m/s | TrackPos: {d.state.getTrackPos():.2f} | Angle: {d.state.getAngle():.2f} rad | RPM: {d.state.getRpm():.0f} | Gear: {d.state.getGear()}")
                        #print(f"Output: Accel: {d.control.getAccel():.2f} | Brake: {d.control.getBrake():.2f} | Steer: {d.control.getSteer():.2f} | Gear: {d.control.getGear()}")
                        #print("---")
                        pass
            else:
                buf = '(meta 1)'
            
            if verbose:
                print('Sending:', buf)
            
            if buf:
                try:
                    sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
                except socket.error as msg:
                    print("Failed to send data...Exiting...")
                    sys.exit(-1)
    
    curEpisode += 1
    
    if curEpisode == arguments.max_episodes:
        shutdownClient = True

sock.close()