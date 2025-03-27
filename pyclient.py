#!/usr/bin/env python3
import sys
import argparse
import socket
import csv
import time
import os
import driver  # Assuming driver.py is available

if __name__ == '__main__':
    pass

# Configure argument parser
parser = argparse.ArgumentParser(description='Python client to connect to the TORCS SCRC server in manual mode and log telemetry data to a CSV file.')
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

arguments = parser.parse_args()

# Print summary
print('Connecting to server host ip:', arguments.host_ip, '@ port:', arguments.host_port)
print('Bot ID:', arguments.id)
print('Maximum episodes:', arguments.max_episodes)
print('Maximum steps:', arguments.max_steps)
print('Track:', arguments.track)
print('Stage:', arguments.stage)
print('*********************************************')

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except socket.error as msg:
    print('Could not make a socket.')
    sys.exit(-1)

# One second timeout
sock.settimeout(1.0)

# Open CSV file in append mode for telemetry logging
csv_filename = "telemetry_log.csv"
file_exists = os.path.isfile(csv_filename) and os.path.getsize(csv_filename) > 0
csv_file = open(csv_filename, "a", newline="")
csv_writer = csv.writer(csv_file)

# If the file is new or empty, write the header
if not file_exists:
    header = ["timestamp", "angle", "curLapTime", "damage", "distFromStart",
              "distRaced", "fuel", "gear", "lastLapTime", "racePos", "rpm",
              "speedX", "speedY", "speedZ", "trackPos", "z"]
    csv_writer.writerow(header)
    csv_file.flush()

shutdownClient = False
curEpisode = 0

verbose = False

# Instantiate Driver for manual control only
d = driver.Driver(arguments.stage)

while not shutdownClient:
    while True:
        print('Sending id to server:', arguments.id)
        buf = arguments.id + d.init()
        print('Sending init string to server:', buf)
        
        try:
            sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
        except socket.error as msg:
            print("Failed to send data...Exiting...")
            sys.exit(-1)
            
        try:
            buf, addr = sock.recvfrom(1000)
            buf = buf.decode()
        except socket.error as msg:
            print("Didn't get response from server...")
    
        if '***identified***' in buf:
            print('Received:', buf)
            break

    currentStep = 0
    
    while True:
        # Wait for an answer from server
        buf = None
        try:
            buf, addr = sock.recvfrom(1000)
            buf = buf.decode()
        except socket.error as msg:
            print("Didn't get response from server...")
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

        # Log telemetry data: parse the received message using the driver's parser.
        telemetry = d.state.parser.parse(buf)
        if telemetry:
            # Extract fields with defaults in case a field is missing.
            row = [
                time.time(),
                telemetry.get("angle", [""])[0],
                telemetry.get("curLapTime", [""])[0],
                telemetry.get("damage", [""])[0],
                telemetry.get("distFromStart", [""])[0],
                telemetry.get("distRaced", [""])[0],
                telemetry.get("fuel", [""])[0],
                telemetry.get("gear", [""])[0],
                telemetry.get("lastLapTime", [""])[0],
                telemetry.get("racePos", [""])[0],
                telemetry.get("rpm", [""])[0],
                telemetry.get("speedX", [""])[0],
                telemetry.get("speedY", [""])[0],
                telemetry.get("speedZ", [""])[0],
                telemetry.get("trackPos", [""])[0],
                telemetry.get("z", [""])[0]
            ]
            csv_writer.writerow(row)
            csv_file.flush()
        
        currentStep += 1
        if currentStep != arguments.max_steps:
            if buf:
                buf = d.drive(buf)
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
csv_file.close()
print("Telemetry logged to", csv_filename)
