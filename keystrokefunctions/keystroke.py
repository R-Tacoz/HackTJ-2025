import os
from pynput import keyboard
import time
import csv

# Data structures to store timing information
keystroke_data = []  # For storing general keystroke data
key_pairs_data = []  # For storing inter-key latencies and dwell times for each key pair
last_key_time = None  # Time of the last key press
last_key = None  # Last key pressed for digraph latencies
typed_string = []  # To store the string the user has entered

# Time thresholds for digraph or other timing-based features (optional)
time_threshold = 0.5  # For relative timing differences

# Function to record the press event
def on_press(key):
    global last_key_time, last_key
    
    # Capture the time of the key press
    press_time = time.time()

    # Log the key press and the press time
    keystroke_data.append({'key': key, 'press_time': press_time, 'event': 'press'})

    # If the key is a printable character, add it to the string
    if hasattr(key, 'char') and key.char is not None:
        typed_string.append(key.char)

    # Calculate flight time (time from previous release to current press)
    if last_key_time:
        flight_time = press_time - last_key_time
        keystroke_data[-1]['flight_time'] = flight_time  # Store flight time

    # Track inter-key latency for every pair of keys
    if last_key:
        inter_key_latency = press_time - last_key_time
        # Append the inter-key latency and dwell time for this pair of keys
        key_pairs_data.append({
            'key_pair': (last_key, key),
            'inter_key_latency': inter_key_latency,
            'dwell_time': None  # We will calculate the dwell time when the key is released
        })

    # Update the last key and press time
    last_key = key
    last_key_time = press_time

# Function to record the release event
def on_release(key):
    release_time = time.time()

    # Log the release event and time
    keystroke_data.append({'key': key, 'release_time': release_time, 'event': 'release'})

    # Calculate dwell time (how long the key was pressed)
    for data in reversed(keystroke_data):
        if data['key'] == key and data['event'] == 'press':
            dwell_time = release_time - data['press_time']
            data['dwell_time'] = dwell_time  # Store dwell time
            break

    # Update the dwell time in the key_pairs_data for the current key pair
    for pair_data in key_pairs_data:
        if pair_data['key_pair'][1] == key and pair_data['dwell_time'] is None:
            pair_data['dwell_time'] = dwell_time

    # Stop the listener when the 'Esc' key is pressed
    if key == keyboard.Key.esc:
        write_to_csv()  # Write to CSV before exiting
        return False  # Stop the listener

# Function to write the keystroke data to a CSV file
def write_to_csv():
    # Include 'release_time' in the fieldnames for keystroke data
    keys = ['key', 'press_time', 'event', 'flight_time', 'dwell_time', 'release_time']
    with open('keystrokesdata.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        for data in keystroke_data:
            writer.writerow(data)
    
    # Write the inter-key latency and dwell time data to a separate CSV
    key_pair_keys = ['key_pair', 'inter_key_latency', 'dwell_time']
    with open('keypairsdata.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=key_pair_keys)
        writer.writeheader()
        for pair_data in key_pairs_data:
            writer.writerow(pair_data)

# Function to start the key logger
def start_keylogger():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

# Start the keylogger (run this in your script)
start_keylogger()

# Once the keylogger stops (e.g., after pressing 'Esc'), you can inspect the collected data
print("Keystroke data collected:")
for data in keystroke_data:
    print(data)

# Print the string typed by the user
typed_string_output = ''.join(typed_string)
print("\nString typed by user:", typed_string_output)

# Debugging: Print the current working directory to locate the CSV files
import os
print("\nCurrent working directory:", os.getcwd())
