import time
import csv
from pynput import keyboard
import hashlib

# Initialize variables
participant = "p001"
key_data = []
last_key_time = None
last_key = None
last_release_time = None

# Define output file
# output_file = "data/keylogger_data.csv"
output_file = "data/keylogger_data_new_training.csv"

FIXED_SALT = "d89a9c3f5bcad3a4fe908c12347a6f52"

# Hash key values
# Hash key values with the fixed salt
def hash_key(key):
    # Combine the key and fixed salt
    salted_key = f"{FIXED_SALT}{key}"
    # Hash the salted key
    hashed_key = hashlib.sha256(salted_key.encode()).hexdigest()
    return hashed_key

# Write header to CSV
with open(output_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "participant", "key1", "key2",
        "DU.key1.key1", "DD.key1.key2",
        "DU.key1.key2", "UD.key1.key2",
        "UU.key1.key2"
    ])

# Define key press event
def on_press(key):
    global last_key_time, last_key, last_release_time, key_data

    try:
        # Convert key to a readable format
        key_char = key.char if hasattr(key, "char") and key.char is not None else str(key)
    except Exception:
        key_char = str(key)

    current_time = time.time()

    if last_key_time is not None:
        # Calculate DU.key1.key1 (Duration of the previous key press)
        du_key1_key1 = current_time - last_key_time

        # Calculate DD.key1.key2 (Key Down-to-Key Down time)
        dd_key1_key2 = current_time - last_key_time

        # Calculate DU.key1.key2 (Key Down-to-Key Up time for consecutive keys)
        du_key1_key2 = current_time - last_release_time if last_release_time else 0

        # Calculate UD.key1.key2 (Key Up-to-Key Down time)
        ud_key1_key2 = last_release_time - last_key_time if last_release_time else 0

        # Calculate UU.key1.key2 (Key Up-to-Key Up time)
        uu_key1_key2 = current_time - last_release_time if last_release_time else 0

        # Append the data for the last key pair
        key_data.append([
            participant, hash_key(last_key), hash_key(key_char),
            round(du_key1_key1, 3), round(dd_key1_key2, 3),
            round(du_key1_key2, 3), round(ud_key1_key2, 3),
            round(uu_key1_key2, 3)
        ])

        # Write data to CSV in real-time
        with open(output_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(key_data[-1])

    # Update the last key and time
    last_key = key_char
    last_key_time = current_time

# Define key release event
def on_release(key):
    global last_release_time

    try:
        # Update the last release time
        last_release_time = time.time()
    except Exception:
        pass

    # Stop listener on Esc key
    if key == keyboard.Key.esc:
        return False

# Start listening to the keyboard
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    print("Keylogger started. Press 'Esc' to stop.")
    listener.join()

print("Keylogger stopped. Data saved to:", output_file)