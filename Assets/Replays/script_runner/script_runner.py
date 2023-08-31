import time
import tkinter as tk
from tkinter import filedialog
import os
import random
import watchdog.observers
import watchdog.events
import csv

# Genetic algorithm parameters
num_children = 5
param_names = ["speed", "rawThrottle", "rawBrake", "brakePressure", "throttle"]

def modify_file_content(base_content):
    lines = base_content.split("\n")
    modified_content = []

    for line in lines:
        line = line.strip()
        if line.startswith("- speed"):
            value = float(line.split(":")[1].strip())
            new_value = value * (1 + random.uniform(-0.2, 0.2))
            new_value = max(0, min(100, new_value))
            modified_content.append(f"  - speed: {new_value:.6f}")
        elif any(line.strip().startswith(param) for param in param_names):
            param, value = line.split(":")[0].strip(), float(line.split(":")[1].strip())
            new_value = value * (1 + random.uniform(-0.2, 0.2))

            if param == "rawThrottle" or param == "rawBrake":
                new_value = int(max(0, min(10000, new_value)))
            elif param == "throttle":
                new_value = max(0, min(100, new_value))
            elif param == "brakePressure":
                new_value = max(0, min(1, new_value))

            modified_content.append(f"    {param}: {new_value:.6f}")
        else:
            modified_content.append(line)

    return "\n".join(modified_content)

def generate_next_generation(base_file, generation):
    base_content = load_base_input_file(base_file)
    gen_folder = os.path.join(os.path.dirname(base_file), f"generation_{generation}")
    os.makedirs(gen_folder, exist_ok=True)

    for child in range(num_children):
        modified_content = modify_file_content(base_content)
        with open(os.path.join(gen_folder, f"generation{generation}_modified{child}.asset"), 'w') as file:
            file.write(modified_content)

def load_base_input_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def on_created(event):
    if event.src_path.endswith('.csv'):
        print(f"Detected new CSV: {event.src_path}")
        best_two_files = select_best_times(event.src_path)
        print(f"Best two files from CSV: {best_two_files}")

def select_best_times(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        times = [(row[0], float(row[1])) for row in reader]
    times.sort(key=lambda x: x[1])
    return times[:2]

class MyHandler(watchdog.events.FileSystemEventHandler):
    def on_created(self, event):
        on_created(event)

def run_genetic_algorithm():
    base_file = input_file_entry.get()
    generation = 0

    while True:
        print(f"Generating generation {generation}")
        generate_next_generation(base_file, generation)
        generation += 1

        # Wait for CSV result
        print(f"Waiting for CSV for generation {generation}")
        path_to_watch = os.path.dirname(base_file)
        event_handler = MyHandler()
        observer = watchdog.observers.Observer()
        observer.schedule(event_handler, path_to_watch, recursive=False)
        observer.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

# GUI setup
root = tk.Tk()
root.title("Genetic Algorithm for Input File Generation")

input_file_label = tk.Label(root, text="Select Base Input File:")
input_file_label.pack()
input_file_entry = tk.Entry(root)
input_file_entry.pack()

def browse_input_file():
    file_path = filedialog.askopenfilename()
    input_file_entry.delete(0, tk.END)
    input_file_entry.insert(0, file_path)

browse_input_button = tk.Button(root, text="Browse", command=browse_input_file)
browse_input_button.pack()

run_button = tk.Button(root, text="Run Genetic Algorithm", command=run_genetic_algorithm)
run_button.pack()

root.mainloop()
