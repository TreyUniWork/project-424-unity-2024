import tkinter as tk
from tkinter import filedialog
import os
import random
import csv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Define the genetic algorithm parameters
num_children = 5
num_generations = 4
param_names = ["speed", "rawThrottle", "rawBrake", "brakePressure", "throttle"]


class CSVHandler(FileSystemEventHandler):
    def __init__(self, observer):
        self.observer = observer

    def on_created(self, event):
        if event.is_directory:
            return
        elif event.src_path.endswith('.csv'):
            print(f"Detected new CSV: {event.src_path}")
            self.observer.stop()


def load_base_input_file(file_path):
    print(f"Loading base input file from: {file_path}")
    with open(file_path, 'r') as file:
        content = file.read()
    return content


def read_csv_laptimes(file_path):
    lap_times = {}
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip header row
        for row in reader:
            filename, laptime_str = row[0], row[1]
            minutes, seconds_milliseconds = laptime_str.split(":")
            seconds, milliseconds = seconds_milliseconds.split(".")
            laptime = int(minutes) * 60 + int(seconds) + int(milliseconds) * 0.001
            lap_times[filename] = laptime
    return lap_times



def watch_for_csv(directory):
    print(f"Watching for CSV files in: {directory}")
    observer = Observer()
    event_handler = CSVHandler(observer)
    observer.schedule(event_handler, directory, recursive=False)
    observer.start()
    observer.join()


def run_genetic_algorithm():
    base_input_file = input_file_entry.get()
    print(f"Starting the genetic algorithm with base input file: {base_input_file}")
    base_input_content = load_base_input_file(base_input_file)
    lines = base_input_content.split("\n")
    script_location = os.path.dirname(os.path.abspath(__file__))

    for generation in range(num_generations):
        print(f"Generating children for generation {generation}")
        output_folder = os.path.join(script_location, f"generation_{generation}")
        os.makedirs(output_folder, exist_ok=True)

        for child_index in range(num_children):
            modified_content = []
            for line in lines:
                line = line.strip()
                if line.startswith("- speed"):
                    value = float(line.split(":")[1].strip())
                    new_value = value * (1 + random.uniform(-0.2, 0.2))
                    modified_content.append(f"  - speed: {new_value:.6f}")
                elif any(line.strip().startswith(param) for param in param_names):
                    param, value = line.split(":")[0].strip(), float(line.split(":")[1].strip())
                    new_value = value * (1 + random.uniform(-0.2, 0.2))
                    modified_content.append(f"    {param}: {new_value:.6f}")
                else:
                    modified_content.append(f"    {line}")

            output_file_name = f"generation{generation}_modified{child_index}.asset"
            output_file_path = os.path.join(output_folder, output_file_name)
            with open(output_file_path, 'w') as file:
                file.write("\n".join(modified_content))

        print(f"Waiting for Unity to run generation {generation} and produce a CSV with results...")
        watch_for_csv(script_location)

        csv_files = [file for file in os.listdir(script_location) if file.endswith('.csv')]
        latest_csv = max(csv_files, key=os.path.getctime)
        print(f"Found latest CSV: {latest_csv}")
        lap_times = read_csv_laptimes(os.path.join(script_location, latest_csv))
        best_parents = sorted(lap_times, key=lap_times.get)[:2]
        print(f"Best parents for next generation are: {best_parents}")


# GUI
root = tk.Tk()
root.title("Genetic Algorithm for Input File Generation")

input_file_label = tk.Label(root, text="Select Base Input File:")
input_file_label.pack()
input_file_entry = tk.Entry(root)
input_file_entry.pack()


def browse_input_file():
    input_file_path = filedialog.askopenfilename()
    if input_file_path:
        print(f"Selected input file: {input_file_path}")
        input_file_entry.delete(0, tk.END)
        input_file_entry.insert(0, input_file_path)


browse_input_button = tk.Button(root, text="Browse", command=browse_input_file)
browse_input_button.pack()

run_genetic_algorithm_button = tk.Button(root, text="Run Genetic Algorithm", command=run_genetic_algorithm)
run_genetic_algorithm_button.pack()

root.mainloop()
