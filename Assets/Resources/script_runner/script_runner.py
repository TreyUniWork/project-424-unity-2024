import tkinter as tk
from tkinter import filedialog
import os
import random
import csv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Define the genetic algorithm parameters
num_children = 5
num_generations = 100
param_limits = {
    "speed": (0, 100),  # float
    "rawThrottle": (0, 10000),  # int
    "rawBrake": (0, 10000),  # int
    "throttle": (0, 100),  # float
    "brakePressure": (0, 1),  # float
}


class CSVHandler(FileSystemEventHandler):
    def __init__(self, observer):
        self.observer = observer

    def on_created(self, event):
        if event.is_directory:
            return
        elif event.src_path.endswith(".csv"):
            print(f"Detected new CSV: {event.src_path}")
            self.observer.stop()


def load_base_input_file(file_path):
    print(f"Loading base input file from: {file_path}")
    with open(file_path, "r") as file:
        content = file.read()
    return content


def read_csv_laptimes(file_path):
    lap_times = {}
    with open(file_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip header row
        for row in reader:
            filename, laptime_str = row[0], row[1]
            seconds, milliseconds = map(int, laptime_str.split("."))
            laptime = seconds + milliseconds * 0.001
            lap_times[filename] = laptime
    return lap_times


def watch_for_csv(directory):
    print(f"Watching for CSV files in: {directory}")
    observer = Observer()
    event_handler = CSVHandler(observer)
    observer.schedule(event_handler, directory, recursive=False)
    observer.start()
    observer.join()


def modify_parameter(param, value):
    limit = param_limits[param]
    try:
        mutation_percentage = float(entries[param].get()) / 100
    except ValueError:
        print(f"Invalid mutation percentage for {param}. Using 0%.")
        mutation_percentage = 0

    if param in ["rawThrottle", "rawBrake"]:
        new_value = int(value * (1 + random.uniform(-mutation_percentage, mutation_percentage)))
    else:
        new_value = value * (1 + random.uniform(-mutation_percentage, mutation_percentage))

    return max(limit[0], min(limit[1], new_value))


def extract_params_from_content(content):
    params = {}
    lines = content.split("\n")
    for line in lines:
        for param in param_limits:
            if line.strip().startswith(f"- {param}"):
                value = float(line.split(":")[1].strip())
                params[param] = value
                break
    return params


def crossover_and_mutate(parent1_content, parent2_content):
    parent1_params = extract_params_from_content(parent1_content)
    parent2_params = extract_params_from_content(parent2_content)

    child = {}
    for param, limits in param_limits.items():
        if random.uniform(0, 1) < 0.5:
            value = parent1_params[param]
        else:
            value = parent2_params[param]
        child[param] = modify_parameter(param, value)
    return child

def convert_params_to_content(params):
    content = []
    for param, value in params.items():
        content.append(f"  - {param}: {value:.6f}")
    return "\n".join(content)

def run_genetic_algorithm():
    base_input_file = input_file_entry.get()
    base_input_content = load_base_input_file(base_input_file)
    lines = base_input_content.split("\n")
    script_location = os.path.dirname(os.path.abspath(__file__))

    for generation in range(num_generations):
        output_folder = os.path.join(
            script_location, "..", "GeneticAssets", f"GEN{generation + 1}"
        )
        os.makedirs(output_folder, exist_ok=True)
        if generation == 0:
            current_generation = [base_input_content] * num_children
        else:
            current_generation = next_generation.copy()

        for child_index, child_content in enumerate(current_generation):
            lines = child_content.split("\n")
            modified_content = []
            for line in lines:
                for param, limits in param_limits.items():
                    if line.strip().startswith(f"- {param}"):
                        value = float(line.split(":")[1].strip())
                        new_value = modify_parameter(param, value)
                        modified_content.append(f"  - {param}: {new_value:.6f}")
                        break
                else:
                    modified_content.append(line)

            output_file_name = f"gen{generation + 1}asset{child_index + 1}.asset"
            output_file_path = os.path.join(output_folder, output_file_name)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            with open(output_file_path, "w") as file:
                file.write("\n".join(modified_content))

        watch_for_csv(script_location)

        csv_files = [
            os.path.join(script_location, file)
            for file in os.listdir(script_location)
            if file.endswith(".csv")
        ]
        latest_csv = max(csv_files, key=os.path.getctime)
        lap_times = read_csv_laptimes(latest_csv)

        best_parents_filenames = sorted(lap_times, key=lap_times.get)[:2]
        best_parents_content = [load_base_input_file(filename) for filename in best_parents_filenames]
        print(f"Best parents for next generation are: {best_parents_filenames}")

        next_generation = []
        for _ in range(num_children):
            child_params = crossover_and_mutate(best_parents_content[0], best_parents_content[1])
            child_content = convert_params_to_content(child_params)
            next_generation.append(child_content)


# GUI setup
root = tk.Tk()
root.title("Genetic Algorithm for Asset File Generation")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

load_button = tk.Button(frame, text="Load Base Input File",
                        command=lambda: input_file_entry.insert(0, filedialog.askopenfilename()))
load_button.grid(row=0, column=0, padx=5, pady=5)

input_file_entry = tk.Entry(frame, width=50)
input_file_entry.grid(row=0, column=1, padx=5, pady=5)

run_button = tk.Button(frame, text="Run Genetic Algorithm", command=run_genetic_algorithm)
run_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

entries = {}
for index, (param, limits) in enumerate(param_limits.items()):
    label = tk.Label(frame, text=f"{param} mutation %:")
    label.grid(row=index + 2, column=0, padx=5, pady=5)

    entry = tk.Entry(frame)
    entry.grid(row=index + 2, column=1, padx=5, pady=5)
    entries[param] = entry

root.mainloop()
