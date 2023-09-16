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
    "throttle": (0, 100),  # float
    "brakePressure": (0, 100),  # float
}

# The maximum percentage difference the param can change by
# First number = min, second = max
param_modification_limits = {
    "speed": (-0.05, 0.05),
    "rawThrottle": (-0.05, 0.05),
    "throttle": (-0.05, 0.05),
    "brakePressure": (-0.05, 0.05),
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
    script_directory = os.path.dirname(
        os.path.abspath(__file__)
    )  # Get the absolute path of the script's directory
    target_file_path = os.path.abspath(os.path.join(script_directory, file_path))

    print(f"Loading base input file from: {target_file_path}")

    try:
        with open(target_file_path, "r") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"File not found: {target_file_path}")
        return None


def read_csv_laptimes(file_path):
    lap_times = {}
    with open(file_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip header row
        for row in reader:
            filename, laptime_str = row[0], row[1]
            laptime = float(laptime_str)
            lap_times[filename] = laptime
    return lap_times


def watch_for_csv(directory):
    print(f"Watching for CSV files in: {directory}")
    observer = Observer()
    event_handler = CSVHandler(observer)
    observer.schedule(event_handler, directory, recursive=False)
    observer.start()
    observer.join()


def modify_params(lines):
    modified_lines = []

    # Init multipliers
    # speed
    min_speed_mod, max_speed_mod = param_modification_limits["speed"]
    new_speed_multi = random.uniform(min_speed_mod, max_speed_mod)

    # rawThrottle
    min_raw_throttle_mod, max_raw_throttle_mod = param_modification_limits[
        "rawThrottle"
    ]
    new_raw_throttle_multi = random.uniform(min_raw_throttle_mod, max_raw_throttle_mod)

    # throttle
    min_throttle_mod, max_throttle_mod = param_modification_limits["throttle"]
    new_throttle_multi = random.uniform(min_throttle_mod, max_throttle_mod)

    # brakePressure
    min_brake_pressure_mod, max_brake_pressure_mod = param_modification_limits[
        "brakePressure"
    ]
    new_brake_pressure_multi = random.uniform(
        min_brake_pressure_mod, max_brake_pressure_mod
    )

    for line in lines:
        stripped_line = line.strip()
        # Adjust speed
        if stripped_line.startswith("- speed:"):
            speed_str = stripped_line.split(":")[1].strip()
            speed = float(speed_str)
            new_speed = speed * (1 + new_speed_multi)
            if "speed" in param_limits:
                min_limit, max_limit = param_limits["speed"]
                if new_speed < min_limit:
                    new_speed = min_limit
                elif new_speed > max_limit:
                    new_speed = max_limit

            modified_lines.append(f"  - speed: {new_speed:.2f}")
        # Adjust rawThrottle with given formula
        elif stripped_line.startswith("rawThrottle:"):
            raw_throttle_str = line.split(":")[1].strip()
            raw_throttle = float(raw_throttle_str)
            new_raw_throttle = raw_throttle * (1 + new_raw_throttle_multi)
            if "rawThrottle" in param_limits:
                min_limit, max_limit = param_limits["rawThrottle"]
                if new_raw_throttle < min_limit:
                    new_raw_throttle = min_limit
                elif new_raw_throttle > max_limit:
                    new_raw_throttle = max_limit

            modified_lines.append(f"    rawThrottle: {new_raw_throttle:.2f}")
        # Adjust throttle to be closer to maximum without maxing out
        elif stripped_line.startswith("throttle:"):
            throttle_str = stripped_line.split(":")[1].strip()
            throttle = float(throttle_str)
            if "throttle" in param_limits:
                min_limit, max_limit = param_limits["throttle"]
                new_throttle = throttle * (1 + new_throttle_multi)
                if new_throttle < min_limit:
                    new_throttle = min_limit
                elif new_throttle > max_limit:
                    new_throttle = max_limit

            modified_lines.append(f"    throttle: {new_throttle:.2f}")
        # Slightly reduce brakePressure to allow for faster deceleration when needed
        elif stripped_line.startswith("brakePressure:"):
            brake_pressure_str = stripped_line.split(":")[1].strip()
            brake_pressure = float(brake_pressure_str)
            if "brakePressure" in param_limits:
                min_limit, max_limit = param_limits["brakePressure"]
                new_brake_pressure = brake_pressure * (1 + new_brake_pressure_multi)
                if new_brake_pressure < min_limit:
                    new_brake_pressure = min_limit
                elif new_brake_pressure > max_limit:
                    new_brake_pressure = max_limit

            modified_lines.append(f"    brakePressure: {new_brake_pressure:.2f}")
        else:
            modified_lines.append(line)

    return modified_lines


# Define a function to run the genetic algorithm
def run_genetic_algorithm():
    # Get the input file path from an entry field
    base_input_file = input_file_entry.get()

    # Load the content of the input file
    base_input_content = load_base_input_file(base_input_file)

    # Split the input content into lines
    # lines = base_input_content.split("\n")

    # Get the directory where the script is located
    script_location = os.path.dirname(os.path.abspath(__file__))

    # Iterate through a specified number of generations
    for generation in range(num_generations):
        print(f"Generation {generation + 1}:")

        # Create an output folder for the current generation
        output_folder = os.path.join(
            script_location, "..", "GeneticAssets", f"GEN{generation + 1}"
        )
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created output folder: {output_folder}")

        current_generation = [base_input_content] * num_children

        # Iterate through each child in the current generation
        for child_index, child_content in enumerate(current_generation):
            print(f"Child {child_index+1}")
            lines = child_content.split("\n")
            modified_content = []

            # Skip modification for the first child (child_index == 0)
            if child_index == 0:
                modified_content = lines
            else:
                # Modify parameters within the child's content
                modified_content = modify_params(lines)
                print(f"Modified child {child_index + 1} content")

            # Generate an output file name for each child and write the modified content to a file
            output_file_name = f"gen{generation + 1}asset{child_index + 1}.asset"
            output_file_path = os.path.join(output_folder, output_file_name)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            with open(output_file_path, "w") as file:
                file.write("\n".join(modified_content))
            print(f"Created output file: {output_file_path}")

        # Call a function to watch for CSV files
        watch_for_csv(script_location)
        print("Watching for CSV files...")

        # Scan the script's directory for CSV files and select the latest one
        csv_files = [
            os.path.join(script_location, file)
            for file in os.listdir(script_location)
            if file.endswith(".csv")
        ]
        latest_csv = max(csv_files, key=os.path.getctime)
        print(
            f"Selected latest CSV file: {latest_csv} (Shouldn't take long so close if it's frozen)"
        )

        # Read lap times from the selected CSV file
        lap_times = {}
        while not lap_times or len(lap_times) < 5:
            lap_times = read_csv_laptimes(latest_csv)

        print(f"Read lap times from CSV file")

        # Get the laptime of the best parent and update the base file for the next generation
        if lap_times:
            best_parent_filename = min(lap_times, key=lap_times.get)
            base_input_content = load_base_input_file(best_parent_filename)
            print(f"Lap Times: {lap_times}")
            print(f"Selected best parent file: {best_parent_filename}\n")
        else:
            print(f"Error reading laptimes: {lap_times}")

        # Identify the two best parents from the previous generation based on lap times
        # best_parents_filenames = sorted(lap_times, key=lap_times.get)[:2]
        # best_parents_content = [
        #     load_base_input_file(filename) for filename in best_parents_filenames
        # ]

        # # Print the names of the best parents for the next generation
        # print(f"Best parents for next generation are: {best_parents_filenames}")

        # # Create a new next generation by performing crossover and mutation on the best parents' content
        # next_generation = []
        # for _ in range(num_children):
        #     child_params = crossover_and_mutate(
        #         best_parents_content[0], best_parents_content[1]
        #     )
        #     child_content = convert_params_to_content(child_params)
        #     next_generation.append(child_content)


# GUI setup
root = tk.Tk()
root.title("Genetic Algorithm for Asset File Generation")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

load_button = tk.Button(
    frame,
    text="Load Base Input File",
    command=lambda: input_file_entry.insert(0, filedialog.askopenfilename()),
)
load_button.grid(row=0, column=0, padx=5, pady=5)

input_file_entry = tk.Entry(frame, width=50)
input_file_entry.grid(row=0, column=1, padx=5, pady=5)

run_button = tk.Button(
    frame, text="Run Genetic Algorithm", command=run_genetic_algorithm
)
run_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

root.mainloop()
