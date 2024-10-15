import tkinter as tk
from tkinter import filedialog
import os
import random
import csv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# Define the genetic algorithm parameters
num_children = 5
num_generations = 30
param_limits = {
    "speed": (0, 100),  # float
    "rawThrottle": (0, 10000),  # int
    "throttle": (0, 100),  # float
    "brakePressure": (0, 100),  # float
    "steeringAngle": (0, 300),
    "rawSteer": (0, 300),
}

# The maximum percentage difference the param can change by
# First number = min, second = max

# MUTATION RATE
param_modification_limits = {
    "speed": (-0.05, 0.05),
    "rawThrottle": (-0.05, 0.05),
    "throttle": (-0.05, 0.05),
    "brakePressure": (-0.05, 0.05),
    "steeringAngle": (-0.1, 0.1),
    "rawSteer": (-0.1, 0.1),
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
    # Get the absolute path of the script's directory
    script_directory = os.path.dirname(os.path.abspath(__file__))
    target_file_path = os.path.abspath(
        os.path.join(script_directory, file_path))

    print(f"Loading base input file from: {target_file_path}")

    try:
        with open(target_file_path, "r") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"File not found: {target_file_path}")
        return None


def read_csv_laptimes_with_sectors(file_path):
    lap_times = {}
    sector_times = {}

    with open(file_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        headers = next(reader)  # Skip header row
        
        for row in reader:
            filename = row[0]
            lap_time = float(row[1])
            sectors = [float(row[i + 2]) for i in range(len(row) - 2)]  # Assuming sectors start from column 3
            lap_times[filename] = lap_time
            sector_times[filename] = sectors

    return lap_times, sector_times

def select_best_sectors_steering(sector_times):
    best_sectors = []
    num_sectors = len(next(iter(sector_times.values())))  # Get number of sectors from the first entry

    # Initialize best sector times for steering-related parameters
    for i in range(num_sectors):
        best_sectors.append((None, float('inf')))  # (child, best time)

    # Find the best child for each sector based on sector time
    for child, sectors in sector_times.items():
        for i, sector_time in enumerate(sectors):
            if sector_time < best_sectors[i][1]:  # Compare sector times
                best_sectors[i] = (child, sector_time)

    return best_sectors  # [(best_child_for_sector_1, time), (best_child_for_sector_2, time), ...]

def create_hybrid_child(best_sectors, best_lap_child, best_parents_content):
    # Extract params from the best lap child (for non-steering parameters)
    best_lap_params = extract_params(best_parents_content[best_lap_child])
    new_child_params = best_lap_params.copy()

    # Override steeringAngle and rawSteer based on the best sectors
    for i, (best_child, _) in enumerate(best_sectors):
        best_sector_content = best_parents_content[best_child]
        best_sector_params = extract_params(best_sector_content)

        # Override steering-related parameters from the best sector
        new_child_params["steeringAngle"] = best_sector_params["steeringAngle"]
        new_child_params["rawSteer"] = best_sector_params["rawSteer"]

    return convert_params_to_content(new_child_params)



def watch_for_csv(directory):
    print(f"Watching for CSV files in: {directory}")
    observer = Observer()
    event_handler = CSVHandler(observer)
    observer.schedule(event_handler, directory, recursive=False)
    observer.start()
    observer.join()


# Dynamic steeringAngle and rawSteer based on speed
def get_dynamic_steering_mod(speed, param_name):
    if speed < 40:
        return random.uniform(-0.15, 0.15)
    elif 40 <= speed <= 80:
        return random.uniform(*param_modification_limits[param_name])
    else:
        return random.uniform(-0.05, 0.05)

def modify_params(lines):
    modified_lines = []

    current_speed = None

    for line in lines:
        stripped_line = line.strip()
        # Adjust speed
        if stripped_line.startswith("- speed:"):
            speed_str = stripped_line.split(":")[1].strip()
            current_speed = float(speed_str)
            new_speed_multi = random.uniform(*param_modification_limits["speed"])
            new_speed = current_speed * (1 + new_speed_multi)
            if "speed" in param_limits:
                min_limit, max_limit = param_limits["speed"]
                new_speed = max(min(new_speed, max_limit), min_limit)

            modified_lines.append(f"  - speed: {new_speed:.2f}")
        # Adjust rawThrottle
        elif stripped_line.startswith("rawThrottle:"):
            raw_throttle_str = stripped_line.split(":")[1].strip()
            raw_throttle = float(raw_throttle_str)
            new_raw_throttle_multi = random.uniform(*param_modification_limits["rawThrottle"])
            new_raw_throttle = raw_throttle * (1 + new_raw_throttle_multi)
            if "rawThrottle" in param_limits:
                min_limit, max_limit = param_limits["rawThrottle"]
                new_raw_throttle = max(min(new_raw_throttle, max_limit), min_limit)

            modified_lines.append(f"    rawThrottle: {new_raw_throttle:.2f}")
        # Adjust throttle
        elif stripped_line.startswith("throttle:"):
            throttle_str = stripped_line.split(":")[1].strip()
            throttle = float(throttle_str)
            new_throttle_multi = random.uniform(*param_modification_limits["throttle"])
            new_throttle = throttle * (1 + new_throttle_multi)
            if "throttle" in param_limits:
                min_limit, max_limit = param_limits["throttle"]
                new_throttle = max(min(new_throttle, max_limit), min_limit)

            modified_lines.append(f"    throttle: {new_throttle:.2f}")
        # Adjust brakePressure
        elif stripped_line.startswith("brakePressure:"):
            brake_pressure_str = stripped_line.split(":")[1].strip()
            brake_pressure = float(brake_pressure_str)
            new_brake_pressure_multi = random.uniform(*param_modification_limits["brakePressure"])
            new_brake_pressure = brake_pressure * (1 + new_brake_pressure_multi)
            if "brakePressure" in param_limits:
                min_limit, max_limit = param_limits["brakePressure"]
                new_brake_pressure = max(min(new_brake_pressure, max_limit), min_limit)

            modified_lines.append(f"    brakePressure: {new_brake_pressure:.2f}")
        # Adjust steeringAngle based on speed
        elif stripped_line.startswith("steeringAngle:"):
            steering_angle_str = stripped_line.split(":")[1].strip()
            steering_angle = float(steering_angle_str)
            new_steering_angle_multi = get_dynamic_steering_mod(current_speed, "steeringAngle")
            new_steering_angle = steering_angle * (1 + new_steering_angle_multi)
            if "steeringAngle" in param_limits:
                min_limit, max_limit = param_limits["steeringAngle"]
                new_steering_angle = max(min(new_steering_angle, max_limit), min_limit)

            modified_lines.append(f"    steeringAngle: {new_steering_angle:.2f}")
        # Adjust rawSteer based on speed
        elif stripped_line.startswith("rawSteer:"):
            raw_steer_str = stripped_line.split(":")[1].strip()
            raw_steer = float(raw_steer_str)
            new_raw_steer_multi = get_dynamic_steering_mod(current_speed, "rawSteer")
            new_raw_steer = raw_steer * (1 + new_raw_steer_multi)
            if "rawSteer" in param_limits:
                min_limit, max_limit = param_limits["rawSteer"]
                new_raw_steer = max(min(new_raw_steer, max_limit), min_limit)

            modified_lines.append(f"    rawSteer: {new_raw_steer:.2f}")
        else:
            modified_lines.append(line)

    return modified_lines


# TO WRITE CSV FILES

# THIS ISN'T BEING CALLED IN THE CODE


def write_lap_times_to_csv(output_folder, generation, child_index, laptime):
    csv_file = os.path.join(
        output_folder, f"generation_{generation+1}_lap_times.csv")
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, "a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Child", "LapTime"])
        writer.writerow(
            [f"gen{generation + 1}_child{child_index + 1}.asset", laptime])

# THIS IS ALSO NOT CALLED IN THE CURRENT CODE
# THIS WILL SUPPOSEDLY HANDLE THE SIMULATIONS FOR EACH CHILD


def simulate_child(child_index, child_content, output_folder, generation):
    print(f"Simulating Child {child_index + 1}")
    lines = child_content.split("\n")

    # Modify parameters for the child
    modified_content = modify_params(lines) if child_index > 0 else lines

    # Generate output file name
    output_file_name = f"gen{generation + 1}asset{child_index + 1}.asset"
    output_file_path = os.path.join(output_folder, output_file_name)

    # Write modified content to a file
    with open(output_file_path, "w") as file:
        file.write("\n".join(modified_content))

    print(f"Created output file: {output_file_path}")


# MUTATION LOGIC
def crossover_and_mutate(parent1_content, parent2_content):
    # Example: Extract parameters from the content (you may need to adjust this)
    parent1_params = extract_params(parent1_content)
    parent2_params = extract_params(parent2_content)

    # Simple crossover logic (mix parameters)
    child_params = {}
    for param in param_limits.keys():
        if random.random() < 0.6:  # 60% chance to take from parent 1 or parent 2
            child_params[param] = parent1_params[param]
        else:
            child_params[param] = parent2_params[param]

    # Mutation
    for param in param_limits.keys():
        if random.random() < 0.05:  # 5% chance to mutate
            child_params[param] += random.uniform(
                param_modification_limits[param][0], param_modification_limits[param][1])
            # Ensure it stays within limits
            child_params[param] = max(param_limits[param][0], min(
                param_limits[param][1], child_params[param]))

    return child_params

# TO EXTRACT THE PARAMETERS


def extract_params(content):
    params = {}
    for line in content.splitlines():
        if "speed:" in line:
            params["speed"] = float(line.split(":")[1].strip())
        elif "rawThrottle:" in line:
            params["rawThrottle"] = float(line.split(":")[1].strip())
        elif "throttle:" in line:
            params["throttle"] = float(line.split(":")[1].strip())
        elif "brakePressure:" in line:
            params["brakePressure"] = float(line.split(":")[1].strip())
        elif "steeringAngle:" in line:
            params["steeringAngle"] = float(line.split(":")[1].strip())
        elif "rawSteer:" in line:
            params["rawSteer"] = float(line.split(":")[1].strip())
    return params

# HELPER FUNCTION


def convert_params_to_content(params):
    content_lines = []
    content_lines.append("- speed: {:.2f}".format(params["speed"]))
    content_lines.append("  rawThrottle: {:.2f}".format(params["rawThrottle"]))
    content_lines.append("  throttle: {:.2f}".format(params["throttle"]))
    content_lines.append(
        "  brakePressure: {:.2f}".format(params["brakePressure"]))
    content_lines.append(
        "  steeringAngle: {:.2f}".format(params["steeringAngle"]))
    content_lines.append("  rawSteer: {:.2f}".format(params["rawSteer"]))
    return "\n".join(content_lines)


# Define a function to run the genetic algorithm
def run_genetic_algorithm():
    base_input_file = input_file_entry.get()
    base_input_content = load_base_input_file(base_input_file)
    script_location = os.path.dirname(os.path.abspath(__file__))

    for generation in range(num_generations):
        print(f"Generation {generation + 1}:")
        output_folder = os.path.join(script_location, "..", "GeneticAssets", f"GEN{generation + 1}")
        os.makedirs(output_folder, exist_ok=True)

        current_generation = [base_input_content] * num_children

        for child_index, child_content in enumerate(current_generation):
            lines = child_content.split("\n")
            modified_content = modify_params(lines) if child_index > 0 else lines

            output_file_name = f"gen{generation + 1}asset{child_index + 1}.asset"
            output_file_path = os.path.join(output_folder, output_file_name)
            with open(output_file_path, "w") as file:
                file.write("\n".join(modified_content))

        watch_for_csv(script_location)
        csv_files = [os.path.join(script_location, file) for file in os.listdir(script_location) if file.endswith(".csv")]

        if not csv_files:
            print("No CSV files found!")
            return

        latest_csv = max(csv_files, key=os.path.getctime)
        lap_times, sector_times = read_csv_laptimes_with_sectors(latest_csv)

        if lap_times:
            # Select the best child based on total lap time (for non-steering parameters)
            best_lap_child = min(lap_times, key=lap_times.get)

            # Select the best child for each sector based on sector times (for steering-related parameters)
            best_sectors = select_best_sectors_steering(sector_times)

            # Load content of best parents
            best_parents_content = {child: load_base_input_file(f"gen{generation + 1}asset{child}.asset") for child in lap_times.keys()}

            # Create a new child based on best lap time for non-steering and best sectors for steering
            next_generation = []
            for _ in range(num_children):
                new_child_content = create_hybrid_child(best_sectors, best_lap_child, best_parents_content)
                next_generation.append(new_child_content)



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
