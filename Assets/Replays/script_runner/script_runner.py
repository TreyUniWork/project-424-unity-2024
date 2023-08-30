import tkinter as tk
from tkinter import filedialog
import os
import random

# Define the genetic algorithm parameters
num_children = 5
num_generations = 1
param_names = ["speed", "rawThrottle", "rawBrake", "brakePressure", "throttle"]


# Define the base input file (load and export)
def load_base_input_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content


def run_genetic_algorithm():
    base_input_file = input_file_entry.get()
    base_input_content = load_base_input_file(base_input_file)
    lines = base_input_content.split("\n")

    script_location = os.path.dirname(os.path.abspath(__file__))

    for generation in range(num_generations):
        output_folder = os.path.join(script_location, f"generation_{generation}")
        os.makedirs(output_folder, exist_ok=True)

        for child_index in range(num_children):
            modified_content = []
            for line in lines:
                line = line.strip()
                if line.startswith("- speed"):
                    value = float(line.split(":")[1].strip())
                    new_value = value * (1 + random.uniform(-0.2, 0.2))  # Example percentage range (-20% to +20%)
                    modified_content.append(f"  - speed: {new_value:.6f}")
                elif any(line.strip().startswith(param) for param in param_names):
                    param, value = line.split(":")[0].strip(), float(line.split(":")[1].strip())
                    new_value = value * (1 + random.uniform(-0.2, 0.2))  # Example percentage range (-20% to +20%)
                    modified_content.append(f"    {param}: {new_value:.6f}")
                else:
                    modified_content.append(f"    {line}")

            output_file_name = f"generation{generation}_modified{child_index}.asset"
            output_file_path = os.path.join(output_folder, output_file_name)
            with open(output_file_path, 'w') as file:
                file.write("\n".join(modified_content))


# Create the main window
root = tk.Tk()
root.title("Genetic Algorithm for Input File Generation")

# Input file selection
input_file_label = tk.Label(root, text="Select Base Input File:")
input_file_label.pack()
input_file_entry = tk.Entry(root)
input_file_entry.pack()


def browse_input_file():
    input_file_path = filedialog.askopenfilename()
    input_file_entry.delete(0, tk.END)
    input_file_entry.insert(0, input_file_path)


browse_input_button = tk.Button(root, text="Browse", command=browse_input_file)
browse_input_button.pack()

# Run genetic algorithm button
run_genetic_algorithm_button = tk.Button(root, text="Run Genetic Algorithm", command=run_genetic_algorithm)
run_genetic_algorithm_button.pack()

# Start the main event loop
root.mainloop()
