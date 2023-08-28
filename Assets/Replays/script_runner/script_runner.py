import tkinter as tk
from tkinter import filedialog
import os
import random
import csv

# Define the genetic algorithm parameters
num_genes = 5
population_size = 10
num_children = 5
num_generations = 5
decay_rate = 0.8
current_generation = 0
param_names = ["", "rawThrottle", "rawBrake", "brakePressure", "throttle"]

num_generations = 10

# Define the base input file (load and export)
def load_base_input_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content


def extract_parameters_from_content(content):
    lines = content.split("\n")
    params = {}
    for line in lines:
        for param in param_names:
            if line.strip().startswith(param):
                params[param] = float(line.split(":")[1].strip())
                print(f"Extracted: {param} -> {params[param]}")  # Debugging line
    return params


def generate_content_from_parameters(base_content, parameters):
    lines = base_content.split("\n")
    new_content = []
    for line in lines:
        modified_line = line
        for param, value in parameters.items():
            if param == "speed":
                modified_line = f"  - {param}: {value:.6f}"
            else line.strip().startswith(param):
                modified_line = f"    {param}: {value:.6f}"
                break
        new_content.append(modified_line)
    return "\n".join(new_content)


def initial_random_modify(base_params):
    new_params = {}
    for param in param_names:
        change = random.uniform(-decay_rate, decay_rate)
        new_params[param] = base_params[param] * (1 + change)
    return new_params


def export_modified_input_file(modified_content, generation, child_index):
    output_folder = output_folder_entry.get()
    output_file_name = f"modified_{generation}_{child_index}.asset"
    output_file_path = os.path.join(output_folder, output_file_name)
    with open(output_file_path, 'w') as file:
        file.write(modified_content)

# Genetic Algorithm Functions

def initialize_individual():
    return [random.uniform(0, 100) for _ in range(num_genes)]  # Set the range for parameter values (0-100)

def modify_input_content(content, genes):
    # Define placeholders for parameters in the content
    placeholders = ["PARAM_SPEED", "PARAM_THROTTLE", "PARAM_RAWTHROTTLE", "PARAM_RAWBRAKE", "PARAM_BRAKEPRESSURE"]
    
    modified_content = content
    for placeholder, gene in zip(placeholders, genes):
        modified_content = modified_content.replace(placeholder, str(gene))
    return modified_content

def read_laptimes_from_csv(csv_path):
    lap_times = []
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip header
        for row in reader:
            lap_times.append(float(row[0]))
    return lap_times

def run_genetic_algorithm():
    base_input_file = input_file_entry.get()

    for generation in range(num_generations):
        if generation == 0:
            for child_index in range(num_children):
                genes = initialize_individual()
                genes = [round(gene, 2) for gene in genes]  # Round to two decimal places
                input_content = load_base_input_file(base_input_file)
                modified_content = modify_input_content(input_content, genes)
                export_modified_input_file(modified_content, generation, child_index)
        else:
            csv_path = laptime_csv_entry.get()
            lap_times = read_laptimes_from_csv(csv_path)
            fastest_indices = sorted(range(len(lap_times)), key=lambda i: lap_times[i])[:2]
            
            for child_index in range(num_children):
                if child_index < 2:
                    # Use the genes from the fastest children
                    genes = populations[generation - 1][fastest_indices[child_index]]
                else:
                    # Generate new children using genetic algorithm
                    genes = initialize_individual()
                genes = [round(gene, 2) for gene in genes]  # Round to two decimal places
                input_content = load_base_input_file(base_input_file)
                modified_content = modify_input_content(input_content, genes)
                export_modified_input_file(modified_content, generation, child_index)
                
                populations[generation].append(genes)

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

# Laptime CSV selection
laptime_csv_label = tk.Label(root, text="Select Laptime CSV File:")
laptime_csv_label.pack()
laptime_csv_entry = tk.Entry(root)
laptime_csv_entry.pack()

def browse_laptime_csv():
    laptime_csv_path = filedialog.askopenfilename()
    laptime_csv_entry.delete(0, tk.END)
    laptime_csv_entry.insert(0, laptime_csv_path)

browse_laptime_csv_button = tk.Button(root, text="Browse", command=browse_laptime_csv)
browse_laptime_csv_button.pack()

# Output folder entry
output_folder_label = tk.Label(root, text="Output Folder:")
output_folder_label.pack()
output_folder_entry = tk.Entry(root)
output_folder_entry.pack()

# Run genetic algorithm button
run_genetic_algorithm_button = tk.Button(root, text="Run Genetic Algorithm", command=run_genetic_algorithm)
run_genetic_algorithm_button.pack()

# Store populations
populations = [[] for _ in range(num_generations)]

# Start the main event loop
root.mainloop()
