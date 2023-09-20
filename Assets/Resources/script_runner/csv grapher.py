import os
import csv
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import re
from tkinter import ttk


# Function to handle folder selection
def select_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        process_folder(folder_path)


# Custom sorting function
def custom_sort(file_name):
    # Extract the numeric part of the file name
    match = re.match(r"gen(\d+)", file_name)
    if match:
        return int(match.group(1))
    else:
        return float("inf")  # Return a very large number for non-matching names


# Function to process the selected folder
def process_folder(folder_path):
    # Initialize lists to store the lowest laptime and generation for each CSV file
    filenames = []
    lowest_laptimes = []
    generations = []
    change_rate = 0
    base_lap_time = 0
    lowest_lap_name = ""
    lowest_lap_time = float("inf")
    last_lap_name = ""
    last_lap_time = float("inf")

    # Loop through all CSV files in the selected folder
    for root_dir, _, files in os.walk(folder_path):
        # Sort the files by custom sorting function
        files.sort(key=custom_sort)

        for csv_file in files:
            if csv_file.endswith(".csv"):
                csv_file_path = os.path.join(root_dir, csv_file)

                # Read the lap times from the CSV file using the existing function
                lap_times = read_csv_laptimes(csv_file_path)

                # Find the minimum laptime and extract the generation number from the file name
                min_laptime = min(lap_times.values())
                min_laptime_filename = os.path.splitext(
                    os.path.basename(min(lap_times, key=lap_times.get))
                )[0]
                match = re.match(r"gen(\d+)", csv_file)
                generation = match.group(1) if match else ""

                # Include base file in the dataset
                if not lowest_laptimes and lap_times:
                    filenames.append("base")
                    base_lap_time = list(lap_times.values())[0]
                    lowest_laptimes.append(base_lap_time)
                    generations.append("0")

                # Append data for this generation to the respective lists
                filenames.append(min_laptime_filename)
                lowest_laptimes.append(min_laptime)
                generations.append(generation)

                # Update change rate and lowest lap time
                if lowest_laptimes and generations:
                    # Iterate through both filenames and lowest_laptimes simultaneously
                    for filename, lap_time in zip(filenames, lowest_laptimes):
                        if lap_time < lowest_lap_time:
                            lowest_lap_time = lap_time
                            lowest_lap_name = filename

                    last_lap_name = min_laptime_filename
                    last_lap_time = min_laptime

                    change_rate = (last_lap_time - base_lap_time) / len(generations)

    # Create a button to show the graph
    show_graph_button = tk.Button(
        root,
        text="Show Graph",
        command=lambda: show_graph(
            filenames,
            lowest_laptimes,
            change_rate,
            base_lap_time,
            lowest_lap_name,
            lowest_lap_time,
            last_lap_name,
            last_lap_time,
        ),
    )
    show_graph_button.pack()

    # Create a button to open the table in a separate popup window
    table_button = tk.Button(
        root,
        text="Open Table",
        command=lambda: show_table(generations, filenames, lowest_laptimes),
    )
    table_button.pack()


# Function to display the graph in a separate window
def show_graph(
    filenames,
    lowest_laptimes,
    change_rate,
    base_lap_time,
    lowest_lap_name,
    lowest_lap_time,
    last_lap_name,
    last_lap_time,
):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [4, 1]}
    )

    # Subplot 1: Line graph
    ax1.plot(filenames, lowest_laptimes, marker="o", linestyle="-")
    ax1.set_xlabel("CSV File")
    ax1.set_ylabel("Lowest LapTime")
    ax1.set_title("Lowest LapTime for Each CSV File")
    ax1.grid(True)
    ax1.set_xticklabels(filenames, rotation=45, ha="right")  # Set x-axis labels

    # Subplot 2: Table
    table_data = [
        ["Change Rate", change_rate],
        ["Base Lap Time", base_lap_time],
        ["Lowest Lap Name", lowest_lap_name],
        ["Lowest Lap Time", lowest_lap_time],
        ["Last Lap Name", last_lap_name],
        ["Last Lap Time", last_lap_time],
    ]

    table = ax2.table(
        cellText=table_data,
        cellLoc="center",
        colLabels=["Metric", "Value"],
        loc="center",
    )

    # Customize table properties
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)

    # Remove axis labels and ticks for the table subplot
    ax2.axis("off")

    # Adjust layout and spacing
    plt.tight_layout()

    # Show the figure
    plt.show()


# Function to read lap times from a CSV file
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


# Function to display the table in a separate popup window with a scroll bar
def show_table(generations, filenames, lowest_laptimes):
    table_window = tk.Toplevel(root)
    table_window.title("Table")

    # Create a frame to contain the table and scrollbars
    frame = ttk.Frame(table_window)
    frame.pack(fill=tk.BOTH, expand=True)

    # Create a vertical scrollbar for the table
    y_scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL)
    y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Create a table to display the Generation, FileName, and LapTime in the popup window
    table_data = list(zip(generations, filenames, lowest_laptimes))
    table_columns = ["Generation", "FileName", "LapTime"]
    table = ttk.Treeview(
        frame,
        columns=table_columns,
        show="headings",
        yscrollcommand=y_scrollbar.set,
    )

    for col in table_columns:
        table.heading(col, text=col)
        table.column(col, width=100)

    for data in table_data:
        table.insert("", "end", values=data)

    table.pack(fill=tk.BOTH, expand=True)
    y_scrollbar.config(command=table.yview)


# Create the GUI window
root = tk.Tk()
root.title("Generation CSV Files Grapher")

# Create a frame to hold the content
frame = tk.Frame(root)
frame.pack(padx=20, pady=20)

# Create a label for the folder selection
label = tk.Label(frame, text="Select the folder containing CSV files:")
label.pack(pady=10)

# Create a button to select the folder
select_button = tk.Button(frame, text="Select Folder", command=select_folder)
select_button.pack(pady=5)

# Run the GUI
root.mainloop()
