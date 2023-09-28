# Laptime Optimization Algorithm From AUT Students
## Before You Begin
Before you proceed with setting up and using the laptime optimization algorithm, please ensure that you have the project itself properly installed. Detailed instructions on how to set up the project can be found starting from the [Project 424](#project-424) section.

Once you have successfully installed the project, you can follow the instructions in the [Algorithm Requirements](#Algorithm-Requirements) section below to install the necessary Python dependencies for the laptime optimization algorithm.

## Algorithm Requirements

Before using the laptime optimization algorithm, you need to install the required Python dependencies. Follow these steps:

1. **Navigate to the Algorithm Directory**:
   - Open a terminal/command prompt.
   - Use the terminal/command prompt to navigate to the following directory:
     ```
     Assets/Resources/script_runner/
     ```

2. **Install Python Dependencies**:
   - Run the following command to install the required Python packages from the `requirements.txt` file:
     ```
     pip install -r requirements.txt
     ```

By following these steps, you'll have the necessary Python dependencies installed and ready to use for the laptime optimization algorithm.

## Running the Algorithm
We have implemented an algorithm to optimize lap times for the PERRINN 424 hypercar simulation. Follow these steps to use the algorithm:

1. **Generate the First Generation of Autopilot Asset Files**:
   - Run the algorithm script located at:
     ```
     Assets\Resources\script_runner\script_runner.py
     ```
   - This script takes a base asset file and creates five child asset files in GEN1. The first child is identical to the base to keep track of the best child from the previous generation, while the other four have mutations with different input values (throttle, brake, etc.).

2. **Clear Player Preferences**:
   - In Unity, go to `Edit` and select `Clear PlayerPrefs`. This ensures a clean slate for the simulation.

3. **Run the Unity Simulation**:
   - Start the Unity simulation in play mode.

4. **Monitor Autopilot Asset File Completion**:
   - Once all five autopilot asset files in GEN1 complete their laps, a CSV file containing lap times for each asset file will be generated. The Python script will read this CSV file and generate the next generation of autopilot asset files (e.g., GEN1 -> GEN2) based on the best-performing asset file from the previous generation.

5. **Automatic Simulation Execution**:
   - The Unity simulation will automatically detect the next generation of files and run the simulation with them.

6. **Repeat for Multiple Generations**:
   - Repeat the process for as many generations as desired to further optimize lap times.

7. **CSV Grapher Tool**:
   - To visualize the performance of different generations, you can use the CSV grapher tool located at:
     ```
     Assets\Resources\script_runner\csv grapher.py
     ```
   - Ensure that you move all the output CSV files (e.g., gen1.csv, gen2.csv, etc.) all into one folder for the tool the tool to read.

By following these steps, you can use the laptime optimization algorithm to improve the performance of the PERRINN 424 hypercar in the simulation.


# Project 424
Simulation of the PERRINN 424 electric hypercar in Unity using Vehicle Physics Pro

[Hot Lap video in Monza](https://www.youtube.com/watch?v=OMoQGtA3gCs)

## Requirements

- Unity 2019.4 LTS (using 2019.4.18f1 at the time of writing this)

## How to set up and open the project in Unity

1. Clone the repository to your computer.
2. Add the repository folder to Unity Hub: Projects > Open > Add project from disk, select the folder with the repository. 
3. Click the newly added project in the list

NOTE: Don't copy the repository folder to an existing Unity project. The simulation won't likely work.

## How to run the PERRINN 424 hypercar in autopilot

1. Open the scene "Scenes/424 Nordschleife Scene".
2. Play the scene. The car is at the starting point.
3. Press **Q** to enable the autopilot.

All other features work normally: telemetry, cameras, time scale...

## How to drive the 424

1. Open one of the scenes in the Scenes folder and Play it.
2. Press **I** to open the input settings. The first time it shows the default keyboard mappings.
3. Click the inputs and follow the instructions to map the inputs to your device. Currently keyboard and DirectInput devices (with or without force feedback) are supported. Your settings will be saved and remembered.
4. Press the **Gear Up** input to engage the **D** (drive) mode.
5. Drive!

## Development guidelines

Writing code and components for the Project 424 should follow these rules:

#### Code

Code should follow the conventions of the Unity API:

- Namespace, class, methods, properties, etc.
- Naming and case as in the Unity API.

#### Components

Components must support the same operations supported by built-in Unity components without errors, including:

- Enable / disable in runtime.
- Instance / destroy in runtime.
- Instance / destroy prefabs using the component.
- Modify the public properties in the inspector in runtime.
- Modify the public properties from other scripts, both in editor and runtime.
- Hot script reload.
