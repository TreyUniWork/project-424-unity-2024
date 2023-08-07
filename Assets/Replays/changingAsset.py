def adjust_parameters_for_faster_lap_25_percent_speed(file_path):
    """
    Adjust various parameters in the provided Unity asset file to aim for a faster lap time with a 25% speed increase.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    for line in lines:
        stripped_line = line.strip()

        # Adjust speed by 25%
        if stripped_line.startswith("- speed:"):
            speed_str = stripped_line.split(":")[1].strip()
            speed = float(speed_str)
            new_speed = speed * 1.26  # Increase by 25%
            modified_lines.append(f"  - speed: {new_speed:.2f}\n")
        
        # Adjust rawThrottle with given formula
        elif stripped_line.startswith("rawThrottle:"):
            raw_throttle_str = line.split(":")[1].strip()
            raw_throttle = float(raw_throttle_str)
            new_raw_throttle = min(raw_throttle * 1.15, 10000)
            modified_lines.append(f"    rawThrottle: {new_raw_throttle:.2f}\n")

        # Adjust throttle to be closer to maximum without maxing out
        elif stripped_line.startswith("throttle:"):
            throttle_str = stripped_line.split(":")[1].strip()
            throttle = float(throttle_str)
            new_throttle = min(throttle * 1.02, 100)  # Increase by 10%, but cap at 100
            modified_lines.append(f"    throttle: {new_throttle:.2f}\n")

        # Slightly reduce brakePressure to allow for faster deceleration when needed
        elif stripped_line.startswith("brakePressure:"):
            brake_pressure_str = stripped_line.split(":")[1].strip()
            brake_pressure = float(brake_pressure_str)
            new_brake_pressure = brake_pressure * 0.95  # Decrease by 10%
            modified_lines.append(f"    brakePressure: {new_brake_pressure:.2f}\n")

        else:
            modified_lines.append(line)

    # Write the modified content to a new file
    output_file_path = file_path.replace(".asset", "_speed_26.asset")
    with open(output_file_path, 'w') as file:
        file.writelines(modified_lines)

    return output_file_path

# Adjusting parameters for a potentially faster lap time with 25% speed increase
faster_lap_file_path_25_percent_speed = adjust_parameters_for_faster_lap_25_percent_speed('2022-07-27 20.00.32 UTC 05.05.920 ideal_autopilot.asset')
faster_lap_file_path_25_percent_speed
