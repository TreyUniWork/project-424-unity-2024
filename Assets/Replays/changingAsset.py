def adjust_speed_in_asset_file(input_path, output_path, increase_percentage):
    """
    Adjust the speed values in the provided Unity asset file.



    Parameters:
    - input_path: Path to the original asset file.
    - output_path: Path to save the modified asset file.
    - increase_percentage: Percentage to increase the speed by.
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()

    # Change the speed values
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith('- speed:'):
            speed = float(line.split(' ')[2])
            # Increase the speed by the given percentage
            new_speed = speed * (1 + increase_percentage / 100)
            lines[i] = '  - speed: {}\n'.format(new_speed)

    # Write the modified lines back to the new file
    with open(output_path, 'w') as f:
        f.writelines(lines)

# Example usage:
adjust_speed_in_asset_file('2022-07-27 20.00.32 UTC 05.05.920 ideal_autopilot.asset', 'Modified 2022-07-27 20.00.32 UTC 05.05.920 ideal_autopilot.asset', 27.5)