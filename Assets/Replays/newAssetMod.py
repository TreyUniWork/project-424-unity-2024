import yaml

def modify_speed("2022-04-05 18.35.24 UTC 05.10.230 ideal.asset", 10):
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        content = yaml.safe_load(f)

    # Adjust the frequency by the specified percentage
    content['MonoBehaviour']['metadata']['frequency'] *= (1 + (percentage_increase / 100))

    with open(filename, 'w', encoding='utf-8') as f:
        yaml.safe_dump(content, f)

# Example usage:
# modify_speed("path_to_file.asset", 10)  # Increases the speed by 10%
