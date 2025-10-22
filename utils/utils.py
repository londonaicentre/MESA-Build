import json

def load_config(configlocation="config.json"):
    try:
        with open(configlocation, "r") as file:
            config = json.load(file)
            print("Successfully loaded config file.")
            return config
    except FileNotFoundError:
        print("Failed to load {configlocation}")
        raise
    except json.JSONDecodeError as json_error:
        print(f"Failed to parse {configlocation} as it has {json_error}")
        raise
    except Exception as e:
        print(f"Failed to load {configlocation} due to {e}")
        raise
