from yaml import load, SafeLoader


def open_yaml_file(path: str) -> dict:
    
    with open(f'{path}.yaml', 'r') as f:
        data = load(f, Loader=SafeLoader)

    return data
