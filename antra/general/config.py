import configparser
from pathlib import Path

from totalsegmentator.map_to_binary import class_map

'''
Handlers for loading / accessing / saving the configuration
'''

# Parsers for certain config values (string in, value out)

def parse_margin_pair(value: str):
    ''''parses the margin config's value. returns a detailed error if it fails'''
    try:
        parts = [p.strip() for p in value.split(',')]
        if len(parts) != 2: raise ValueError("expected two comma-separated values")
        if float(parts[0]) < 0: ValueError(f"Margin size must be positive, got {parts[1]}.")
        if not (0 < float(parts[1]) <= 1): raise ValueError(f"Relative opacity must be between 0 and 1, got {parts[1]}.")
        return float(parts[0]), float(parts[1])
    except Exception as e:
        raise ValueError(f"Invalid config value '{value}', Expected format '<float>,<float>'") from e

def parse_opacity(value: str):
    ''''parses and validates the opacity's config value.'''
    try:
        opacity = float(value)
    except ValueError as e:
        raise ValueError(f"Invalid opacity value '{value}'. Expected a float.") from e

    if 0.0 <= opacity <= 1.0: return opacity
    raise ValueError(f"Opacity must be between 0 and 1, got {value}.")

def parse_floats(value: str):
    ''''parses and validates the score weights config value.'''
    try:
        weights = [float(weight.strip()) for weight in value.split(',')]
    except ValueError as e:
        raise ValueError(f"Invalid weights value '{value}'. Expected comma seperated floats.") from e

    if not any(weight < 0 for weight in weights): return weights
    raise ValueError(f"weights must be larger than 0, got {value}.")


# Main function to use, loads a configparser class

def load_configs(files = ['general.ini', 'opacity.ini', 'margins.ini']) -> configparser.ConfigParser:
    '''Read specified configs into 1 object. Default values stored in general/config'''
    config  = configparser.ConfigParser(converters={'margin': parse_margin_pair, 'opacity': parse_opacity, 'tuple': parse_floats})
    default = [Path(__file__).parent / 'defaults' / file for file in files]
    paths   = [Path(__file__).parent.parent.parent / 'config' / file for file in files]
    config.read(default + paths)
    return config

# functions that return more processed data

def get_one_map(config: configparser.ConfigParser, task: str) -> dict[int, float]:
    '''Returns a single task's opacity map {label [int]: opacity [float]}'''
    # get tissue names
    tissue_names = class_map[task]
    label_opacity_map = {}
    for label, tissue in tissue_names.items():
        # body segmentation should be ignored (0 opacity)
        if 'body' in tissue: label_opacity_map[label] = 0; continue
        
        # add label opacity to map
        shorthand = max([opt for opt in config.options('opacity') if opt in tissue], key=len)
        label_opacity_map[label] = config.getfloat('opacity', shorthand)
    return label_opacity_map

def get_label_opacity_maps() -> dict[str, dict[int, float]]:
    '''Returns the opacities of each task's labels as a dictionary of dictionaries
       {task [str]: {label [int]: opacity [float]}}'''
    config = load_configs()
    return {task: get_one_map(config, task) for task in ['total','body','liver_vessels']}