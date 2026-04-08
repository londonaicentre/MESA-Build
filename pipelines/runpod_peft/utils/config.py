"""
Configuration loading and validation
"""

import yaml
from typing import Any


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """
    Load training configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
