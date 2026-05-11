"""Configuration loader for CLIF 2.1 Validation server."""

import json
from pathlib import Path


def load_config(config_path: str = "config/config.json") -> dict:
    """Load and return config dict from a JSON file.

    Args:
        config_path: Path to the JSON config file.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_output_dir(config: dict) -> str:
    """Return the output directory from config, defaulting to 'output'.

    Args:
        config: Parsed config dictionary.

    Returns:
        Output directory path string.
    """
    return config.get("output_dir", "output")
