"""Configuration loading: default YAML merged with an optional override file."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"
TUNED_FUSION = ROOT / "config" / "tuned_fusion.yaml"


def _merge(base, override):
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path=None, tuned=True):
    """Defaults, then validation-tuned fusion settings if present, then ``path``."""
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    if tuned and TUNED_FUSION.exists():
        config = _merge(config, yaml.safe_load(TUNED_FUSION.read_text()) or {})
    if path:
        config = _merge(config, yaml.safe_load(Path(path).read_text()) or {})
    return config


def resolve(path):
    """Resolve a config path relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
