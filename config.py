from __future__ import annotations

from pathlib import Path

import yaml


class ConfigNode(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    __setattr__ = dict.__setitem__


def _to_node(value):
    if isinstance(value, dict):
        return ConfigNode({key: _to_node(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_node(item) for item in value]
    return value


def _merge(target, source):
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _merge(target[key], value)
        else:
            target[key] = value


def load_config(config_dir):
    config_dir = Path(config_dir).resolve()
    if not config_dir.is_dir():
        raise IOError("Config directory not found: {}".format(config_dir))

    merged = {}
    for path in sorted(config_dir.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError("Config file must contain a mapping: {}".format(path))
        _merge(merged, data)
    return _to_node(merged)

