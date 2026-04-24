"""Carga de configuraciones YAML con herencia via _base_.

Compartido entre scripts/train.py y scripts/evaluate.py para evitar duplicar
el loader en ambos entry points.
"""

from pathlib import Path
from types import SimpleNamespace

import yaml


def load_raw(config_path: Path) -> dict:
    """Carga un YAML resolviendo herencia via _base_ de forma recursiva.

    El campo _base_ se interpreta como una ruta relativa al propio fichero
    y el contenido hijo se fusiona encima del padre en profundidad.
    """
    with config_path.open() as f:
        raw: dict = yaml.safe_load(f)
    base_key = raw.pop("_base_", None)
    if base_key:
        base = load_raw(config_path.parent / base_key)
        deep_merge(base, raw)
        return base
    return raw


def deep_merge(base: dict, override: dict) -> None:
    """Fusiona override sobre base en profundidad (in-place)."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            deep_merge(base[key], val)
        else:
            base[key] = val


def dict_to_namespace(d) -> SimpleNamespace:
    if not isinstance(d, dict):
        return d
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, dict_to_namespace(v) if isinstance(v, dict) else v)
    return ns


def load_config(config_path: str | Path) -> SimpleNamespace:
    """Atajo: load_raw + dict_to_namespace."""
    return dict_to_namespace(load_raw(Path(config_path)))
