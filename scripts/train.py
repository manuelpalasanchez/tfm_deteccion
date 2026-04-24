"""Script de entrenamiento de una arquitectura.
Uso: python scripts/train.py --config configs/resnet50.yaml
"""

import argparse
import importlib
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import torch
import yaml
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, str(Path(__file__).parent.parent))
import data.cnndetection_dataset  
from data.dataloader_factory import build_dataset
from data.transforms import get_eval_transforms, get_train_transforms
from models import model_registry
from training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Mapa de nombre de modelo a modulo que contiene su @register
_MODEL_MODULES = {
    "resnet50": "models.resnet50",
    "freqnet": "models.freqnet",
    "vit": "models.vit",
    "universalfakedetect": "models.universalfakedetect",
}


def _load_raw(config_path: Path) -> dict:
    """Carga un YAML resolviendo herencia via _base_ de forma recursiva."""
    with config_path.open() as f:
        raw: dict = yaml.safe_load(f)
    base_key = raw.pop("_base_", None)
    if base_key:
        base = _load_raw(config_path.parent / base_key)
        _deep_merge(base, raw)
        return base
    return raw


def _load_config(config_path: Path) -> SimpleNamespace:
    return _dict_to_namespace(_load_raw(config_path))


def _deep_merge(base: dict, override: dict) -> None:
    """Fusiona override sobre base en profundidad (in-place)."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


def _dict_to_namespace(d) -> SimpleNamespace:
    if not isinstance(d, dict):
        return d
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_namespace(v) if isinstance(v, dict) else v)
    return ns


def _make_loader(
    dcfg: SimpleNamespace,
    transform,
    shuffle: bool,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = build_dataset(
        dataset_name=dcfg.dataset,
        root=dcfg.root,
        split=dcfg.split,
        transform=transform,
    )

    max_samples = getattr(dcfg, "max_samples", None)
    if max_samples is not None and len(dataset) > max_samples:
        dataset = Subset(dataset, list(range(max_samples)))
        logger.info("Subset aplicado: %d muestras", max_samples)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = _load_config(args.config)

    model_name = cfg.model.name
    if model_name not in _MODEL_MODULES:
        raise ValueError(
            f"Modelo '{model_name}' no tiene modulo registrado. "
            f"Disponibles: {list(_MODEL_MODULES)}"
        )
    importlib.import_module(_MODEL_MODULES[model_name])

    model = model_registry.build(model_name, **vars(cfg.model.kwargs))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.output.base_dir) / f"{model_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, output_dir / "config.yaml")

    batch_size = cfg.training.batch_size
    num_workers = getattr(cfg.data, "num_workers", 0)
    pin_memory = getattr(cfg.data, "pin_memory", True)

    train_loader = _make_loader(
        cfg.data.train,
        get_train_transforms(),
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = _make_loader(
        cfg.data.val,
        get_eval_transforms(),
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        output_dir=output_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
