"""Script de evaluacion de una arquitectura entrenada.
Uso: python scripts/evaluate.py --config configs/resnet50.yaml --checkpoint experiments/runs/.../checkpoint_best.pth
"""

import argparse
import importlib
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import data.cnndetection_dataset
import data.genimage_dataset
from evaluation.evaluator import Evaluator
from models import model_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directorio donde guardar los resultados. Por defecto: misma carpeta que el checkpoint.",
    )
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

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(
        "Checkpoint cargado: %s (best_auc=%.4f)",
        args.checkpoint,
        ckpt.get("best_auc", float("nan")),
    )

    output_dir = args.output_dir or args.checkpoint.parent

    evaluator = Evaluator(model=model, cfg=cfg, output_dir=output_dir)
    metrics = evaluator.evaluate()

    print("\n--- Resultados ---")
    for round_name, m in metrics.items():
        print(
            f"  {round_name.upper()}: AUC={m['auc_roc']:.4f}  "
            f"AP={m['average_precision']:.4f}  Acc={m['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
