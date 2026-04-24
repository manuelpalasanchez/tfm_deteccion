"""Script de evaluacion de una arquitectura entrenada.
Uso: python scripts/evaluate.py --config configs/resnet50.yaml --checkpoint experiments/runs/.../checkpoint_best.pth
"""

import argparse
import importlib
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import data.cnndetection_dataset
import data.genimage_dataset
from evaluation.evaluator import Evaluator
from models import model_registry
from utils.config import load_config

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

    cfg = load_config(args.config)
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
