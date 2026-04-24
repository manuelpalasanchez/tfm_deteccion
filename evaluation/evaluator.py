"""Evaluador: corre inferencia y guarda metricas y graficas por ronda."""

import logging
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.dataloader_factory import build_dataset
from data.transforms import get_eval_transforms
from evaluation.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    save_metrics,
)
from models.base_model import BaseDetector

logger = logging.getLogger(__name__)


class Evaluator:
    """Orquesta las rondas de evaluacion E1, E2 y opcionalmente E3."""

    def __init__(
        self,
        model: BaseDetector,
        cfg: SimpleNamespace,
        output_dir: Path,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _run_inference(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (targets, scores) para todo el loader."""
        all_scores: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for batch in tqdm(loader, desc="Inferencia", leave=False):
            images, labels, _ = batch
            images = self.model.preprocess(images.to(self.device))
            logits = self.model(images).squeeze(1)
            scores = torch.sigmoid(logits).cpu()
            all_scores.append(scores)
            all_targets.append(labels.float())

        targets = torch.cat(all_targets).numpy()
        scores = torch.cat(all_scores).numpy()
        return targets, scores

    def _build_loader(self, dcfg: SimpleNamespace) -> DataLoader:
        dataset = build_dataset(
            dataset_name=dcfg.dataset,
            root=dcfg.root,
            split=dcfg.split,
            transform=get_eval_transforms(),
        )
        max_samples = getattr(dcfg, "max_samples", None)
        if max_samples and len(dataset) > max_samples:
            # Muestreo aleatorio con semilla fija: evita sesgo por orden de disco
            # (el scan agrupa por categoria/label y coger range(N) da bloques homogeneos)
            rng = random.Random(42)
            indices = rng.sample(range(len(dataset)), max_samples)
            dataset = Subset(dataset, indices)

        num_workers = getattr(self.cfg.data, "num_workers", 0)
        pin_memory = getattr(self.cfg.data, "pin_memory", True)
        batch_size = self.cfg.training.batch_size

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def _eval_round(self, round_name: str, dcfg: SimpleNamespace) -> dict:
        """Corre una ronda de evaluacion y guarda metricas y graficas."""
        logger.info("Evaluando ronda %s (dataset=%s split=%s)...", round_name, dcfg.dataset, dcfg.split)
        loader = self._build_loader(dcfg)
        targets, scores = self._run_inference(loader)

        metrics = compute_metrics(targets, scores)
        logger.info(
            "%s -> AUC=%.4f AP=%.4f Acc=%.4f",
            round_name,
            metrics["auc_roc"],
            metrics["average_precision"],
            metrics["accuracy"],
        )

        plot_roc_curve(
            targets,
            scores,
            self.output_dir / f"roc_curve_{round_name}.png",
            title=f"ROC Curve — {round_name.upper()}",
        )
        plot_confusion_matrix(
            targets,
            scores,
            self.output_dir / f"confusion_matrix_{round_name}.png",
            title=f"Confusion Matrix — {round_name.upper()}",
        )

        return metrics

    def evaluate(self) -> dict:
        """Ejecuta todas las rondas habilitadas y guarda metrics.json."""
        all_metrics: dict[str, dict] = {}
        dcfg = self.cfg.data

        all_metrics["e1"] = self._eval_round("e1", dcfg.eval_e1)

        if getattr(dcfg.eval_e1b, "enabled", False):
            all_metrics["e1b"] = self._eval_round("e1b", dcfg.eval_e1b)
        else:
            logger.info("E1b deshabilitada en config (eval_e1b.enabled=false).")

        if getattr(dcfg.eval_e2, "enabled", False):
            all_metrics["e2"] = self._eval_round("e2", dcfg.eval_e2)
        else:
            logger.info("E2 deshabilitada en config (eval_e2.enabled=false).")

        if getattr(dcfg.eval_e3, "enabled", False):
            all_metrics["e3"] = self._eval_round("e3", dcfg.eval_e3)
        else:
            logger.info("E3 deshabilitada en config (eval_e3.enabled=false).")

        save_metrics(all_metrics, self.output_dir / "metrics.json")
        logger.info("Metricas guardadas en %s", self.output_dir / "metrics.json")
        return all_metrics
