"""Bucle de entrenamiento unificado para todos los detectores"""

import logging
import time
from pathlib import Path
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.base_model import BaseDetector
from training.losses import get_lossFunction
import wandb

logger = logging.getLogger(__name__)

class Trainer:
    """Bucle unificado: forward, loss, backprop, scheduler, checkpoint por AUC."""

    def __init__(self, model: BaseDetector,train_loader: DataLoader, 
                 val_loader: DataLoader, cfg, output_dir: Path) -> None:
        
        self.model = model #Modelo a entrenar
        self.train_loader = train_loader #DataLoader para entrenamiento
        self.val_loader = val_loader #DataLoader para validación
        self.cfg = cfg #Configuracion
        self.output_dir = output_dir #Directorio de salida
        self.output_dir.mkdir(parents=True, exist_ok=True) 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Entrenar con GPU si está disponible
        self.model.to(self.device) #Mover modelo a dispositivo donde se va a entrenar

        tcfg = cfg.training #Configuración de entrenamiento
        self.epochs = tcfg.epochs
        self.optimizer = model.get_optimizer(lr=tcfg.lr) #Realmente no se usa el que venga en la config, se usa el que define cada modelo
        self.criterion = get_lossFunction(tcfg.loss) 
        self.scheduler = CosineAnnealingLR(self.optimizer,
            T_max=tcfg.scheduler.T_max,
            eta_min=getattr(tcfg.scheduler, "eta_min", 0.0),
        )

        self._best_auc = 0.0
        self._wandb_run = None
        self._setup_wandb()

    def _setup_wandb(self) -> None:
        wcfg = getattr(self.cfg, "wandb", None)
        if wcfg is None or not getattr(wcfg, "enabled", False):
            return
        try:
            self._wandb_run = wandb.init(
                project=wcfg.project,
                entity=getattr(wcfg, "entity", None) or None,
                config={
                    "epochs": self.cfg.training.epochs,
                    "batch_size": self.cfg.training.batch_size,
                    "lr": self.cfg.training.lr,
                },
                name=self.output_dir.name,
            )
        except Exception as exc:
            logger.warning("wandb no disponible: %s", exc)

    def _log(self, metrics: dict, step: int) -> None:
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)

    def _prepare_batch(self, batch):
        images, labels, _ = batch
        images = images.to(self.device)
        labels = labels.float().unsqueeze(1).to(self.device)
        return images, labels
    
    def _forward_loss(self, images, labels):
        images = self.model.preprocess(images)
        logits = self.model(images)
        loss = self.criterion(logits, labels)#Mide la perdida entre las predicciones (logits) y las etiquetas reales
        return logits, loss

    def _train_epoch(self, epoch: int) -> float:
        """
        Recorre el conjunto de entrenamiento por completo, actualizando los pesos del modelo.
        Devuelve la perdida media por batch para este epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        #Del conjunto de entrenamiento:
        # - obtenemos batches de imagenes y etiquetas
        # - los movemos al dispositivo
        # - hacemos el forward 
        # - calculamos la perdida 
        # - hacemos backprop 
        # - actualizamos los pesos
        # Acumulamos la perdida para calcular el promedio al final del epoch.
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch} train", leave=False):
            images, labels = self._prepare_batch(batch)

            _, loss = self._forward_loss(images, labels)

            # Backprop y optimizacion
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() #Actualiza los pesos del modelo usando el optimizador

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches if n_batches else 0.0 #Devuelve la perdida media por batch para este epoch

    @torch.no_grad() #Desactiva gradientes, solo se hace forward para evaluar el modelo, no se actualizan pesos
    def _val_epoch(self, epoch: int) -> tuple[float, float]:
        """
        Evalua el modelo en el conjunto de validacion. Devuelve la perdida media y el AUC.
        """
        self.model.eval()
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(self.val_loader, desc=f"Epoch {epoch} val", leave=False):
            images, labels = self._prepare_batch(batch)

            logits, loss = self._forward_loss(images, labels)

            total_loss += loss.item()
            n_batches += 1
            all_logits.append(logits.squeeze(1).cpu())
            all_labels.append(labels.cpu())

        avg_loss = total_loss / n_batches if n_batches else 0.0
        scores = torch.sigmoid(torch.cat(all_logits)).numpy()
        targets = torch.cat(all_labels).numpy()

        auc = self._safe_auc(targets, scores)
        return avg_loss, auc

    @staticmethod
    def _safe_auc(targets, scores) -> float:
        from sklearn.metrics import roc_auc_score
        # AUC requiere al menos dos clases presentes
        if len(set(targets.flatten().tolist())) < 2:
            return 0.5
        try:
            return float(roc_auc_score(targets, scores))
        except ValueError:
            return 0.5

    def _save_checkpoint(self) -> None:
        ckpt_path = self.output_dir / "checkpoint_best.pth"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_auc": self._best_auc,
            },
            ckpt_path,
        )
        logger.info("Checkpoint guardado en %s (AUC=%.4f)", ckpt_path, self._best_auc)

    def train(self) -> None:
        """Ejecuta el bucle de entrenamiento completo, con validacion y checkpoints"""
        logger.info(
            "Entrenamiento iniciado: modelo=%s device=%s epochs=%d",
            self.cfg.model.name,
            self.device,
            self.epochs,
        )

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(epoch) #Entrenamiento de una epoch completa
            val_loss, val_auc = self._val_epoch(epoch) #Evaluacion en el conjunto de validacion
            self.scheduler.step() #
            elapsed = time.time() - t0  

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_auc=%.4f | %.1fs",epoch,self.epochs, train_loss, val_loss, val_auc, elapsed,)
            self._log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/auc": val_auc,
                    "lr": self.scheduler.get_last_lr()[0],
                },
                step=epoch,
            )

            if val_auc > self._best_auc:
                self._best_auc = val_auc
                self._save_checkpoint()

        if self._wandb_run is not None:
            self._wandb_run.finish()

        logger.info("Entrenamiento finalizado. Mejor AUC val: %.4f", self._best_auc)
