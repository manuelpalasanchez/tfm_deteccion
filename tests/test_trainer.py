"""Smoke tests del trainer y la funcion de perdida."""

import math
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset

from training.losses import get_lossFunction
from training.trainer import Trainer
from models.resnet50 import ResNet50Detector


class _FakeDataset(Dataset):
    """Dataset minimo con imagenes aleatorias y etiquetas alternadas (real/fake)."""

    def __init__(self, n: int = 16):
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        img = torch.randn(3, 224, 224)
        label = idx % 2  # alternancia garantiza las dos clases
        return img, label, "gen"


def _make_loader(n: int = 16, batch_size: int = 4) -> DataLoader:
    return DataLoader(_FakeDataset(n), batch_size=batch_size, shuffle=False)


def _make_cfg(epochs: int = 1) -> SimpleNamespace:
    scheduler = SimpleNamespace(T_max=epochs, eta_min=0.0)
    training = SimpleNamespace(
        epochs=epochs,
        lr=1e-4,
        loss="bce_with_logits",
        scheduler=scheduler,
    )
    model_ns = SimpleNamespace(name="resnet50", kwargs=SimpleNamespace())
    wandb_ns = SimpleNamespace(enabled=False)
    return SimpleNamespace(training=training, model=model_ns, wandb=wandb_ns)


# ------------------------------------------------------------------ #
# Tests de losses                                                     #
# ------------------------------------------------------------------ #

def test_build_loss_devuelve_modulo():
    loss = get_lossFunction("bce_with_logits")
    assert isinstance(loss, torch.nn.Module)


def test_build_loss_desconocido_lanza_error():
    try:
        get_lossFunction("perdida_inventada")
        assert False, "Esperaba ValueError"
    except ValueError:
        pass


# ------------------------------------------------------------------ #
# Tests del trainer                                                   #
# ------------------------------------------------------------------ #

def test_trainer_loss_es_finito(tmp_path):
    model = ResNet50Detector(pretrained=False)
    cfg = _make_cfg(epochs=1)
    trainer = Trainer(
        model=model,
        train_loader=_make_loader(16, 4),
        val_loader=_make_loader(16, 4),
        cfg=cfg,
        output_dir=tmp_path,
    )
    loss = trainer._train_epoch(epoch=1)
    assert math.isfinite(loss), f"Loss no finito: {loss}"


def test_trainer_una_epoch_guarda_checkpoint(tmp_path):
    model = ResNet50Detector(pretrained=False)
    cfg = _make_cfg(epochs=1)
    trainer = Trainer(
        model=model,
        train_loader=_make_loader(16, 4),
        val_loader=_make_loader(16, 4),
        cfg=cfg,
        output_dir=tmp_path,
    )
    trainer.train()
    assert (tmp_path / "checkpoint_best.pth").exists(), "No se guardo el checkpoint"


def test_trainer_checkpoint_cargable(tmp_path):
    model = ResNet50Detector(pretrained=False)
    cfg = _make_cfg(epochs=1)
    trainer = Trainer(
        model=model,
        train_loader=_make_loader(16, 4),
        val_loader=_make_loader(16, 4),
        cfg=cfg,
        output_dir=tmp_path,
    )
    trainer.train()
    ckpt = torch.load(tmp_path / "checkpoint_best.pth", weights_only=True)
    assert "model_state_dict" in ckpt
    assert "best_auc" in ckpt
