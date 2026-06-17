"""Interfaz abstracta común para todos los detectores de imágenes sintéticas."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.optim import Optimizer


class BaseDetector(ABC, nn.Module):
    """Contrato que deben cumplir todos los detectores del framework.
    El trainer y el evaluator interactúan exclusivamente con esta interfaz

    Convención de salida: forward devuelve logits sin sigmoid
    La funcion de perdida se aplica externamente.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve logits sin sigmoid con forma [B, 1]."""
        ...

    @abstractmethod
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocesado especifico del modelo, aplicado tras transforms.py."""
        ...

    @abstractmethod
    def get_optimizer(self, lr: float) -> Optimizer:
        """Construye el optimizador para esta arquitectura.
        El trainer delega aqui para respetar elecciones especificas de cada modelo.
        """
        ...
