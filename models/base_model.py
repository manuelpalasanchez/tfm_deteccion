"""Interfaz abstracta común para todos los detectores de imágenes sintéticas."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.optim import Optimizer


class BaseDetector(ABC, nn.Module):
    """Contrato que deben cumplir todos los detectores del framework.

    El trainer y el evaluator interactúan exclusivamente con esta interfaz,
    lo que permite añadir nuevas arquitecturas sin modificar el código cliente.

    Convención de salida:
        ``forward`` devuelve logits sin sigmoid (forma ``[B, 1]`` o ``[B, 2]``).
        La loss ``BCEWithLogitsLoss`` o ``CrossEntropyLoss`` se aplica externamente.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagación hacia adelante.

        Args:
            x: Tensor de imágenes con forma ``[B, 3, H, W]``, ya normalizado.

        Returns:
            Logits de clasificación binaria con forma ``[B, 1]``.
        """
        ...

    @abstractmethod
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocesado específico del modelo, aplicado tras ``transforms.py``.

        La mayoría de detectores no necesitan preprocesado adicional y pueden
        devolver ``x`` sin modificar. Usar este método para ajustes propios
        del modelo (p. ej. normalización CLIP en UniversalFakeDetect).

        Args:
            x: Tensor ya transformado por el pipeline estándar.

        Returns:
            Tensor listo para ser procesado por ``forward``.
        """
        ...

    @abstractmethod
    def get_optimizer(self, lr: float) -> Optimizer:
        """Construye y devuelve el optimizador recomendado para esta arquitectura.

        El trainer llama a este método para respetar elecciones específicas de
        cada modelo (p. ej. AdamW para ViT, SGD para ResNet-50 fine-tuning).

        Args:
            lr: Tasa de aprendizaje base. Puede ignorarse si el modelo usa
                grupos de parámetros con lr distintas internamente.

        Returns:
            Instancia de ``torch.optim.Optimizer`` lista para usar.
        """
        ...
