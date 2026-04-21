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
        """Propagación hacia adelante.
        Recibe x: Tensor de imágenes con forma [B, 3, H, W], ya normalizado.
        B:Batch size (imagenes juntas en lote)
        3:Canales RGB
        H:Altura de la imagen
        W:Anchura de la imagen
        Devuelve Logits de clasificación binaria con forma [B, 1] (pertenece o no pertenece)
        """
        ...

    @abstractmethod
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocesado específico del modelo, aplicado tras transforms.py.
        Recibe x: Tensor ya transformado por el pipeline estándar.
        Devuelve x: Tensor listo para ser procesado por forward.
        """
        ...

    @abstractmethod
    def get_optimizer(self, lr: float) -> Optimizer:
        """Construye y devuelve el optimizador recomendado para esta arquitectura.
        El trainer llama a este método para respetar elecciones específicas de
        cada modelo 
        Recibe lr (learning rate): Tasa de aprendizaje base
        Devuelve una Instancia de torch.optim.Optimizer lista para usar.
        """
        ...
