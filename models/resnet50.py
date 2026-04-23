"""ResNet-50"""
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torchvision.models import ResNet50_Weights, resnet50

from models.base_model import BaseDetector
from models.model_registry import register


@register("resnet50")
class ResNet50Detector(BaseDetector):
    """ResNet-50 preentrenado en ImageNet con la capa fc sustituida por un
    clasificador binario de una sola salida (logit real vs. sintetico).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1) #sustitucion de la ultima capa por una de salida binaria
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # La normalizacion ImageNet ya se aplica en transforms.py
        return x

    def get_optimizer(self, lr: float) -> Optimizer:
        return Adam(self.parameters(), lr=lr)
