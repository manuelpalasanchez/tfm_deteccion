"""ViT-B/16 detector."""

import torch
from torch.optim import Adam, Optimizer
from transformers import ViTConfig, ViTForImageClassification

from models.base_model import BaseDetector
from models.model_registry import register


@register("vit")
class ViTDetector(BaseDetector):
    """ViT-B/16 preentrenado en ImageNet-21k con la cabeza de clasificacion
    sustituida por clasificador binario de una sola salida.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        model_name = "google/vit-base-patch16-224"
        if pretrained:
            self.vit = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=1,
                ignore_mismatched_sizes=True,  # cabeza (fc) original tiene 1000 clases
            )
        else:
            config = ViTConfig.from_pretrained(model_name)
            config.num_labels = 1
            self.vit = ViTForImageClassification(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(pixel_values=x).logits

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # La normalizacion ImageNet ya se aplica en transforms.py
        return x

    def get_optimizer(self, lr: float) -> Optimizer:
        return Adam(self.parameters(), lr=lr)
