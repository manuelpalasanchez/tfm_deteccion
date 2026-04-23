"""UniversalFakeDetect detector (Ojha et al., CVPR 2023)."""

import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from transformers import CLIPConfig, CLIPModel

from models.base_model import BaseDetector
from models.model_registry import register





@register("universalfakedetect")
class UniversalFakeDetectDetector(BaseDetector):
    """Extractor de features CLIP ViT-L/14 congelado mas clasificador lineal.
    Solo se entrena la cabeza lineal; el backbone CLIP queda fijo.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        model_name = "openai/clip-vit-large-patch14"
        if pretrained:
            self.model = CLIPModel.from_pretrained(model_name)
        else:
            config = CLIPConfig.from_pretrained(model_name)
            self.model = CLIPModel(config)
        for param in self.model.parameters():
            param.requires_grad = False  # backbone congelado segun el paper
        self.fc = nn.Linear(self.model.config.projection_dim, 1)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get_image_features retorna BaseModelOutputWithPooling en transformers 5.x;
        # pooler_output contiene los features proyectados [B, projection_dim]
        features = self.model.get_image_features(pixel_values=x).pooler_output
        return self.fc(features)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # La normalizacion ImageNet ya se aplica en transforms.py
        # Las stats de CLIP son muy proximas a ImageNet
        return x

    def get_optimizer(self, lr: float) -> Optimizer:
        return Adam(self.fc.parameters(), lr=lr)  # solo fc tiene requires_grad=True
