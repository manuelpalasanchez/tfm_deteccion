"""Smoke tests del detector ViT-B/16.
Usa una config minima para evitar descargas de HuggingFace.
"""

import torch
import pytest
from unittest.mock import patch
from transformers import ViTConfig

from models.vit import ViTDetector

_X = torch.randn(1, 3, 224, 224)


def _tiny_config() -> ViTConfig:
    return ViTConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        image_size=224,
        patch_size=32,
        num_labels=1,
    )


@patch("models.vit.ViTConfig.from_pretrained")
def test_forward_shape(mock_cfg):
    mock_cfg.return_value = _tiny_config()
    model = ViTDetector(pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(_X)
    assert out.shape == (1, 1), f"Esperaba (1,1), obtuvo {out.shape}"


@patch("models.vit.ViTConfig.from_pretrained")
def test_optimizer(mock_cfg):
    mock_cfg.return_value = _tiny_config()
    model = ViTDetector(pretrained=False)
    opt = model.get_optimizer(lr=1e-4)
    assert hasattr(opt, "step")
