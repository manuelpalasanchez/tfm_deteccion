"""Smoke tests del detector UniversalFakeDetect (CLIP ViT-L/14).

Usa una config minima para evitar descargas de HuggingFace.
"""

import torch
from unittest.mock import patch
from transformers import CLIPConfig, CLIPVisionConfig, CLIPTextConfig

from models.universalfakedetect import UniversalFakeDetectDetector

_X = torch.randn(1, 3, 224, 224)


def _tiny_config() -> CLIPConfig:
    vision = CLIPVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=224,
        patch_size=32,
    )
    text = CLIPTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=100,
        max_position_embeddings=32,
    )
    return CLIPConfig(
        vision_config=vision.to_dict(),
        text_config=text.to_dict(),
        projection_dim=64,
    )


@patch("models.universalfakedetect.CLIPConfig.from_pretrained")
def test_forward_shape(mock_cfg):
    mock_cfg.return_value = _tiny_config()
    model = UniversalFakeDetectDetector(pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(_X)
    assert out.shape == (1, 1), f"Esperaba (1,1), obtuvo {out.shape}"


@patch("models.universalfakedetect.CLIPConfig.from_pretrained")
def test_backbone_frozen(mock_cfg):
    mock_cfg.return_value = _tiny_config()
    model = UniversalFakeDetectDetector(pretrained=False)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "No hay parametros entrenables"
    assert all(n.startswith("fc") for n in trainable), (
        f"Solo fc debe ser entrenable. Entrenables: {trainable}"
    )


@patch("models.universalfakedetect.CLIPConfig.from_pretrained")
def test_optimizer(mock_cfg):
    mock_cfg.return_value = _tiny_config()
    model = UniversalFakeDetectDetector(pretrained=False)
    opt = model.get_optimizer(lr=1e-3)
    assert hasattr(opt, "step")
