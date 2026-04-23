"""Smoke tests del detector ResNet-50."""

import torch
from models.resnet50 import ResNet50Detector

_X = torch.randn(1, 3, 224, 224)


def test_forward_shape():
    model = ResNet50Detector(pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(_X)
    assert out.shape == (1, 1), f"Esperaba (1,1), obtuvo {out.shape}"


def test_optimizer():
    model = ResNet50Detector(pretrained=False)
    opt = model.get_optimizer(lr=1e-4)
    assert hasattr(opt, "step")


def test_preprocess_identity():
    model = ResNet50Detector(pretrained=False)
    assert torch.equal(model.preprocess(_X), _X)
