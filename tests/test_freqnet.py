"""Smoke tests del detector FreqNet."""

import torch
from models.freqnet import FreqNetDetector

_X = torch.randn(1, 3, 224, 224)


def test_forward_shape():
    model = FreqNetDetector()
    model.eval()
    with torch.no_grad():
        out = model(_X)
    assert out.shape == (1, 1), f"Esperaba (1,1), obtuvo {out.shape}"


def test_optimizer():
    model = FreqNetDetector()
    opt = model.get_optimizer(lr=1e-4)
    assert hasattr(opt, "step")


def test_preprocess_identity():
    model = FreqNetDetector()
    assert torch.equal(model.preprocess(_X), _X)
