"""Tests del model_registry: registro, lookup y construccion."""

import pytest

import models.resnet50
import models.freqnet
import models.vit
import models.universalfakedetect

from models.model_registry import build, list_models
from models.resnet50 import ResNet50Detector
from models.freqnet import FreqNetDetector


def test_all_models_registered():
    registered = list_models()
    for name in ("freqnet", "resnet50", "universalfakedetect", "vit"):
        assert name in registered, f"'{name}' no esta en el registry"


def test_list_models_is_sorted():
    names = list_models()
    assert names == sorted(names)


def test_registry_build_resnet50():
    model = build("resnet50", pretrained=False)
    assert isinstance(model, ResNet50Detector)


def test_registry_build_freqnet():
    model = build("freqnet")
    assert isinstance(model, FreqNetDetector)


def test_registry_unknown_name_raises():
    with pytest.raises(ValueError, match="no encontrada"):
        build("modelo_que_no_existe")
