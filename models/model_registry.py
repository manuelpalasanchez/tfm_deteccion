"""Registry global de arquitecturas detectoras.

Uso — registrar una arquitectura:
    >>> from models.model_registry import register
    >>> @register("resnet50")
    ... class ResNet50Detector(BaseDetector): ...

Uso — instanciar por nombre desde configuración:
    >>> from models.model_registry import build
    >>> model = build("resnet50", num_classes=1)
"""

import logging
from typing import Type

from models.base_model import BaseDetector

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, Type[BaseDetector]] = {}


def register(name: str):
    """Decorador que registra una clase ``BaseDetector`` bajo ``name``.

    Args:
        name: Clave única que identifica la arquitectura en configs YAML.

    Raises:
        ValueError: Si ``name`` ya está registrado (protege contra colisiones
            silenciosas en imports condicionales).
    """
    def decorator(cls: Type[BaseDetector]) -> Type[BaseDetector]:
        if name in _REGISTRY:
            raise ValueError(
                f"El nombre '{name}' ya está registrado por {_REGISTRY[name].__qualname__}. "
                "Usa un identificador único por arquitectura."
            )
        _REGISTRY[name] = cls
        logger.debug("Arquitectura registrada: '%s' -> %s", name, cls.__qualname__)
        return cls

    return decorator


def build(name: str, **kwargs) -> BaseDetector:
    """Instancia la arquitectura registrada bajo ``name``.

    Args:
        name: Clave de la arquitectura (debe coincidir con el decorador ``@register``).
        **kwargs: Argumentos de construcción del modelo (p. ej. ``pretrained=True``).

    Returns:
        Instancia de ``BaseDetector`` lista para entrenamiento o inferencia.

    Raises:
        ValueError: Si ``name`` no existe en el registry.
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise ValueError(
            f"Arquitectura '{name}' no encontrada en el registry. "
            f"Disponibles: {available}"
        )
    logger.info("Construyendo arquitectura '%s' con kwargs=%s", name, kwargs)
    return _REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Devuelve los nombres de todas las arquitecturas registradas, ordenados."""
    return sorted(_REGISTRY.keys())
