"""Registry global de detectores.
Registra una arquitectura y permite instanciarla por nombre
"""

import logging
from typing import Type

from models.base_model import BaseDetector

logger = logging.getLogger(__name__)

_MODEL_REGISTRY: dict[str, Type[BaseDetector]] = {}


def register(name: str):
    """Decorador que registra una clase BaseDetector bajo name.
    Recibe name: Clave única que identifica la arquitectura en configs YAML.
    Raises: ValueError: Si name ya está registrado 
    """
    def decorator(cls: Type[BaseDetector]) -> Type[BaseDetector]:
        if name in _MODEL_REGISTRY:
            raise ValueError(
                f"El nombre '{name}' ya está registrado por {_MODEL_REGISTRY[name].__qualname__}. "
                "Usa un identificador único por arquitectura."
            )
        _MODEL_REGISTRY[name] = cls
        logger.debug("Arquitectura registrada: '%s' -> %s", name, cls.__qualname__)
        return cls

    return decorator


def build(name: str, **kwargs) -> BaseDetector:
    """Instancia la arquitectura registrada bajo name.
    Recibe name: Clave de la arquitectura (debe coincidir con el decorador @register).
        **kwargs: Argumentos de construcción del modelo (p. ej. pretrained=True).
    Devuelve la instancia del modelo
    Raises ValueError: Si name no existe en el registry.
    """
    if name not in _MODEL_REGISTRY:
        available = sorted(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Arquitectura '{name}' no encontrada en el registry. "
            f"Disponibles: {available}"
        )
    logger.info("Construyendo arquitectura '%s' con kwargs=%s", name, kwargs)
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Devuelve los nombres de todas las arquitecturas registradas, ordenados."""
    return sorted(_MODEL_REGISTRY.keys())
