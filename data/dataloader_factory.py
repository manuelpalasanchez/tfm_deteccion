"""Factory de DataLoaders: registra datasets por nombre y los instancia a partir de conguracion"""

import logging
from torch.utils.data import DataLoader

from data.base_dataset import BaseDataset

logger = logging.getLogger(__name__)

_DATASET_REGISTRY: dict[str, type[BaseDataset]] = {}


def register_dataset(name: str):
    """Decorador que registra una clase BaseDataset bajo con nombre name.
    Recibe name: Clave única que identifica el dataset en los configs YAML.
    ValueError: Si name ya está registrado.
    """
    def decorator(cls: type[BaseDataset]) -> type[BaseDataset]:
        if name in _DATASET_REGISTRY:
            raise ValueError(
                f"Dataset '{name}' ya registrado por {_DATASET_REGISTRY[name].__qualname__}."
            )
        _DATASET_REGISTRY[name] = cls
        logger.debug("Dataset registrado: '%s' -> %s", name, cls.__qualname__)
        return cls

    return decorator


def build_dataloader(
    dataset_name: str,
    root: str,
    split: str,
    transform,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Instancia el dataset registrado y devuelve su DataLoader.
        Recibe:
        dataset_name: Clave del dataset .
        root: Ruta raíz del dataset en disco.
        split: Partición a cargar: "train", "val" o "test".
        transform: Pipeline de transformsaciones 
        batch_size: Número de muestras por batch.
        num_workers: Procesos de carga paralela. 
        shuffle: Mezclar el orden de las muestras. True en train, False en eval.
    Devuelve Dataloader
    ValueError: Si dataset_name no está registrado.
    """
    if dataset_name not in _DATASET_REGISTRY:
        available = sorted(_DATASET_REGISTRY.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' no encontrado en el registry. "
            f"Disponibles: {available}"
        )

    dataset: BaseDataset = _DATASET_REGISTRY[dataset_name](
        root=root, split=split, transform=transform
    )
    logger.info(
        "DataLoader creado - dataset='%s' split='%s' samples=%d batch_size=%d",
        dataset_name, split, len(dataset), batch_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def list_datasets() -> list[str]:
    """Devuelve los nombres de todos los datasets registrados, ordenados."""
    return sorted(_DATASET_REGISTRY.keys())
