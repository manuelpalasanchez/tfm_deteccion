"""Funciones de perdida"""

import torch
import torch.nn as nn


def get_lossFunction(name: str, pos_weight: torch.Tensor | None = None) -> nn.Module:
    """Construye  funcion de perdida por nombre.
    Recibe 
        name: Identificador de la perdida (Solo 'bce_with_logits' soportada por ahora)
        pos_weight: Peso para la clase positiva (sintetico)
    """
    if name == "bce_with_logits":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    raise ValueError(
        f"F. Perdida '{name}' no reconocida. Disponibles: ['bce_with_logits']"
    )
