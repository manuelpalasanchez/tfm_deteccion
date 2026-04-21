"""Contrato abstracto compartido por todos los datasets del experimento."""

from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    """Interfaz común para CNNDetection, FaceForensics++ y GenImage.

    Las subclases deben implementar ``__len__``, ``__getitem__`` y ``generators``.
    La propiedad ``generators`` permite al evaluador segmentar métricas por modelo
    generativo sin acoplar el evaluador a cada dataset concreto.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Número total de muestras en el split."""
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple:
        """Devuelve ``(imagen_tensor, etiqueta_int, generador_str)``.

        La etiqueta sigue la convención 0 = real, 1 = sintético.
        El campo ``generador_str`` identifica el modelo generativo de origen
        (o ``"real"`` para muestras auténticas), lo que permite al evaluador
        calcular métricas por subgrupo.
        """
        ...

    @property
    @abstractmethod
    def generators(self) -> list[str]:
        """Lista de identificadores de modelos generativos presentes en el split."""
        ...
