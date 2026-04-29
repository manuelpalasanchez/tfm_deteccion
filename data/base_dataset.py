from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    """Interfaz comun para los datasets de deteccion (CNNDetection, GenImage).

    Las subclases deben implementar __len__, __getitem__ y generators.
    La propiedad generators permite al evaluador segmentar metricas por modelo
    generativo sin acoplar el evaluador a cada dataset concreto.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Para len(dataset) - Número total de muestras en el split."""
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple:
        
        """Para dataset[idx]
        Devuelve (imagen_tensor, etiqueta_int, generador_str).
        La etiqueta sigue la convención 0 = real, 1 = sintético.
        El campo generador_str identifica el modelo generativo de origen
        (o "real" para muestras auténticas), lo que permite al evaluador
        calcular métricas por subgrupo.
        """
        ...

    @property
    @abstractmethod
    def generators(self) -> list[str]:
        """Lista de identificadores de modelos generativos presentes en el split."""
        ...
