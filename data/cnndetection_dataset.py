"""Dataset CNNDetection para detección de imágenes sintéticas generadas por GANs.
"""
import logging
from pathlib import Path

from PIL import Image
from torch import Tensor

from data.base_dataset import BaseDataset
from data.dataloader_factory import register_dataset

logger = logging.getLogger(__name__)

_REAL_DIR = "0_real"
_FAKE_DIR = "1_fake"
_LABEL_REAL = 0
_LABEL_FAKE = 1
_VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@register_dataset("cnndetection")
class CNNDetectionDataset(BaseDataset):
    """Estructura
        {root}/{generador}_{split}/{categoria}/0_real/*.png
        {root}/{generador}_{split}/{categoria}/1_fake/*.png
    """

    def __init__(self, root: str, split: str, transform=None) -> None:
        """
        Args:
            root: Ruta raíz que contiene las carpetas {generador}_{split}
            split: Partición a cargar: "train", "val" o "test" tiene que ser explícita
            transform: Pipeline de transforms
        """
        self._root = Path(root)
        self._split = split
        self._transform = transform
        self._samples: list[tuple[Path, int, str]] = []  # (ruta, etiqueta, generador)
        self._generator_list: list[str] = []

        self._scan()

    def _scan(self) -> None:
        """Recorre el árbol de directorios y construye la lista de muestras"""
        generator_dirs = sorted(
            d for d in self._root.iterdir()
            if d.is_dir() and d.name.endswith(f"_{self._split}")
        )

        if not generator_dirs:
            logger.warning("No se encontraron directorios para split='%s' en '%s'.",self._split, self._root,)
            return

        for gen_dir in generator_dirs:
            generator = gen_dir.name.rsplit("_", 1)[0]
            self._generator_list.append(generator) #progan

            for category_dir in sorted(gen_dir.iterdir()): #Carpetas de categorias
                if not category_dir.is_dir():
                    continue

                for label, subdir_name in ((_LABEL_REAL, _REAL_DIR), (_LABEL_FAKE, _FAKE_DIR)):
                    img_dir = category_dir / subdir_name
                    if not img_dir.exists():
                        continue

                    for img_path in sorted(img_dir.iterdir()):
                        if img_path.suffix.lower() in _VALID_EXTENSIONS:
                            self._samples.append((img_path, label, generator))

        logger.info(
            "CNNDetectionDataset cargado - split='%s' generadores=%s muestras=%d",
            self._split, self._generator_list, len(self._samples),
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, int, str]:
        """Para dataset[idx]
        Devuelve (imagen_tensor, etiqueta, generador).
        Etiqueta 0 = real, 1 = sintético.
        """
        img_path, label, generator = self._samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self._transform is not None:
            image = self._transform(image)

        return image, label, generator

    @property
    def generators(self) -> list[str]:
        """Generadores GAN presentes en el split"""
        return list(self._generator_list)
