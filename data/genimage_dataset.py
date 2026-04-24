"""Dataset GenImage para evaluacion cross-generator (E2).
    Sacado del repo de GenImage: https://github.com/GenImage-Dataset/GenImage
├── Midjourney
│   ├── train
│   │   ├── ai
│   │   ├── nature
│   ├── val
│   │   ├── ai
│   │   ├── nature
├── VQDM
│   ├── train
│   │   ├── ai
│   │   ├── nature
│   ├── val
│   │   ├── ai
│   │   ├── nature
├── Wukong
│   ├── ...
├── Stable Diffusion V1.4
│   ├── ...
├── Stable Diffusion V1.5
│   ├── ...
├── GLIDE
│   ├── ...
├── BigGAN
│   ├── ...
├── ADM
│   ├── ...
"""

import logging
from pathlib import Path

from PIL import Image
from torch import Tensor

from data.base_dataset import BaseDataset
from data.dataloader_factory import register_dataset

logger = logging.getLogger(__name__)

_REAL_DIRS = {"nature", "0_real"}
_FAKE_DIRS = {"ai", "1_fake"}
_LABEL_REAL = 0
_LABEL_FAKE = 1
_VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@register_dataset("genimage")
class GenImageDataset(BaseDataset):
    """Dataset GenImage.

    Cada muestra es (imagen_tensor, etiqueta, generador).
    Etiqueta 0 = real, 1 = sintetico.
    """

    def __init__(self, root: str, split: str, transform=None) -> None:
        self._root = Path(root)
        self._split = split
        self._transform = transform
        self._samples: list[tuple[Path, int, str]] = []
        self._generator_list: list[str] = []

        self._scan()

    def _scan(self) -> None:
        generator_dirs = sorted(d for d in self._root.iterdir() if d.is_dir())

        if not generator_dirs:
            logger.warning(
                "No se encontraron generadores en '%s'.", self._root
            )
            return

        for gen_dir in generator_dirs:
            split_dir = gen_dir / self._split
            if not split_dir.exists():
                continue

            generator = gen_dir.name
            found_in_generator = 0

            for subdir in sorted(split_dir.iterdir()):
                if not subdir.is_dir():
                    continue

                name = subdir.name.lower()
                if name in _REAL_DIRS:
                    label = _LABEL_REAL
                elif name in _FAKE_DIRS:
                    label = _LABEL_FAKE
                else:
                    continue

                for img_path in sorted(subdir.rglob("*")):
                    if img_path.suffix.lower() in _VALID_EXTENSIONS:
                        self._samples.append((img_path, label, generator))
                        found_in_generator += 1

            if found_in_generator > 0:
                self._generator_list.append(generator)

        logger.info(
            "GenImageDataset cargado - split='%s' generadores=%s muestras=%d",
            self._split,
            self._generator_list,
            len(self._samples),
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, int, str]:
        img_path, label, generator = self._samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self._transform is not None:
            image = self._transform(image)
        return image, label, generator

    @property
    def generators(self) -> list[str]:
        return list(self._generator_list)
