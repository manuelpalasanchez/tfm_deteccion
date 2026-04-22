"""Preprocesado homogéneo compartido por todos los datasets y modelos."""

import torchvision.transforms as T
from torchvision.models import ResNet50_Weights

# Media y std extraídas directamente de los metadatos de los pesos IMAGENET1K_V2
# Todos los modelos preentrenados en ImageNet comparten estos valores.
_imagenet_meta = ResNet50_Weights.IMAGENET1K_V2.transforms()
IMAGENET_MEAN: list[float] = list(_imagenet_meta.mean)
IMAGENET_STD: list[float] = list(_imagenet_meta.std)

INPUT_SIZE: int = 224

# GaussianBlur destruye artefactos espectrales de alta frecuencia propios de cada GAN.
# Sin él, el detector aprende firmas específicas del generador de train y no generaliza.
# Fuente: Wang (2020)"CNN-generated images are surprisingly easy to spot"
_BLUR_PROB: float = 0.5
_BLUR_KERNEL: int = 3


def get_train_transforms() -> T.Compose:
    """Pipeline de augmentación para el split de entrenamiento."""
    return T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.GaussianBlur(kernel_size=_BLUR_KERNEL)], p=_BLUR_PROB),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms() -> T.Compose:
    """Pipeline determinista para validación y test (sin augmentación)."""
    return T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])