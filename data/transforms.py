"""Preprocesado homogéneo compartido por todos los datasets y modelos."""

import torchvision.transforms as T

# Estadísticas de normalización estándar ImageNet (RGB)
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

INPUT_SIZE: int = 224

# Probabilidad de aplicar GaussianBlur durante el entrenamiento.
# Wang et al. (2020) muestran que blur + JPEG mejoran la generalización cross-generator.
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
