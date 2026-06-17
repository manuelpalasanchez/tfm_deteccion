# Deteccion de imagenes sinteticas generadas por modelos de difusion

Codigo del TFM "Deteccion de imagenes sinteticas generadas por modelos de difusion:
evaluacion comparativa de arquitecturas de Deep Learning" (Universidad, 2026).

Implementa un framework modular en PyTorch que entrena y evalua tres detectores
(ResNet-50, ViT-B/16, UniversalFakeDetect) sobre el dataset CNNDetection, con tres
protocolos de evaluacion: in-distribution (E1), cross-architecture GAN (E1b) y
cross-paradigm diffusion (E2).

---

## Estructura del repositorio

```
tfm_deteccion/
├── configs/                # Configuraciones YAML de entrenamiento
│   ├── base.yaml           # Defaults compartidos (epochs, lr, scheduler, loss)
│   ├── resnet50.yaml       # ResNet-50 pretrained ImageNet1K_V2
│   ├── vit.yaml            # ViT-B/16 pretrained ImageNet-21k (batch=64 por OOM en T4)
│   ├── universalfakedetect.yaml  # CLIP ViT-L/14 frozen + head lineal (epochs=3)
│   ├── freqnet.yaml        # FreqNet en reserva (sin pretrain, bajo rendimiento sin el)
│   └── mini/               # Configs de la prueba de concepto inicial (2 categorias, 2 epochs)
├── data/                   # Modulos de dataset y transformaciones
│   ├── base_dataset.py     # Clase abstracta BaseDataset
│   ├── cnndetection_dataset.py   # CNNDetection (ProGAN + 12 GANs cross-arch)
│   ├── genimage_dataset.py       # GenImage para E2 (6 modelos de difusion)
│   ├── transforms.py       # Pipeline comun: resize 224, norm ImageNet, blur aug
│   └── dataloader_factory.py     # Registry @register_dataset + build_dataloader
├── models/                 # Definiciones de arquitecturas
│   ├── base_model.py       # Interfaz abstracta BaseDetector
│   ├── model_registry.py   # Registry @register + build()
│   ├── resnet50.py
│   ├── vit.py
│   ├── universalfakedetect.py
│   └── freqnet.py          # En reserva
├── training/
│   ├── trainer.py          # Bucle de entrenamiento: forward, backprop, scheduler, wandb
│   └── losses.py           # BCEWithLogitsLoss con pos_weight opcional
├── evaluation/
│   ├── evaluator.py        # Rondas E1, E1b, E2 con inferencia y guardado de metricas
│   └── metrics.py          # AUC-ROC, AP, Accuracy, curvas ROC, matrices de confusion
├── scripts/
│   ├── train.py            # Entrenamiento de un modelo desde config YAML
│   ├── evaluate.py         # Evaluacion de un checkpoint
│   ├── build_trainset_drive_api.py   # Muestreo del trainset desde Drive con streaming
│   ├── build_trainset_ejecucion.py   # Alternativa offline desde zip local
│   ├── run_all.py          # Entrena y evalua los 3 modelos activos en secuencia
│   └── scan_cnndetection.py          # Inventario del dataset (rutas, conteos)
├── notebooks/
│   ├── tfm_deteccion_ejecucion.ipynb  # Notebook principal: pipeline completo en Colab
│   ├── notebook_genimage.ipynb        # Extension E2: evaluacion cross-paradigm
│   ├── notebook_genimage_analisis_familia.ipynb  # Analisis E2 por familia de difusion
│   └── figuras_resultados.ipynb       # Regeneracion de figuras E1b sin reentrenar
├── tests/                  # Suite pytest (modelos, datasets, trainer, registry)
├── utils/
│   └── config.py           # Carga YAML con herencia via _base_
└── reports/                # Resultados y documentacion de experimentos
```

---

## Requisitos

- Python 3.11
- CUDA 12.x (los experimentos se ejecutaron en Google Colab con GPU T4/L4)
- Dependencias listadas en `requirements.txt`

```
pip install -r requirements.txt
```

Los pesos de los modelos se descargan automaticamente la primera vez via HuggingFace
(`google/vit-base-patch16-224`, `openai/clip-vit-large-patch14`) y torchvision
(ResNet-50 ImageNet1K_V2).

---

## Datos

El dataset de entrenamiento (CNNDetection, split ProGAN train, ~70 GB) vive en Google Drive.
No se incluye en el repositorio.

### Preparacion del trainset en Colab

```bash
python scripts/build_trainset_drive_api.py \
    --file-id <ID_del_zip_en_Drive> \
    --out /content/cnndetection/progan_train \
    --n-per-cat 5000 \
    --seed 42
```

Extrae por streaming un subconjunto de 5.000 imagenes por categoria (2.500 reales +
2.500 sinteticas), 20 categorias, total 100.000 imagenes. El muestreo es determinista:
la misma llamada con `--seed 42` produce siempre el mismo subconjunto.

El dataset de validacion y test (progan_val, progan_test, CNN_synth_testset) sigue el
mismo patron: se extrae desde Drive antes de ejecutar la evaluacion. El notebook
`tfm_deteccion_ejecucion.ipynb` contiene el flujo completo con las celdas de extraccion.

### Estructura esperada en disco

```
/content/cnndetection/
├── progan_train/
│   └── {categoria}/
│       ├── 0_real/   # imagenes reales
│       └── 1_fake/   # imagenes ProGAN
├── progan_val/        # mismo patron
└── progan_test/       # mismo patron

/content/CNN_synth_testset/
└── {arquitectura_gan}_test/  # 12 GANs para E1b
```

Los paths en los configs YAML apuntan a `/content/...` (rutas de Colab). Ajustar si
se ejecuta en local.

---

## Reproducir un experimento

### Via notebook (forma principal)

Abrir `notebooks/tfm_deteccion_ejecucion.ipynb` en Google Colab y ejecutar
secuencialmente. El notebook cubre: mount de Drive, extraccion del trainset,
entrenamiento de los tres modelos y evaluacion E1 + E1b.

### Via scripts CLI

**Entrenamiento de un modelo:**

```bash
python scripts/train.py --config configs/resnet50.yaml
```

Guarda el checkpoint en `experiments/runs/resnet50_YYYYMMDD_HHMMSS/checkpoint_best.pth`.

**Evaluacion de un checkpoint:**

```bash
python scripts/evaluate.py \
    --config configs/resnet50.yaml \
    --checkpoint experiments/runs/resnet50_YYYYMMDD_HHMMSS/checkpoint_best.pth
```

Guarda metricas JSON, curvas ROC y matrices de confusion en el mismo directorio del
checkpoint.

**Entrenamiento y evaluacion de los tres modelos en secuencia:**

```bash
python scripts/run_all.py
```

---

## Seed 42 y reproducibilidad

El muestreo del trainset usa `random.Random(42)` en dos puntos:

1. `build_trainset_drive_api.py`: selecciona los 5.000 ficheros por categoria del zip.
2. `scripts/train.py`: cuando `max_samples` esta definido en el config, aplica un
   `random.Random(42).sample()` sobre el dataset ya cargado (usado en los configs mini).

Los experimentos finales usaron el trainset extraido por el primer punto; el segundo
solo actua en los configs mini de prueba.

---

## Entorno de ejecucion

Los experimentos finales se ejecutaron en Google Colab Pro con GPU L4 (24 GB VRAM).
Los configs estan ajustados para ese entorno:

- ViT usa `batch_size: 64` en lugar de 128 para evitar OOM en T4.
- UniversalFakeDetect usa `epochs: 3` porque el backbone CLIP esta congelado y
  converge mas rapido.
- `num_workers: 0` en los configs (comportamiento de Colab con el filesystem de Drive).

No se garantiza reproduccion exacta en entornos distintos a Colab con GPU L4 o T4.

---

## Tests

```bash
pytest tests/
```

La suite cubre: instanciacion y forward pass de los cuatro modelos, registros de
modelos y datasets, bucle de entrenamiento simplificado.
