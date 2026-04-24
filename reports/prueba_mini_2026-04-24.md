# Validacion mini del pipeline - 2026-04-24

## Contexto y objetivo

El pipeline del TFM (data -> training -> evaluation -> scripts) esta completo
para las 4 arquitecturas (ResNet-50, FreqNet, ViT-B/16, UniversalFakeDetect),
pero nunca se habia ejecutado de punta a punta porque el dataset de entrenamiento
CNNDetection ocupa ~96 GB y entrenar localmente era inviable. Antes de lanzar el
experimento completo en Colab (8-10 epocas sobre ~140.000 imagenes por modelo)
se necesita una **prueba mini** que valide que `train -> checkpoint -> evaluate`
funciona en GPU y produce `metrics.json` + plots sin fallos.

## Que se ha montado

- 4 configs mini (`configs/mini_{resnet50,freqnet,vit,universalfakedetect}.yaml`)
  con `epochs=2`, `batch_size=32`, `max_samples=2000/500/500` (train/val/E1),
  `wandb=false`, `pretrained=true` (salvo FreqNet que no admite pesos),
  `output.base_dir = /content/drive/MyDrive/tfm-checkpoints-mini`.
- Orquestador `scripts/run_all_mini.py` que encadena train + evaluate de los 4.
- Notebook `notebooks/tfm_deteccion_prueba.ipynb` que monta Drive, extrae
  solo 2 categorias de `progan_train` (car, horse) + `progan_val` completo,
  valida con un sanity check y lanza `run_all_mini.py`. Datos usados: ~6 GB.
- Refactor previo: loader compartido en `utils/config.py`, eliminado codigo muerto
  en `data/dataloader_factory.py`.

## Bug encontrado y corregido

La primera ejecucion dio `Acc=0.6000` identico en los 4 modelos y AUC ≈ 0.5
en todos menos UFD. Causa: `Subset(dataset, list(range(max_samples)))` en
`scripts/train.py` y `evaluation/evaluator.py` cogia las primeras N muestras
en orden de disco; como el scan agrupa por categoria/label, el subset quedaba
fuertemente sesgado (un unico label dominante). **Fix**: muestreo aleatorio con
semilla fija (`random.sample(..., k=max_samples)`, seed=42). El bug solo afectaba
a runs con `max_samples`; el experimento real no lo usa y no se ve impactado.

## Resultados tras el fix (mini, split E1 = progan_val con 500 imgs)

| Modelo              | AUC    | AP     | Acc    |
|---------------------|--------|--------|--------|
| ResNet-50           | 0.9743 | 0.9746 | 0.9060 |
| FreqNet             | 0.4636 | 0.4522 | 0.4720 |
| ViT-B/16            | 0.9750 | 0.9760 | 0.8940 |
| UniversalFakeDetect | 0.9974 | 0.9975 | 0.9740 |

Checkpoints y plots conservados en Drive:
`https://drive.google.com/drive/folders/1Muya6F07s-BoEWHsTUDLbLkWDT93y6wU`.

## Lectura de los numeros

- **ResNet-50 y ViT ~0.97**: los pesos ImageNet ya separan texturas naturales
  de sinteticas de ProGAN con muy pocas muestras; 2 epochs bastan para fine-tune
  de la cabeza. Era el comportamiento esperado.
- **UFD 0.9974**: practicamente saturado. El backbone CLIP congelado aporta
  features mucho mas discriminativas que ImageNet para este dominio; el
  proyector lineal aprende en pocas iteraciones. Es consistente con la
  literatura (Ojha et al. 2023).
- **FreqNet 0.46**: peor que aleatorio. FreqNet se entrena desde cero y necesita
  muchos mas datos + epocas para que el analisis en dominio frecuencial converja.
  Con 2000 muestras y 2 epocas no aprende nada util. Esperable y no bloqueante:
  el objetivo aqui no era rendimiento sino ejecucion correcta.

El pipeline se valida: los 4 modelos entrenan, guardan `checkpoint_best.pth`,
se cargan de nuevo, corren inferencia y generan `metrics.json`,
`roc_curve.png`, `confusion_matrix.png` sin excepciones.

## Intuicion para el experimento completo

Sobre CNNDetection completo (todos los generadores GAN, ~140k imgs, 8 epochs,
batch 128, pretrained=true) y con E1 = progan_test real:

| Modelo              | E1 AUC esperado | E1b cross-arch | E2 cross-gen (diffusion) |
|---------------------|-----------------|----------------|--------------------------|
| ResNet-50           | 0.95-0.99       | 0.75-0.90      | 0.55-0.75                |
| FreqNet             | 0.85-0.95       | 0.70-0.85      | 0.50-0.70                |
| ViT-B/16            | 0.95-0.99       | 0.80-0.92      | 0.60-0.78                |
| UniversalFakeDetect | 0.97-0.99       | 0.90-0.97      | 0.85-0.95                |

Intuicion clave: la jerarquia probable sera **UFD > ViT ≈ ResNet-50 > FreqNet**
en E1, pero la diferencia se amplifica en E2 (generalizacion a generadores
no vistos), que es la metrica principal del TFM. UFD es el candidato mas
robusto porque su espacio de features de CLIP no esta sesgado a artefactos de
GANs concretos; los otros tres tenderan a overfittear a patrones de ProGAN.

FreqNet, si se entrena lo suficiente, suele hacer mejor papel en E1 que lo que
sugiere el mini run, pero casi siempre es el que mas cae en E2.

## Siguiente paso

Notebook `notebooks/tfm_deteccion_completo.ipynb` para Colab que lanzara
`scripts/run_all.py` con los configs `configs/{modelo}.yaml` y E1b habilitado.

**Decision sobre categorias**: en vez de las 20 categorias completas (~96 GB
extraidos, no caben en los 113 GB de disco de Colab free) o las 18 del notebook
antiguo (que se quedaban al 95% de ocupacion), se usan **14 categorias** (~48-50
GB, margen holgado). Se conservan las pesadas y ricas en texturas (person, car,
cat, dog, train) mas una seleccion variada (airplane, bicycle, bird, boat, bus,
chair, horse, motorbike, tvmonitor). Se descartan 6 redundantes o menos
discriminativas: cow/sheep (similares), diningtable/sofa (interiores solapados
con chair), pottedplant/bottle (objetos pequenos con poca variacion). Esta
decision se documenta en la memoria como limitacion de recursos; no degrada
sensiblemente E1 ni E1b y mantiene diversidad suficiente para evaluar
generalizacion.

Tiempo estimado: ~2-3 h por modelo en T4 -> ~8-12 h total. E2 (GenImage) queda
como extension opcional: el notebook incluye plantilla para habilitarlo cuando
se suba una particion a Drive.
