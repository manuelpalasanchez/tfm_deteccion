# Extension cross-generator: evaluacion sobre GenImage

_Documento de referencia para el experimento adicional E2 (cross-generator
sobre modelos de difusion). Acompana al notebook `notebooks/notebook_genimage.ipynb`._

---

## 1. Motivacion

El experimento principal del TFM evalua tres detectores (ResNet-50,
ViT-B/16, UniversalFakeDetect) entrenados sobre **ProGAN** y testeados
sobre:

- **E1**: ProGAN test (in-distribution).
- **E1b**: CNN_synth_testset, 12 arquitecturas **GAN** (cross-architecture
  pero dentro del dominio GAN).

Pero el ecosistema actual de imagenes sinteticas esta dominado por
**modelos de difusion** (Stable Diffusion, Midjourney, DALL-E, etc.),
que generan imagenes con artefactos cualitativamente distintos a los
de las GANs. La pregunta abierta es: **un detector entrenado solo con
GANs, ¿generaliza a difusion?**

Esta extension añade una tercera ronda de evaluacion, **E2**, que
mide exactamente eso. No se reentrena ningun modelo; se reutilizan los
checkpoints finales del experimento principal y se evaluan sobre la
particion `val/` del dataset GenImage.

---

## 2. El dataset GenImage

### 2.1 Que es

GenImage es un dataset publico de detección de imagenes sinteticas
(NeurIPS 2023, paper: arXiv:2306.08571) que contiene imagenes
generadas por 8 arquitecturas distintas. Las imagenes reales (`nature`)
proceden de ImageNet, y las sinteticas (`ai`) se generan condicionando
cada generador sobre las mismas 1000 clases.

Repo oficial: https://github.com/GenImage-Dataset/GenImage

### 2.2 Las 8 arquitecturas

| Carpeta | Resolucion | Tipo |
|---|---|---|
| ADM | 256 px | Difusion (Ablated Diffusion Model) |
| GLIDE | 256 px | Difusion guiada por texto |
| VQDM | 256 px | Difusion vectorial cuantizada |
| BigGAN | 256 px | GAN class-conditional |
| Stable Diffusion V1.4 | 512 px | Difusion latente |
| Stable Diffusion V1.5 | 512 px | Difusion latente |
| Wukong | 512 px | Difusion latente (analoga a SD) |
| Midjourney | 1024 px | Difusion comercial |

**BigGAN se excluye** del E2: ya esta cubierta en E1b
(CNN_synth_testset) y no aporta cross-domain. **Midjourney tambien
se excluye** por motivos metodologicos y de disco (ver seccion 8.1).
Las **6 arqs restantes** (ADM, GLIDE, VQDM, SD v1.4, SD v1.5,
Wukong) son todas difusion y cubren el espectro relevante: pixel-space,
vectorial cuantizada y latente.

### 2.3 Estructura interna

Cada arquitectura, al descomprimirse, deja:

```
{arch}/
  train/
    ai/      *.png   (imagenes generadas por esa arq)
    nature/  *.png   (imagenes reales de ImageNet)
  val/
    ai/      *.png
    nature/  *.png
```

Mapeo de labels (definido en `data/genimage_dataset.py`):
- `nature` -> 0 (real)
- `ai` -> 1 (sintetico)

### 2.4 Formato de distribucion (multi-volume zip)

En Drive cada arq esta empaquetada como un **ZIP multi-volumen**
(formato split zip de Info-ZIP):

```
{arch}/
  archive.z01       (volumen 1, ~tamaño fijo p.ej. 4 GB)
  archive.z02       (volumen 2)
  ...
  archive.zNN       (volumenes intermedios)
  archive.zip       (volumen final, contiene el central directory)
```

Detalles importantes:
- Para extraer hay que tener **todos los volumenes** juntos en el mismo
  directorio. No se descomprimen por separado.
- El `.zip` final lleva el indice (central directory); los `.zNN` solo
  contienen datos.
- Los nombres son los nativos del zip (no se pueden renombrar).

Tamaños aproximados de los archivos comprimidos por arq:
- 256 px (ADM, GLIDE, VQDM, BigGAN): ~15-25 GB.
- 512 px (SD v1.4, SD v1.5, Wukong): ~60-90 GB.
- 1024 px (Midjourney): >100 GB.

---

## 3. Restricciones y estrategia de carga

### 3.1 El problema

Colab Free ofrece **~80 GB efimeros** en `/content`. No caben:
- Varias arquitecturas a la vez (de 512 px en adelante).
- Cualquier arq descomprimida con su `train/` incluido.

### 3.2 La estrategia

Procesar **una arq cada vez**, descargar volumenes, descomprimir
**solo `val/`** (que es lo que necesitamos), borrar los volumenes
antes de pasar a la siguiente. Implementado en `notebook_genimage.ipynb`:

1. **Listar** con Drive API las 8 subcarpetas de la raiz de GenImage
   (folder_id pasado por el usuario).
2. **Filtrar** las arqs de difusion (excluir BigGAN).
3. Por cada arq:
   - Sumar tamanos comprimidos. Si excede el disco libre menos un
     margen (15 GB), **saltar con aviso**.
   - Descargar los `.zNN + .zip` a `/content/genimage_zips/`.
   - `7z x archive.zip "*/val/*" -o/content/genimage/` para
     descomprimir solo la particion `val/`.
   - `rm -rf /content/genimage_zips/` para liberar espacio.
4. Al final, `/content/genimage/{arch}/val/{ai,nature}/` contiene
   las imagenes de las arqs que pasaron el check. Suma estimada:
   7 arqs * ~1.5 GB val = ~10 GB.

### 3.3 Por que 7z y no unzip

`unzip` estandar no soporta bien los split zips de Info-ZIP. La
alternativa documentada (`zip -s 0 archive.zip --out joined.zip &&
unzip joined.zip`) requiere duplicar el zip en disco, lo cual no cabe
para arqs grandes. `7z` (paquete `p7zip-full` de apt) descomprime
multi-volumenes directamente sin reconstruccion intermedia y permite
**filtrar por patron** durante la extraccion (`"*/val/*"`), lo que
evita descomprimir el train.

### 3.4 Midjourney: excluida por decision metodologica

Midjourney queda fuera de E2 por una combinacion de restriccion de
disco y argumento metodologico (el resize a 224x224 destruye su
senal caracteristica de 1024 px). Detalle completo en la **seccion
8.1** de este documento.

En el codigo del notebook esta excluida via `EXCLUIR_ARQS` para
mantener coherencia con la documentacion. Reactivarla requiere quitar
`'midjourney'` de ese set, pero hacerlo sin adaptar la resolucion de
entrada del pipeline daria resultados enganiosos (ver 8.1).

---

## 4. Estrategia de evaluacion

### 4.1 Por que no usar `scripts/evaluate.py`

`scripts/evaluate.py` esta acoplado al sistema de configs YAML
(`base.yaml -> eval_e2.enabled=true`) y al `evaluator.py`, que asume el
patron CNNDetectionDataset (estructura `{gen}_{split}/{cat}/0_real|1_fake`).
GenImage tiene **otra estructura** (`{gen}/{split}/{ai|nature}`) y
**otro nombre de dataset** (`genimage` vs `cnndetection`). Parchear
todo para que el evaluador funcione con GenImage seria mas codigo
fragil que hacer la inferencia directamente.

### 4.2 Inferencia manual dentro del notebook

El patron es identico al de la celda 36 del notebook principal
(desglose E1b por arquitectura GAN):

```python
cfg = load_config(info['config'])
importlib.import_module(info['module'])
model = model_registry.build(cfg.model.name, **vars(cfg.model.kwargs))
ckpt = torch.load(str(info['checkpoint']), map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.to(DEVICE).eval()

dataset = GenImageDataset(root='/content/genimage', split='val',
                          transform=get_eval_transforms())
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

with torch.no_grad():
    for images, labels, generators in loader:
        images = model.preprocess(images.to(DEVICE))
        logits = model(images).squeeze(1)
        scores = torch.sigmoid(logits).cpu().numpy()
        # ...recoger y_true, y_score, generator
```

`GenImageDataset` ya existe en `data/genimage_dataset.py` (no requiere
modificacion). Devuelve `(image, label, generator)` exactamente como
`CNNDetectionDataset`, asi que el codigo de la inferencia es
reutilizable.

### 4.3 Metricas calculadas

- **AUC-ROC** (principal): metrica invariante al umbral y al N.
- **AP** (Average Precision).
- **Accuracy** con threshold 0.5.
- **Matriz de confusion** [[TN, FP], [FN, TP]].
- **Desglose por arquitectura**: las mismas 4 metricas calculadas
  filtrando los samples por el campo `generator` del dataset.

### 4.4 Comparabilidad con E1b

El N de E2 es mayor que el de E1b (~12 k por arq de difusion frente a
~2.5 k por arq GAN en CNN_synth_testset). Esto **no compromete la
comparativa**:

- AUC y AP son invariantes al tamano del split (siempre que sea
  estadisticamente significativo; ambos lo son).
- Accuracy depende del balance real/fake. GenImage val esta 50/50 por
  construccion (~6 k `ai` + ~6 k `nature` por arq), igual que
  CNN_synth_testset. Comparable.
- En la memoria conviene reportar el N junto a cada metrica para que
  el lector tenga el contexto.

No se submuestrea GenImage para igualarlo al N de E1b: perderiamos
poder estadistico sin ganancia metodologica.

---

## 5. Salidas del experimento

Todo se guarda en Drive bajo
`MyDrive/tfm-checkpoints/figuras_memoria_genimage/`:

| Archivo | Contenido |
|---|---|
| `predicciones_{modelo}.pt` | tensores y_true, y_score, generator por muestra |
| `metricas_genimage.json` | AUC/AP/Acc por modelo, agregado y por arq |
| `desglose_genimage_por_arquitectura.csv` | tabla 7 arqs x 3 modelos x 3 metricas |
| `matrices_confusion_e2.png` | matrices de confusion comparativas |
| `barras_metricas_e2.png` | AUC/AP/Acc agrupadas por modelo |
| `roc_curves_e2_superpuestas.png` | 3 curvas ROC superpuestas |
| `heatmap_auc_por_generador_e2.png` | arq x modelo, ordenado por AUC media |
| `comparativa_gan_vs_difusion.png` | AUC en E1 vs E1b vs E2 por modelo |

Las predicciones `.pt` permiten regenerar las figuras sin reejecutar
inferencia. Las figuras se generan a 300 dpi en estilo academico, con
las mismas convenciones que las del experimento principal.

---

## 6. Reproduccion paso a paso

### Requisitos

- Acceso a la carpeta de GenImage en Drive (folder_id necesario; el
  usuario pega el enlace o el id al inicio del notebook).
- Los 3 checkpoints `MyDrive/tfm-checkpoints/{modelo}_*/checkpoint_best.pth`
  del experimento principal.
- Colab Free con GPU T4 (o equivalente).

### Sesion esperada

1. **Setup** (~10 min): mount Drive, git clone, pip install,
   `apt-get install p7zip-full`.
2. **Auth Drive API** (~1 min): igual que en el notebook principal.
3. **Listado de arqs** (~1 min): listar las 8 carpetas + tamanos.
4. **Bucle de descarga + extraccion** (~1-2 h):
   - ADM, GLIDE, VQDM (256 px): ~10-15 min cada una.
   - SD v1.4, SD v1.5, Wukong (512 px): ~25-40 min cada una.
   - Midjourney: skip por espacio.
5. **Carga de checkpoints** (~2 min).
6. **Inferencia** (~30-45 min: ~10-15 min por modelo).
7. **Metricas + figuras** (~5 min).
8. **Resumen** (~1 min).

**Total estimado: ~3-4 h en una sola sesion de Colab Free.**

### Recuperacion ante caidas

- Las descargas no se reanudan; si Colab corta, hay que volver a
  empezar el bucle de descarga desde la siguiente arq.
- Las predicciones `.pt` se guardan en Drive segun se generan, asi que
  si se cae despues de un modelo, los siguientes no se pierden.

---

## 7. Resultados esperados

Hipotesis a contrastar:

1. **Caida significativa de AUC** respecto a E1b (cross-arch dentro de
   GAN). El degradado tipico en literatura es del orden de 10-30
   puntos de AUC cuando se cruza de GAN a difusion.
2. **UFD deberia ser el mas robusto** (relativamente): su backbone CLIP
   no esta sesgado a artefactos GAN especificos, mientras que ResNet y
   ViT entrenados con ProGAN pueden quedarse muy especificos.
3. **Heatmap por arq esperado**: las SDs (V1.4, V1.5, Wukong, todas
   difusion latente) suelen agruparse en rendimiento; ADM y GLIDE
   (difusion pixel-space) pueden dar resultados muy distintos.

Estas hipotesis se actualizaran con los resultados reales una vez se
ejecute el notebook.

---

## 8. Limitaciones

### 8.1 Midjourney queda fuera de la comparativa

Midjourney no se evalua en E2 por dos razones independientes:

**(a) Restricciones de disco**. Los volumenes comprimidos en Drive
suman >100 GB para esa arq (resolucion nativa 1024 px). El disco
efimero de Colab Free son ~80 GB libres. No caben ni siquiera para
descargar y descomprimir solo `val/`. El notebook detecta el caso
en el check de espacio y la salta automaticamente.

**(b) Argumento metodologico (mas importante que el anterior)**. La
pipeline de preprocesado del experimento redimensiona toda imagen a
**224 x 224** (`data/transforms.py`, `Resize((INPUT_SIZE, INPUT_SIZE))`)
antes de la inferencia. Las imagenes nativas de Midjourney son
1024 x 1024 (factor de downscale ~4.5x). Los artefactos
caracteristicos de Midjourney - granularidad en el procesado de luz,
estilo de color, patrones de alta frecuencia - viven precisamente en
esa resolucion alta. Aplicar `Resize(224, 224)` los destruye antes
de que el detector pueda verlos.

Esto implica que aunque la pudieramos evaluar (con un Colab Pro o
preprocesando en local), los resultados serian enganiosos en
cualquier direccion:

- AUC alto -> no demostraria deteccion de Midjourney, sino deteccion
  de artefactos genericos de difusion que sobreviven al downscale.
- AUC bajo -> no indicaria que Midjourney sea mas dificil, sino que
  hemos tirado a la basura su senal caracteristica al redimensionar.

Una evaluacion rigurosa sobre Midjourney requeriria mantener la
resolucion nativa o adaptar el pipeline (entrada 512 o 1024,
posiblemente reentrenar). Eso constituye un experimento independiente
y queda fuera del alcance de este TFM.

**Que decir en la memoria**: Midjourney se documenta como caso no
evaluado, con la justificacion metodologica anterior. Las 6
arquitecturas de difusion evaluadas (ADM, GLIDE, VQDM, SD v1.4,
SD v1.5, Wukong) ya cubren el espectro relevante:

- Difusion en pixel-space: ADM, GLIDE.
- Difusion vectorial cuantizada: VQDM.
- Difusion latente: SD v1.4, SD v1.5, Wukong (este ultimo es
  esencialmente la variante asiatica de SD entrenada con datos
  diferentes, no aporta nueva arquitectura).

Midjourney es difusion latente comercial; no anade arquitectura
nueva, solo "marca" y resolucion. La conclusion cross-domain
GAN -> difusion queda demostrada igual con las 6 arqs.

### 8.2 Otras limitaciones

- **Solo particion val**: no se evalua sobre `train/` porque (a) lleva
  el mismo tipo de imagenes y no aporta senal nueva, (b) ocupa 99 % del
  zip y no cabe.
- **Sin reentrenamiento**: los detectores siguen viendo solo ProGAN
  durante el entrenamiento; este es exactamente el setup cross-domain
  que se quiere medir. Reentrenar con difusion seria otro experimento.
- **Una sola seed**: los checkpoints son los del experimento principal,
  no se promedia sobre varias seeds (limite de computo).
- **Comparabilidad de N con E1b**: el split de E2 es mayor (~12 k
  por arq de difusion) que el de E1b (~2.5 k por arq de GAN). AUC y
  AP son invariantes a N siempre que sea estadisticamente
  significativo; Accuracy depende solo del balance real/fake (50/50
  en ambos casos por construccion). La comparativa es valida; se
  reporta el N junto a cada metrica para contexto.

### 8.3 Linea de trabajo futuro

- Evaluacion de Midjourney en su resolucion nativa (pipeline adaptado).
- Re-entrenamiento de los detectores con un trainset mixto GAN +
  difusion para medir si la generalizacion mejora.
- Inclusion de generadores mas recientes (SD v2.x, SDXL, FLUX,
  DALL-E 3) que han aparecido despues de la publicacion de GenImage.
