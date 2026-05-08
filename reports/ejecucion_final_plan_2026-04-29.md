# Plan de ejecucion final - 2026-04-29

## Contexto

Tras la prueba mini del 2026-04-24 (pipeline validado e2e en Colab GPU,
ver reports/prueba_mini_2026-04-24.md) se acota el alcance del experimento
final para que sea estable, reproducible y ejecutable en Colab free en una
unica sesion.

## Alcance final

- **Dataset**: solo CNNDetection. GenImage y FaceForensics quedan fuera.
- **Arquitecturas**: 3 activas (ResNet-50, ViT-B/16, UniversalFakeDetect).
  FreqNet en stand-by (validado en el mini que sin pretrain no rinde).
- **Train**: trainset de ejecucion de 100.000 imagenes (5000 por categoria,
  balance 50/50 real/fake, seed=42) muestreado del zip progan_train.zip.
- **Val**: progan_val completo (8.000 imgs).
- **Test**: progan_test completo (E1, in-distribution) + CNN_synth_testset
  completo (E1b, cross-architecture).
- **Hiperparametros**: 8 epocas, batch 128, lr=1e-4 (1e-3 para UFD), Adam,
  CosineAnnealingLR, BCEWithLogitsLoss.

## Por que un trainset de ejecucion (resumen)

El zip de progan_train pesa ~70 GB descomprimido y ~720k imgs en 20
categorias balanceadas (~36k cada una). Extraerlo entero llena el disco de
Colab al 95% y deja sin margen para evaluacion. Entrenar con todo se va a
varios dias, durante los cuales Colab corta la sesion al menos una vez.

La solucion es muestrear sin descomprimir: scripts/build_trainset_ejecucion.py
abre el zip, indexa por (categoria, label), elige 5000 imgs por categoria
con seed fija y extrae solo esas. El subset ocupa ~10 GB (vs ~70 GB) y se
construye en ~5-8 min (vs ~60 min de unzip completo). Detalles y metricas
de ahorro en problema_carga_datasets.md.

## Resultados esperados (intuicion)

A falta de ejecutar, expectativa razonable sobre cada ronda con backbones
preentrenados y este trainset:

| Modelo              | E1 AUC esperado | E1b cross-arch esperado |
|---------------------|-----------------|--------------------------|
| ResNet-50           | 0.95-0.99       | 0.75-0.90                |
| ViT-B/16            | 0.95-0.99       | 0.80-0.92                |
| UniversalFakeDetect | 0.97-0.99       | 0.90-0.97                |

Hipotesis principal del TFM: la jerarquia probable sera
**UFD > ViT >= ResNet-50** en E1, pero la diferencia se amplifica en E1b.
UFD es el candidato mas robusto porque su espacio de features de CLIP no
esta sesgado a artefactos de GANs concretos; los otros tienden a
overfittear a patrones de ProGAN.

## Pasos de ejecucion

Notebook: notebooks/tfm_deteccion_ejecucion.ipynb (Colab T4).

1. Drive + clonar repo + pip install (celdas 1-3).
2. Inventario opcional del zip (celda 5; ya validado en este informe).
3. Construir trainset de ejecucion (celda 7): genera
   /content/cnndetection/progan_train/ + manifest JSON.
4. Extraer val + tests completos desde los zips de Drive (celda 9).
5. Patch de base.yaml para apuntar a /content/cnndetection y guardar
   checkpoints en Drive (celda 11).
6. wandb login si se quiere tracking online (celda 13, opcional).
7. Sanity check de los 3 splits (celda 15).
8. python scripts/run_all.py (celda 17): entrena y evalua los 3 modelos
   en serie. Genera por cada modelo: checkpoint_best.pth, metrics.json,
   roc_curve.png, confusion_matrix.png.
9. Resumen tabular + CSV (celda 22).

Tiempo estimado total en T4: ~3-3.5 h (incluye descompresion de val+tests
y tres train+eval).

## Recuperacion ante caidas

- Los checkpoints viven en /content/drive/MyDrive/tfm-checkpoints/, no se
  pierden si Colab corta la sesion.
- El subset se reconstruye exactamente igual con la seed; basta relanzar
  la celda 7.
- Para retomar solo la evaluacion sin reentrenar:
  `python scripts/run_all.py --eval-only`.

## Limitaciones documentadas

- Resultados sobre una unica semilla de muestreo. No se reportan
  intervalos de confianza por coste de computo.
- E2 (cross-generator a difusion) no se evalua: GenImage no esta
  disponible localmente y queda como extension futura. El codigo del
  dataset (data/genimage_dataset.py) y el bloque eval_e2 en base.yaml se
  conservan para reactivacion.
- FreqNet queda fuera de la comparativa final por bajo rendimiento sin
  pretrain en el computo disponible (validado en el mini).
- Colab free no garantiza GPU continua mas alla de unas horas; por eso el
  pipeline esta disenado para caber en una sola sesion (~3-3.5 h).
