# Procedencia y reproducción

- Dataset: bittlingmayer/amazonreviews, versión 7 de Kaggle.
- Descarga: https://www.kaggle.com/api/v1/datasets/download/bittlingmayer/amazonreviews?datasetVersionNumber=7
- Se leyeron 3 600 000 filas de entrenamiento y 400 000 de test.
- Se excluyeron 3 filas de train sin tokens utilizables, antes del muestreo.
- Muestras uniformes: 20 000 de train y 5 000 de test. Ajuste: 16 000;
  validación: 4 000; entrenamiento final: 20 000; evaluación: 5 000.
- No hubo duplicados/conflictos ni solapamientos entre las muestras seleccionadas.
- Código del experimento: commit bd3315ea95ff30abdebfc32e854e8a0b704de3e5.
- Los hashes SHA256 y versiones exactas figuran en manifest.json.
- Las métricas corresponden a una muestra, no al entrenamiento con 3.6 millones.
- No se incluyen textos del corpus ni el archivo binario del modelo. Este último
  se genera localmente al ejecutar el entrenamiento.

Comando desde la raíz del repositorio (usar una carpeta nueva):

```bash
python -m clasificador.train --train data/train.ft.txt.bz2 --test data/test.ft.txt.bz2 --output outputs/reproduccion --train-size 20000 --test-size 5000 --seed 42 --jobs 1
```

Resultados: Naive Bayes 88.76 %, regresión logística 89.76 %, SVM lineal 90.14 %
de accuracy. SVM fue elegido por validación antes de abrir test.
Las matrices tienen filas reales y columnas predichas en orden negativo, positivo.
