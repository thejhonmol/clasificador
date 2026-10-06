# Clasificador de sentimientos — Grupo 4

Implementación práctica con **Amazon Reviews for Sentiment Analysis**, de
[bittlingmayer en Kaggle](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews).
Compara Naive Bayes multinomial, regresión logística y SVM lineal con TF-IDF.
Incluye notebook, entrenamiento por consola, predicción, evaluación y pruebas.

**Alcance:** clasificación binaria de reseñas de productos principalmente en inglés.
`__label__1` = negativo (1–2 estrellas); `__label__2` = positivo (4–5 estrellas).
No contiene clase neutral; no es un dataset de comentarios académicos en español.
No se traduce el corpus ni se inventan etiquetas adicionales.

## Inicio rápido en Windows 11 (PowerShell)

Requiere Python **3.11 o 3.12** y Git. Desde una terminal:

```powershell
git clone https://github.com/thejhonmol/clasificador.git
cd clasificador
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m clasificador.download
.\.venv\Scripts\python.exe -m clasificador.train --train data/train.ft.txt.bz2 --test data/test.ft.txt.bz2 --output outputs/amazon --train-size 20000 --test-size 5000
.\.venv\Scripts\python.exe -m clasificador.predict --model outputs/amazon/model.joblib --text "This product works perfectly and is very useful"
```

No es necesario activar el entorno ni cambiar la política de PowerShell.
Si tienes Python 3.11, sustituye `py -3.12` por `py -3.11`.

En Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m clasificador.download
python -m clasificador.train --train data/train.ft.txt.bz2 --test data/test.ft.txt.bz2 --output outputs/amazon
python -m clasificador.predict --model outputs/amazon/model.joblib --text "It broke after one day and does not work"
```

La descarga pública usa la versión 7 (~517 MB). Si Kaggle requiere iniciar sesión,
descarga el ZIP desde su página y coloca `train.ft.txt.bz2` y `test.ft.txt.bz2` en
`data/`. **No hace falta descomprimir los `.bz2`**. También se admiten `.txt`.
Reserva al menos 2 GB de disco para la descarga y sus archivos temporales.
El consumo de RAM/tiempo depende del tamaño de muestra; comienza con los valores
predeterminados y un solo proceso. No se requiere GPU.

## Notebook para la exposición

[Ver notebook](notebooks/Practica_Grupo4.ipynb) ·
[Abrir en Google Colab](https://colab.research.google.com/github/thejhonmol/clasificador/blob/main/notebooks/Practica_Grupo4.ipynb)

El notebook instala dependencias, obtiene el repositorio y los datos, ejecuta el
experimento, muestra las métricas y permite probar una reseña. Revisa las celdas
antes de ejecutarlas. En Jupyter local, instala además `jupyter` en tu entorno.

## Procedimiento experimental

1. Recorrer los archivos oficiales completos y tomar muestras uniformes con
   *reservoir sampling*: 20 000 filas de train y 5 000 de test, semillas 42 y 43.
2. Limpiar HTML, espacios y URLs; conservar palabras de negación. No eliminar
   stopwords ni aplicar stemming. Mantener el título junto al texto.
3. Eliminar duplicados exactos normalizados y textos con etiquetas contradictorias
   dentro de cada muestra. Quitar de test cualquier texto presente en train.
4. Separar train en 80 % de ajuste y 20 % de validación, de forma estratificada.
5. Buscar dos hiperparámetros por modelo mediante `GridSearchCV`, con tres folds
   estratificados sobre ajuste. El `Pipeline` aprende TF-IDF dentro de cada fold.
6. Seleccionar la familia de modelo por **F1 macro de validación**, antes de abrir test.
7. Reentrenar cada configuración elegida con ajuste + validación y evaluar en test.
   La comparación final no cambia el ganador seleccionado anteriormente.
8. Comparar con un clasificador trivial de la clase mayoritaria. Exportar métricas,
   matrices, trazabilidad de filas y el modelo ganador.

TF-IDF usa unigramas y bigramas, `min_df=2`, `sublinear_tf=True` y hasta 50 000
características. NB explora `alpha=[0.5,1.0]`; regresión logística y SVM exploran
`C=[0.5,2.0]`. Consulte [metodología y código](docs/METODOLOGIA.md).

## Resultados y archivos generados

Los resultados del experimento ejecutado se publican en [results/amazon_20k](results/amazon_20k/RESULTADOS.md).
Los archivos de una ejecución nueva se guardan en `--output`:

| Archivo | Contenido |
|---|---|
| `RESULTADOS.md`, `metrics.csv` | Accuracy, precisión, recall y F1 macro de test |
| `selection.json`, `cv_*.csv` | Validación, parámetros y búsqueda cruzada |
| `classification_reports.json` | Métricas y soporte por clase |
| `confusion_*.png`, `confusion_*.json` | Matrices de confusión |
| `manifest.json` | SHA256, semillas, tamaños, versiones y configuración |
| `splits.csv` | Archivo y número de línea de cada partición |
| `predictions.csv`, `errors.csv` | Etiquetas/predicciones e IDs de errores, sin textos |
| `model.joblib` | Pipeline del ganador para predicción local |

El lector recorre todo el archivo aunque se solicite una muestra: evita tomar
solo las primeras reseñas, pero requiere tiempo de lectura. `--train-size 0` y
`--test-size 0` solicitan todo el corpus y pueden exigir mucha memoria; la búsqueda
TF-IDF no es entrenamiento incremental. Una salida ya existente no se sobrescribe:
escoge otra carpeta para repetir el experimento.

## Pruebas

```bash
python -m unittest discover -s tests -v
```

Las pruebas usan textos sintéticos y temporales; sus métricas no se presentan
como evidencia experimental sobre Amazon. Comprueban etiquetas, limpieza,
muestreo, lectura bz2, deduplicación, separación de datos, exportación y predicción.

## Límites y atribución

Las estrellas son una aproximación al sentimiento. No se controlan producto,
autor, sarcasmo ni duplicados aproximados; las métricas de una muestra no equivalen
a las del corpus completo. El modelo devuelve una clase binaria, sin probabilidad
calibrada ni capacidad validada para español. Carga solo modelos joblib de confianza.

Se sube código, documentación técnica y evidencia de ejecución, no el corpus ni
credenciales. Kaggle informa licencia `Unknown` en sus metadatos; consulta las
condiciones de la fuente antes de redistribuir los datos. El código de esta
implementación no es una cita textual de artículos. Las fuentes están en
[REFERENCIAS.md](docs/REFERENCIAS.md); el grupo debe redactar y justificar su
informe académico a partir de los resultados y de las fuentes originales.

Las filas sin tokens utilizables tras la limpieza se excluyen antes del muestreo
y se contabilizan en `skipped_empty_rows`; las etiquetas desconocidas provocan
un error. Los IDs conservan el número de línea original.
