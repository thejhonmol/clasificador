# Guía de la parte práctica (apartados 4.1–4.5 y 5)

Esta documentación describe el código del repositorio. No son fragmentos textuales
de autores ni un informe académico presentado como trabajo humano.

## 4.1 Dataset y etiquetado

Fuente elegida: `bittlingmayer/amazonreviews`, versión 7 en Kaggle. Archivos
`train.ft.txt.bz2` y `test.ft.txt.bz2`, con etiquetas fastText al comienzo de cada
línea. El código convierte etiqueta 1 a 0 (negativo) y etiqueta 2 a 1 (positivo),
y retira la etiqueta antes de vectorizar. El título permanece unido a la reseña.

`data.read_sample` cuenta todas las filas y clases mientras toma una muestra
uniforme. Registra SHA256 de cada archivo, semilla y número de línea de cada
selección. Los conteos realmente observados figuran en `manifest.json`; la
configuración indica tamaños solicitados, y `sizes` los tamaños efectivos.

No se incorpora clase neutral. Este corpus sustituye al corpus multilingüe de
cinco estrellas mencionado anteriormente en la planificación; no son el mismo.

## 4.2 Limpieza y vectorización

`clean_text`: decodifica entidades HTML, convierte a minúsculas, elimina etiquetas
HTML, normaliza URLs al token `url` y colapsa espacios. Conserva negaciones como
`not`, `no` y `don't`; la tokenización posterior es la predeterminada de scikit-learn
(tokens de al menos dos caracteres; el apóstrofo puede separar contracciones).
No aplica traducción, lematización, stemming ni eliminación de stopwords.

TF-IDF: unigramas/bigramas, frecuencia mínima 2, máximo 50 000 rasgos y TF
sublineal. Las matrices son dispersas. Cada fold aprende vocabulario e IDF solo
con sus filas de entrenamiento. `predict.py` usa exactamente la misma limpieza.

Los duplicados se identifican por texto normalizado completo. Dentro de cada
muestra se conserva la primera repetición con etiqueta consistente; si hay
conflicto, se descartan todas las apariciones de ese texto. Se excluyen de test
los textos presentes en train. Esto no detecta paráfrasis ni agrupa por producto.

## 4.3 Partición

Se preservan los archivos oficiales de train/test. Se muestrean de forma separada
con semillas 42 y 43, sin mover filas de test a train. El train muestreado y
limpio se divide 80/20, con `stratify` y semilla 42, en ajuste/validación.
El ajuste alimenta tres folds estratificados y barajados para búsqueda de
hiperparámetros. La validación externa selecciona la familia; el test se abre
solo después de guardar `selection.json`. El ajuste final utiliza todo el
train seleccionado (ajuste + validación), nunca test.

## 4.4 Modelos e hiperparámetros

| Modelo | Parámetro explorado | Otros ajustes |
|---|---|---|
| MultinomialNB | alpha: 0.5, 1.0 | Valores no negativos de TF-IDF |
| LogisticRegression | C: 0.5, 2.0 | liblinear, max_iter=1000, seed=42 |
| LinearSVC | C: 0.5, 2.0 | dual=auto, max_iter=5000, seed=42 |
| DummyClassifier | Sin búsqueda | Clase mayoritaria del train final |

`GridSearchCV` optimiza F1 macro; la familia se elige por F1 macro de validación.
En empate exacto se conserva el primer modelo en orden NB, regresión logística,
SVM. El vocabulario no se comparte entre folds. Un solo trabajo por defecto
limita el uso de RAM; `--jobs` permite modificarlo.

## 4.5 Explicación del código

| Archivo | Responsabilidad |
|---|---|
| `clasificador/download.py` | Descargar versión 7 o explicar alternativa manual |
| `clasificador/data.py` | Parsear, limpiar, muestrear y controlar duplicados |
| `clasificador/train.py` | Partir, buscar, seleccionar, entrenar, evaluar y exportar |
| `clasificador/predict.py` | Cargar pipeline, limpiar entrada y predecir |
| `tests/test_pipeline.py` | Pruebas de datos y del flujo completo con textos sintéticos |
| `notebooks/Practica_Grupo4.ipynb` | Ejecución guiada y visualización para la exposición |

## 5. Resultados y discusión

Leer `RESULTADOS.md` y las matrices para el experimento ejecutado. Comparar contra
la línea base y considerar F1 por clase y macro, no solo accuracy. Los errores
se identifican en `errors.csv`; se recuperan sus textos de las líneas indicadas
del archivo original para discutir negación, ironía, reseñas mixtas o errores de
etiquetado sin inventar observaciones.

La selección no usa el test, aunque se publican resultados de las tres familias
para comparar. No modificar el modelo en función de ese test y volver a presentar
su resultado como evaluación independiente. No se reportan intervalos de confianza
ni generalización entre dominios. No convertir estas métricas en afirmaciones
sobre todas las reseñas de Amazon o sobre textos académicos en español.

Las filas sin tokens utilizables tras la limpieza se excluyen antes del muestreo
y se contabilizan en `skipped_empty_rows`; las etiquetas desconocidas provocan
un error. Los IDs conservan el número de línea original.
