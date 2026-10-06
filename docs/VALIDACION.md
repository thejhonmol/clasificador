# Validación técnica

- Entorno local: Python 3.12.14; dependencias directas fijadas en requirements.txt.
- `python -m unittest discover -s tests -v`: 7 pruebas superadas.
- Comprobadas etiquetas, negaciones, lectura `.bz2`, muestreo reproducible,
  exclusión y conteo de textos vacíos, duplicados/conflictos, separación de
  vocabulario, persistencia y predicción en un proceso independiente.
- Todas las celdas de código del notebook superaron validación de sintaxis.
  No se ha probado una sesión interactiva de Google Colab ni Windows en este entorno.
- El endpoint público de Kaggle para la versión 7 respondió HTTP 200 y contenido ZIP.
- La ejecución real con Amazon se documenta en `results/amazon_20k/RESULTADOS.md`.
  Las pruebas sintéticas no constituyen sus resultados de rendimiento.

Repetir pruebas: `python -m unittest discover -s tests -v`.
Repetir el experimento: seguir README.md y usar otra carpeta de salida.
