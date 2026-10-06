# Resultados de ejecución

Tipo de datos declarado: **amazon**.
Modelo seleccionado por F1 macro de validación: **linear_svm**.
Train final: 20000; test independiente: 5000.
Duplicados entre muestras eliminados de test: 0.

| Modelo | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| dummy_most_frequent | 0.4958 | 0.2479 | 0.5000 | 0.3315 |
| naive_bayes | 0.8876 | 0.8877 | 0.8877 | 0.8876 |
| logistic_regression | 0.8976 | 0.8976 | 0.8976 | 0.8976 |
| linear_svm | 0.9014 | 0.9014 | 0.9014 | 0.9014 |

Resultados medidos en esta ejecución; no equivalen a evaluar todo el corpus.
La comparación en test no cambia la selección previa. Consultar manifest.json y selection.json.
Solo se controlan duplicados exactos tras limpieza dentro de las muestras seleccionadas;
no se garantiza independencia por producto, autor ni ausencia de paráfrasis.
Las etiquetas representan estrellas, no una anotación manual nueva. No hay clase neutral.
No se ha validado uso específico en español ni comentarios académicos.
