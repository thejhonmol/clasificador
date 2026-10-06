"""Entrenamiento, selección sin usar test y exportación del experimento."""
import argparse
import csv
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from .data import NAMES, read_sample, unique_rows, remove_overlap


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def metrics(y, predicted):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, labels=[0, 1], average='macro', zero_division=0)
    return {'accuracy': float(accuracy_score(y, predicted)),
            'precision_macro': float(precision), 'recall_macro': float(recall),
            'f1_macro': float(f1)}


def make_pipeline(model, features):
    # TF-IDF se aprende de nuevo DENTRO de cada fold, sin fuga de vocabulario.
    return Pipeline([('tfidf', TfidfVectorizer(lowercase=False, ngram_range=(1, 2),
                      max_features=features, min_df=2, sublinear_tf=True,
                      dtype=np.float32)), ('model', model)])


def run(args):
    if Path(args.train).resolve() == Path(args.test).resolve():
        raise ValueError('Entrenamiento y prueba deben ser archivos diferentes.')
    if not 0 < args.validation_size < 1:
        raise ValueError('validation-size debe estar entre 0 y 1.')
    if args.cv < 2 or args.max_features < 1 or args.jobs == 0:
        raise ValueError('cv >= 2, max-features >= 1 y jobs != 0 son obligatorios.')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'La carpeta de salida ya contiene archivos: {out}. Usa otra carpeta.')
    print('Leyendo y muestreando entrenamiento completo...', flush=True)
    train_rows, train_info = read_sample(args.train, args.train_size, args.seed)
    train_rows, train_dedup = unique_rows(train_rows)
    # Reserva validación antes de ajustar vectorizador o modelos.
    x = [r[0] for r in train_rows]
    y = np.array([r[1] for r in train_rows])
    if len(set(y)) != 2:
        raise ValueError('La muestra de entrenamiento necesita ambas clases.')
    fit_idx, val_idx = train_test_split(np.arange(len(y)), test_size=args.validation_size,
                                      stratify=y, random_state=args.seed)
    if min(Counter(y[fit_idx]).values()) < args.cv:
        raise ValueError('Muy pocos ejemplos por clase para los folds solicitados.')
    out.mkdir(parents=True, exist_ok=True)
    x_fit, x_val = [x[i] for i in fit_idx], [x[i] for i in val_idx]
    candidates = {
        'naive_bayes': (MultinomialNB(), {'model__alpha': [0.5, 1.0]}),
        'logistic_regression': (LogisticRegression(max_iter=1000, solver='liblinear',
                                  random_state=args.seed), {'model__C': [0.5, 2.0]}),
        'linear_svm': (LinearSVC(dual='auto', max_iter=5000, random_state=args.seed),
                       {'model__C': [0.5, 2.0]}),
    }
    cv = StratifiedKFold(args.cv, shuffle=True, random_state=args.seed)
    selected, selection_rows = {}, []
    for name, (model, grid) in candidates.items():
        print(f'Ajustando {name}...', flush=True)
        start = time.perf_counter()
        search = GridSearchCV(make_pipeline(model, args.max_features), grid,
                              scoring='f1_macro', cv=cv, n_jobs=args.jobs,
                              refit=True, error_score='raise')
        search.fit(x_fit, y[fit_idx])
        val_metrics = metrics(y[val_idx], search.predict(x_val))
        selected[name] = search.best_estimator_
        selection_rows.append({'model': name, 'cv_f1_macro': float(search.best_score_),
                               **val_metrics, 'best_params': search.best_params_,
                               'search_seconds': time.perf_counter() - start})
        pd.DataFrame(search.cv_results_).to_csv(out / f'cv_{name}.csv', index=False)
    winner = max(selection_rows, key=lambda row: row['f1_macro'])['model']
    # Congela la selección antes de abrir/evaluar el archivo test.
    save_json(out / 'selection.json', {'selected_model': winner, 'criterion': 'validation f1_macro',
                                       'models': selection_rows})
    print(f'Modelo seleccionado por validación: {winner}. Leyendo test...', flush=True)
    test_rows, test_info = read_sample(args.test, args.test_size, args.seed + 1)
    test_rows, test_dedup = unique_rows(test_rows)
    test_rows, overlap = remove_overlap(train_rows, test_rows)
    x_test = [r[0] for r in test_rows]
    y_test = np.array([r[1] for r in test_rows])
    if len(set(y_test)) != 2:
        raise ValueError('Test necesita ambas clases después de eliminar duplicados.')
    # Referencia trivial para interpretar las métricas.
    baseline = DummyClassifier(strategy='most_frequent').fit(np.zeros((len(y), 1)), y)
    baseline_pred = baseline.predict(np.zeros((len(y_test), 1)))
    result_rows = [{'model': 'dummy_most_frequent', **metrics(y_test, baseline_pred)}]
    reports = {'dummy_most_frequent': classification_report(y_test, baseline_pred,
                labels=[0, 1], target_names=NAMES, output_dict=True, zero_division=0)}
    predictions = pd.DataFrame({'source_line': [r[2] for r in test_rows], 'label': y_test})
    for name, estimator in selected.items():
        print(f'Reentrenando {name} y evaluando test...', flush=True)
        start = time.perf_counter()
        estimator = clone(estimator).fit(x, y)
        pred = estimator.predict(x_test)
        result_rows.append({'model': name, **metrics(y_test, pred),
                            'fit_and_predict_seconds': time.perf_counter() - start})
        reports[name] = classification_report(y_test, pred, labels=[0, 1],
                            target_names=NAMES, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        save_json(out / f'confusion_{name}.json', cm.tolist())
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=NAMES).plot(ax=ax, colorbar=False)
        ax.set_title(name)
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Etiqueta real')
        fig.tight_layout()
        fig.savefig(out / f'confusion_{name}.png', dpi=140)
        plt.close(fig)
        predictions[name] = pred
        if name == winner:
            joblib.dump(estimator, out / 'model.joblib')
            errors = predictions.loc[pred != y_test, ['source_line', 'label', name]].copy()
            # IDs en lugar de redistribuir reseñas del dataset.
            errors.head(100).to_csv(out / 'errors.csv', index=False)
    predictions.to_csv(out / 'predictions.csv', index=False)
    pd.DataFrame(result_rows).to_csv(out / 'metrics.csv', index=False)
    save_json(out / 'classification_reports.json', reports)
    with (out / 'splits.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source', 'line', 'split'])
        val_set = set(val_idx.tolist())
        for i, row in enumerate(train_rows):
            writer.writerow(['train', row[2], 'validation' if i in val_set else 'fit'])
        for row in test_rows:
            writer.writerow(['test', row[2], 'test'])
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'source_url': 'https://www.kaggle.com/datasets/bittlingmayer/amazonreviews',
                'data_kind': args.data_kind,
                'config': vars(args), 'train': train_info, 'test': test_info,
                'train_dedup': train_dedup, 'test_dedup': test_dedup,
                'test_overlap_removed': overlap,
                'sizes': {'fit': len(fit_idx), 'validation': len(val_idx),
                          'final_train': len(y), 'test': len(y_test)},
                'class_counts': {'fit': dict(Counter(map(int, y[fit_idx]))),
                                 'validation': dict(Counter(map(int, y[val_idx]))),
                                 'test': dict(Counter(map(int, y_test)))},
                'selected_model': winner,
                'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                             'numpy': np.__version__, 'pandas': pd.__version__,
                             'joblib': joblib.__version__, 'matplotlib': matplotlib.__version__},
                'command': sys.argv}
    save_json(out / 'manifest.json', manifest)
    report = ['# Resultados de ejecución', '', f'Tipo de datos declarado: **{args.data_kind}**.',
              f'Modelo seleccionado por F1 macro de validación: **{winner}**.',
              f'Train final: {len(y)}; test independiente: {len(y_test)}.',
              f'Duplicados entre muestras eliminados de test: {overlap}.', '',
              '| Modelo | Accuracy | Precision macro | Recall macro | F1 macro |',
              '|---|---:|---:|---:|---:|']
    for row in result_rows:
        report.append(f"| {row['model']} | {row['accuracy']:.4f} | {row['precision_macro']:.4f} | {row['recall_macro']:.4f} | {row['f1_macro']:.4f} |")
    report += ['', 'Resultados medidos en esta ejecución; no equivalen a evaluar todo el corpus.',
               'La comparación en test no cambia la selección previa. Consultar manifest.json y selection.json.',
               'Solo se controlan duplicados exactos tras limpieza dentro de las muestras seleccionadas;',
               'no se garantiza independencia por producto, autor ni ausencia de paráfrasis.',
               'Las etiquetas representan estrellas, no una anotación manual nueva. No hay clase neutral.',
               'No se ha validado uso específico en español ni comentarios académicos.']
    (out / 'RESULTADOS.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\n'.join(report), flush=True)
    return manifest


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--output', default='outputs/amazon')
    p.add_argument('--train-size', type=int, default=20000, help='0: todas las filas; requiere más RAM')
    p.add_argument('--test-size', type=int, default=5000, help='0: todas las filas')
    p.add_argument('--validation-size', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cv', type=int, default=3)
    p.add_argument('--max-features', type=int, default=50000)
    p.add_argument('--jobs', type=int, default=1)
    p.add_argument('--data-kind', choices=['amazon', 'synthetic-test'], default='amazon')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
