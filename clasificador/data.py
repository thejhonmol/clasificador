"""Lectura fastText, muestreo reproducible y control de duplicados."""
import bz2
import hashlib
import html
import random
import re
from collections import Counter
from pathlib import Path

LABELS = {'__label__1': 0, '__label__2': 1}
NAMES = ['negativo', 'positivo']


class EmptyTextError(ValueError):
    """Etiqueta válida, pero ningún token utilizable tras limpiar."""


def clean_text(text):
    """Conserva palabras de negación; no aplica traducción ni stemming."""
    text = html.unescape(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' url ', text)
    return ' '.join(text.split())


def parse_line(line, number):
    parts = line.strip().split(maxsplit=1)
    if not parts or parts[0] not in LABELS:
        raise ValueError(f'Línea {number}: se espera __label__1 o __label__2 y texto.')
    text = clean_text(parts[1]) if len(parts) == 2 else ''
    if not re.search(r'(?u)\b\w\w+\b', text):
        raise EmptyTextError(f'Línea {number}: texto vacío o sin tokens utilizables.')
    return text, LABELS[parts[0]], number


def read_sample(path, size, seed):
    """Reservoir sampling uniforme sobre TODO el archivo, sin cargarlo completo.

    size=0 conserva todas las filas. SHA256 identifica los bytes originales.
    Los identificadores de fila (base 1) permiten reproducir cada selección.
    """
    path = Path(path)
    if size < 0:
        raise ValueError('El tamaño de muestra no puede ser negativo.')
    rng = random.Random(seed)
    rows, counts = [], Counter()
    opener = bz2.open if path.suffix == '.bz2' else open
    total = eligible = skipped_empty = 0
    with opener(path, 'rt', encoding='utf-8', errors='strict') as stream:
        for total, line in enumerate(stream, 1):
            # Valida todas las etiquetas, incluso fuera de la muestra.
            try:
                row = parse_line(line, total)
            except EmptyTextError:
                counts[LABELS[line.strip().split(maxsplit=1)[0]]] += 1
                skipped_empty += 1
                continue
            counts[row[1]] += 1
            eligible += 1
            if not size or len(rows) < size:
                rows.append(row)
            else:
                index = rng.randrange(eligible)
                if index < size:
                    rows[index] = row
    if not rows:
        raise ValueError(f'Archivo vacío: {path}')
    rows.sort(key=lambda row: row[2])
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return rows, {'file': path.name, 'sha256': digest, 'total_rows': total,
                  'source_class_counts': dict(counts), 'sampled_rows': len(rows),
                  'eligible_rows': eligible, 'skipped_empty_rows': skipped_empty,
                  'requested_rows': size, 'seed': seed, 'sampling': 'uniform reservoir over usable rows'}


def unique_rows(rows):
    """Elimina repeticiones; descarta todo texto con etiquetas contradictorias."""
    first, conflicts = {}, set()
    for row in rows:
        if row[0] in first and first[row[0]][1] != row[1]:
            conflicts.add(row[0])
        first.setdefault(row[0], row)
    kept = [row for text, row in first.items() if text not in conflicts]
    return kept, {'removed_rows': len(rows) - len(kept),
                  'conflicting_texts': len(conflicts)}


def remove_overlap(train_rows, test_rows):
    seen = {row[0] for row in train_rows}
    kept = [row for row in test_rows if row[0] not in seen]
    return kept, len(test_rows) - len(kept)
