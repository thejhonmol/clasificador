"""Predice con el modelo guardado; usa únicamente archivos joblib de confianza."""
import argparse
import json
import joblib
from .data import NAMES, clean_text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--text', required=True)
    args = p.parse_args()
    text = clean_text(args.text)
    if not text:
        p.error('La reseña no puede estar vacía.')
    model = joblib.load(args.model)
    if model.named_steps['tfidf'].transform([text]).nnz == 0:
        p.error('Ninguna palabra está en el vocabulario. Prueba una reseña en inglés.')
    label = int(model.predict([text])[0])
    print(json.dumps({'sentiment': NAMES[label], 'label': label}, ensure_ascii=False))


if __name__ == '__main__':
    main()
