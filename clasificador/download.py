"""Descarga pública de la versión 7; alternativa: descarga manual desde Kaggle."""
import argparse
import json
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

URL = 'https://www.kaggle.com/api/v1/datasets/download/bittlingmayer/amazonreviews?datasetVersionNumber=7'
FILES = ('train.ft.txt.bz2', 'test.ft.txt.bz2')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='data')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in FILES):
        p.error('Ya existen archivos de datos. Usa otra carpeta o la descarga existente.')
    print('Descargando ~517 MB desde Kaggle (requiere espacio adicional temporal)...', flush=True)
    try:
        with tempfile.TemporaryDirectory(dir=out) as temp:
            temp = Path(temp)
            archive = temp / 'amazonreviews.zip'
            request = urllib.request.Request(URL, headers={'User-Agent': 'amazon-sentiment-academic/1.0'})
            with urllib.request.urlopen(request, timeout=60) as response, archive.open('wb') as target:
                shutil.copyfileobj(response, target)
            with zipfile.ZipFile(archive) as z:
                for name in FILES:
                    # Solo nombres conocidos, sin extractall ni rutas del archivo remoto.
                    with z.open(name) as source, (temp / name).open('wb') as target:
                        shutil.copyfileobj(source, target)
            for name in FILES:
                (temp / name).replace(out / name)
    except (urllib.error.URLError, zipfile.BadZipFile, KeyError, OSError) as exc:
        p.exit(1, f'No se pudo completar la descarga: {exc}\nDescarga y extrae manualmente desde https://www.kaggle.com/datasets/bittlingmayer/amazonreviews\n')
    (out / 'source.json').write_text(json.dumps({'dataset': 'bittlingmayer/amazonreviews',
        'version': 7, 'url': URL}, indent=2), encoding='utf-8')
    print(f'Listo: {out.resolve()}')


if __name__ == '__main__':
    main()
