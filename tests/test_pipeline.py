"""Pruebas sintéticas; NO son resultados de Amazon."""
import bz2
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import joblib
import pandas as pd
from clasificador.data import clean_text, parse_line, read_sample, unique_rows, remove_overlap
from clasificador.train import parser, run


class DataTests(unittest.TestCase):
    def test_label_mapping_and_title(self):
        self.assertEqual(parse_line('__label__1 Bad: does not work', 5), ('bad: does not work', 0, 5))
        self.assertEqual(parse_line('__label__2 Great: works well', 1)[1], 1)

    def test_reject_invalid(self):
        for line in ['__label__3 neutral', '__label__1', 'text only', '__label__1 <br>']:
            with self.assertRaises(ValueError):
                parse_line(line, 1)

    def test_preserves_negation(self):
        self.assertEqual(clean_text("<p>NOT good &amp; I don't like it</p>"), "not good & i don't like it")

    def test_sampling_and_bz2_equivalence(self):
        with tempfile.TemporaryDirectory() as folder:
            plain = Path(folder) / 'train.txt'
            compressed = Path(folder) / 'train.txt.bz2'
            content = ''.join(f'__label__{i % 2 + 1} review number {i}\n' for i in range(100))
            plain.write_text(content, encoding='utf-8')
            compressed.write_bytes(bz2.compress(content.encode()))
            a, stats = read_sample(plain, 10, 42)
            b, _ = read_sample(compressed, 10, 42)
            self.assertEqual(a, b)
            self.assertEqual(stats['total_rows'], 100)
            self.assertEqual(len(a), 10)
            self.assertGreater(max(r[2] for r in a), 10)
            self.assertEqual(len(read_sample(plain, 0, 42)[0]), 100)

    def test_conflicts_and_overlap(self):
        rows = [('good', 1, 1), ('good', 1, 2), ('bad', 0, 3), ('bad', 1, 4)]
        unique, stats = unique_rows(rows)
        self.assertEqual(unique, [('good', 1, 1)])
        self.assertEqual(stats['conflicting_texts'], 1)
        test, count = remove_overlap(unique, [('good', 1, 7), ('okay', 1, 8)])
        self.assertEqual(count, 1)
        self.assertEqual(test, [('okay', 1, 8)])

    def test_empty_rows_are_counted_and_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'train.txt'
            path.write_text('__label__1 <br>\n__label__2 good product\n__label__1 x\n', encoding='utf-8')
            rows, stats = read_sample(path, 10, 42)
            self.assertEqual(rows, [('good product', 1, 2)])
            self.assertEqual(stats['total_rows'], 3)
            self.assertEqual(stats['eligible_rows'], 1)
            self.assertEqual(stats['skipped_empty_rows'], 2)
            self.assertEqual(stats['source_class_counts'], {0: 2, 1: 1})

    def test_end_to_end_saved_model_and_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, start, stop in [('train', 0, 80), ('test', 80, 100)]:
                (root / f'{name}.txt').write_text(''.join(
                    f"__label__{i % 2 + 1} {'excellent great useful' if i % 2 else 'terrible broken bad'} product item{i}\n"
                    for i in range(start, stop)), encoding='utf-8')
            out = root / 'results'
            args = parser().parse_args(['--train', str(root/'train.txt'), '--test', str(root/'test.txt'),
                                       '--output', str(out), '--train-size', '0', '--test-size', '0',
                                       '--cv', '2', '--max-features', '100', '--data-kind', 'synthetic-test'])
            manifest = run(args)
            self.assertEqual(manifest['sizes']['test'], 20)
            self.assertEqual(manifest['data_kind'], 'synthetic-test')
            model = joblib.load(out/'model.joblib')
            self.assertEqual(list(model.predict(['excellent great useful', 'terrible broken bad'])), [1, 0])
            self.assertNotIn('item80', model.named_steps['tfidf'].vocabulary_)
            splits = pd.read_csv(out/'splits.csv')
            fit = set(splits.loc[splits.split == 'fit', 'line'])
            val = set(splits.loc[splits.split == 'validation', 'line'])
            self.assertFalse(fit & val)
            self.assertEqual(len(fit | val), 80)
            result = subprocess.run([sys.executable, '-m', 'clasificador.predict', '--model',
                                     str(out/'model.joblib'), '--text', 'excellent great useful'],
                                    check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout)['sentiment'], 'positivo')
            with self.assertRaises(ValueError):
                run(args)


if __name__ == '__main__':
    unittest.main()
