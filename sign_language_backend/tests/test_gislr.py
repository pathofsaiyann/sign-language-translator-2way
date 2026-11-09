import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os
from argparse import Namespace

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('gislr', Path(__file__).resolve().parents[1] / 'scripts/extract_gislr.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


class ExtractionTests(unittest.TestCase):
    def table(self):
        return pd.DataFrame([
            (20, 'right_hand', 20, .4, .5, -.6),
            (10, 'pose', 0, .1, .2, -.3),
            (10, 'left_hand', 1, np.nan, .2, .3),
            (10, 'face', 0, 9., 9., 9.),
        ], columns=['frame', 'type', 'landmark_index', 'x', 'y', 'z'])

    def test_order_missing_padding_and_sampling(self):
        features = g.extract_frame_features(self.table())
        np.testing.assert_allclose(features[0, :3], [.1, .2, -.3])
        np.testing.assert_allclose(features[1, -3:], [.4, .5, -.6])
        self.assertFalse(features[0, 54:].any())
        array = g.resample(features)
        self.assertEqual(array.shape, (64, 180))
        self.assertEqual(array.dtype, np.float32)
        self.assertFalse(array[2:].any())
        long = np.repeat(np.arange(100)[:, None], 180, axis=1)
        sampled = g.resample(long)
        self.assertEqual(sampled[0, 0], 0)
        self.assertEqual(sampled[-1, 0], 99)
        with self.assertRaises(ValueError):
            g.extract_frame_features(pd.concat([self.table(), self.table()]))

    def test_credential_fallback_and_clean_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            config = home / '.kaggle'
            config.mkdir()
            (config / 'kaggle.json').write_text(json.dumps({'username': 'file-user', 'key': 'file-key'}))
            with patch.dict(os.environ, {'KAGGLE_API_TOKEN': 'test-token', 'KAGGLE_USERNAME': 'env-user', 'KAGGLE_KEY': 'env-key', 'KAGGLE_CONFIG_DIR': str(config)}, clear=True), patch.object(g.Path, 'home', return_value=home):
                self.assertEqual(g.credential_options(), [('token', 'test-token'), ('legacy', ('env-user', 'env-key')), ('legacy', ('file-user', 'file-key'))])
                with patch('kagglehub.competition_download', side_effect=[RuntimeError('sensitive-url'), '/tmp/train.csv']) as download, patch('kagglehub.config.set_kaggle_api_token'), patch('kagglehub.config.set_kaggle_credentials') as legacy:
                    self.assertEqual(g.download_file(Path('train.csv'), home), Path('/tmp/train.csv'))
                    self.assertEqual(download.call_count, 2)
                    legacy.assert_called_once_with('env-user', 'env-key')
                with patch('kagglehub.competition_download', side_effect=RuntimeError('sensitive-url')), patch('kagglehub.config.set_kaggle_api_token'), patch('kagglehub.config.set_kaggle_credentials'):
                    with self.assertRaises(g.DownloadError) as raised:
                        g.download_file(Path('train.csv'), home)
                    self.assertNotIn('sensitive-url', str(raised.exception))
                    self.assertIn('kaggle competitions download', str(raised.exception))

    def test_rate_limit_retry(self):
        from types import SimpleNamespace
        error = RuntimeError('private-response')
        error.response = SimpleNamespace(status_code=429)
        with patch.object(g, 'credential_options', return_value=[]), patch('kagglehub.competition_download', side_effect=[error, '/tmp/train.csv']) as download, patch.object(g.time, 'sleep') as sleep:
            self.assertEqual(g.download_file(Path('train.csv'), Path('/tmp')), Path('/tmp/train.csv'))
            self.assertEqual(download.call_count, 2)
            sleep.assert_called_once_with(30)

    def fixture(self, source, words, n):
        source.mkdir()
        template = source / 'template.parquet'
        self.table().to_parquet(template)
        records = []
        for label, sign in enumerate(words):
            for index in range(n):
                sequence_id = label * 1000 + index
                path = f'{sequence_id}.parquet'
                os.link(template, source / path)
                records.append({'sign': sign, 'sequence_id': sequence_id,
                                'path': path, 'participant_id': index % 5})
        pd.DataFrame(records).to_csv(source / 'train.csv', index=False)
        (source / 'sign_to_prediction_index_map.json').write_text(json.dumps({word: i for i, word in enumerate(words)}))

    def test_all_labels_complete_and_no_participant_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'raw'
            self.fixture(source, g.WORDS, 35)
            args = Namespace(dataset_dir=source, output_root=root, download=False,
                             seed=42, val_fraction=.2)
            self.assertEqual(g.run(args), 0)
            metadata = json.loads((root / 'metadata.json').read_text())
            mapping = metadata['samples']
            self.assertEqual(len(mapping), 1050)
            train_ids = {row['participant_id'] for row in metadata['sample_info'].values() if row['split'] == 'train'}
            val_ids = {row['participant_id'] for row in metadata['sample_info'].values() if row['split'] == 'val'}
            self.assertTrue(train_ids)
            self.assertTrue(val_ids)
            self.assertFalse(train_ids & val_ids)
            for label, sign in enumerate(g.WORDS):
                self.assertEqual(sum(value == label for value in mapping.values()), 35)
                self.assertEqual(metadata['stats']['counts']['train'][sign], 28)
                self.assertEqual(metadata['stats']['counts']['val'][sign], 7)
            for path, info in metadata['sample_info'].items():
                self.assertTrue(path.startswith(f"data/processed/{info['split']}/{info['sign']}/"))
                array = np.load(root / path)
                self.assertEqual(array.shape, (64, 180))
                self.assertEqual(array.dtype, np.float32)
                self.assertTrue(np.isfinite(array).all())

    def test_parquet_end_to_end_and_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'raw'
            self.fixture(source, ['hello'], 45)
            args = Namespace(dataset_dir=source, output_root=root, download=False,
                             seed=42, val_fraction=.2)
            previous = None
            for _ in range(2):
                self.assertEqual(g.run(args), 0)
                metadata = json.loads((root / 'metadata.json').read_text())
                self.assertEqual(len(metadata['samples']), 45)
                self.assertEqual(set(metadata['samples'].values()), {0})
                self.assertEqual(len(list((root / 'data/processed').rglob('*.npy'))), 45)
                if previous:
                    self.assertEqual(previous, metadata['samples'])
                previous = metadata['samples']

    def test_single_participant_and_inconsistent_map_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'raw'
            self.fixture(source, ['hello'], 5)
            args = Namespace(dataset_dir=source, output_root=root, download=False,
                             seed=42, val_fraction=.2)
            frame = pd.read_csv(source / 'train.csv')
            frame['participant_id'] = 1
            frame.to_csv(source / 'train.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'At least two participants'):
                g.run(args)
            (source / 'sign_to_prediction_index_map.json').write_text('{"bye": 1}')
            with self.assertRaisesRegex(ValueError, 'absent from source mapping'):
                g.run(args)


if __name__ == '__main__':
    unittest.main()
