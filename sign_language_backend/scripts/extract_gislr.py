#!/usr/bin/env python3
"""Extract GISLR XYZ landmarks. See --help; no video/MediaPipe runtime needed."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from datetime import datetime, timezone
import json
import os
import getpass
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORDS = ['hello', 'bye', 'thankyou', 'please', 'yes', 'no', 'good', 'bad', 'sorry', 'help', 'water', 'eat', 'drink', 'food', 'need', 'want', 'wait', 'work', 'home', 'house', 'friend', 'see', 'look', 'go', 'today', 'tomorrow', 'where', 'who', 'how', 'man']
# 18 upper-body MediaPipe landmarks, in native index order (face through wrists).
POSE = tuple(range(18))
AUTH_HELP = """Kaggle download could not complete. Check credentials, network access and
accept the competition rules at https://www.kaggle.com/competitions/asl-signs/rules.
Configure ONE credential method (never put a real token in source code):
  export KAGGLE_API_TOKEN='<your API token>'
  # Or legacy credentials:
  export KAGGLE_USERNAME='<username>'
  export KAGGLE_KEY='<legacy API key>'
  # Or place the downloaded legacy kaggle.json in ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
Retry:
  python scripts/extract_gislr.py --download
Alternatively download and unzip using the Kaggle CLI:
  pip install kaggle
  kaggle competitions download -c asl-signs -p data/raw_shards/gislr
  unzip data/raw_shards/gislr/asl-signs.zip -d data/raw_shards/gislr
  python scripts/extract_gislr.py --dataset-dir data/raw_shards/gislr
"""


class DownloadError(RuntimeError):
    """A fatal download failure with no credential-bearing exception text."""


def credential_options():
    """Token first, then legacy env/file fallbacks. Never persist credentials."""
    options = []
    token = os.environ.get('KAGGLE_API_TOKEN', '').strip()
    if token:
        options.append(('token', token))
    username = os.environ.get('KAGGLE_USERNAME', '').strip()
    key = os.environ.get('KAGGLE_KEY', '').strip()
    if username and key:
        options.append(('legacy', (username, key)))
    config_dir = Path(os.environ.get('KAGGLE_CONFIG_DIR', str(Path.home() / '.kaggle')))
    for path in dict.fromkeys([config_dir / 'kaggle.json', Path.home() / '.kaggle/kaggle.json']):
        if not path.is_file():
            continue
        try:
            credentials = json.loads(path.read_text())
            username, key = credentials.get('username'), credentials.get('key')
            if isinstance(username, str) and isinstance(key, str) and username.strip() and key.strip():
                option = ('legacy', (username.strip(), key.strip()))
                if option not in options:
                    options.append(option)
        except (OSError, ValueError, AttributeError):
            print('Ignoring an unreadable or invalid Kaggle credentials file.', file=sys.stderr)
    return options


def download_file(relative, source):
    import kagglehub
    from kagglehub.config import set_kaggle_api_token, set_kaggle_credentials

    options = credential_options()
    # Allow native kagglehub credentials (e.g. Colab secrets) if no explicit ones.
    for kind, value in options or [('native', None)]:
        if kind == 'token':
            set_kaggle_api_token(value)
        elif kind == 'legacy':
            set_kaggle_credentials(*value)
        for attempt in range(6):
            try:
                return Path(kagglehub.competition_download(
                    'asl-signs', path=relative.as_posix(), output_dir=str(source)))
            except Exception as exc:
                # Exception strings can contain signed URLs; print only type/status.
                error_type = type(exc).__name__
                response = getattr(exc, 'response', None)
                status = getattr(response, 'status_code', None)
                if isinstance(status, int):
                    error_type += f', HTTP {status}'
                if status in (429, 500, 502, 503, 504) and attempt < 5:
                    print(f'Temporary Kaggle HTTP {status}; retry {attempt + 1}/5 in 30 seconds.', file=sys.stderr, flush=True)
                    time.sleep(30)
                    continue
                break
    guidance = ('Kaggle rate limit retries exhausted. Wait for the server limit to reset, then rerun the same command; cached files are reused.' if status == 429 else AUTH_HELP)
    raise DownloadError(f'Cannot download {relative} ({error_type}).\n{guidance}')



def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def extract_frame_features(table):
    """Use explicit frame/type/index keys, never parquet row ordering."""
    required = {'frame', 'type', 'landmark_index', 'x', 'y', 'z'}
    if not required.issubset(table.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(table.columns))}')
    if table.empty or table['frame'].isna().any():
        raise ValueError('Empty sample or invalid frame IDs')
    frames = np.sort(table['frame'].unique())
    result = np.zeros((len(frames), 60, 3), dtype=np.float32)
    for kind, indices, offset in [('pose', POSE, 0), ('left_hand', range(21), 18), ('right_hand', range(21), 39)]:
        rows = table[(table['type'] == kind) & table['landmark_index'].isin(indices)]
        if rows.duplicated(['frame', 'landmark_index']).any():
            raise ValueError(f'Duplicate {kind} landmarks')
        if rows.empty:
            continue
        coords = rows[['x', 'y', 'z']].to_numpy(dtype=np.float32, copy=True)
        # A partly invalid landmark is undetected; zero all three coordinates.
        coords[~np.isfinite(coords).all(axis=1)] = 0
        slots = {index: offset + slot for slot, index in enumerate(indices)}
        result[np.searchsorted(frames, rows['frame'].to_numpy()),
               [slots[index] for index in rows['landmark_index']], :] = coords
    if not np.any(result):
        raise ValueError('No usable pose or hand coordinates')
    return result.reshape(len(frames), 180)


def resample(features):
    """Keep short sequences and zero-pad; uniformly subsample long sequences."""
    result = np.zeros((64, 180), dtype=np.float32)
    if len(features) > 64:
        result[:] = features[np.rint(np.linspace(0, len(features) - 1, 64)).astype(int)]
    else:
        result[:len(features)] = features
    return result


def split_participants(table, val_fraction, seed):
    """Assign each signer globally, before sampling classes or reading parquet."""
    participants = sorted(table['participant_id'].unique())
    if len(participants) < 2:
        raise ValueError('At least two participants are required for a disjoint split')
    shuffled = np.random.default_rng(seed).permutation(participants)
    n_val = min(len(participants) - 1, max(1, round(len(participants) * val_fraction)))
    val_ids = set(shuffled[:n_val])
    return {participant: ('val' if participant in val_ids else 'train')
            for participant in participants}


def update_tracking(root, report):
    summary = (f"\n## GISLR participant split extraction — {report['timestamp']}\n"
               f"- Status: {report['status']}; saved {report['total']} samples.\n"
               f"- Counts by split/class: {json.dumps(report['counts'])}\n"
               f"- Split totals: {json.dumps(report['split_totals'])}; "
               f"participant overlap: {report.get('participant_overlap', 'not evaluated')}.\n"
               '- Feature order: pose 0–17, left hand 0–20, right hand 0–20; XYZ; '
               'finite float32 (64,180), uniform subsampling/trailing zero padding.\n'
               '- Metadata schema v2 contains class_to_index, path-to-index samples, '
               'sample_info, participants and split stats. Legacy consumers need adaptation.\n'
               '- Report: data/gislr_extraction_report.json. Webcam source preserved.\n')
    if report.get('error'):
        summary += f"- Blocker: {report['error']}\n"
    for name in ('context.md', 'progress.md'):
        with (root / name).open('a') as stream:
            stream.write(summary)


def run(args):
    import pandas as pd

    source = args.dataset_dir.resolve()
    source.mkdir(parents=True, exist_ok=True)
    if not 0 < args.val_fraction < 1:
        raise ValueError('--val-fraction must be strictly between 0 and 1')

    def fetch(relative):
        relative = Path(str(relative))
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Unsafe dataset path: {relative}')
        local = source / relative
        if local.is_file():
            return local
        if not args.download:
            raise FileNotFoundError(local)
        return download_file(relative, source)

    def prepare(row):
        try:
            if not str(row.sequence_id).isdecimal():
                raise ValueError('sequence_id must be numeric')
            array = resample(extract_frame_features(pd.read_parquet(fetch(row.path))))
            return row, array, None
        except (OSError, ValueError, DownloadError) as exc:
            return row, None, exc

    def prepared_rows(rows):
        iterator = iter(rows.itertuples(index=False))
        batch_size = max(1, getattr(args, 'workers', 4))
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            while batch := list(islice(iterator, batch_size)):
                yield from pool.map(prepare, batch)

    train = pd.read_csv(fetch('train.csv'), dtype={'sequence_id': str, 'participant_id': str})
    source_labels = json.loads(fetch('sign_to_prediction_index_map.json').read_text())
    if not isinstance(source_labels, dict) or not source_labels or any(
            not isinstance(key, str) or type(value) is not int or value < 0
            for key, value in source_labels.items()) or len(set(source_labels.values())) != len(source_labels):
        raise ValueError('Source sign mapping must contain unique nonnegative integer indices')
    required = {'sign', 'path', 'sequence_id', 'participant_id'}
    if not required.issubset(train.columns):
        raise ValueError(f'train.csv requires {sorted(required)} columns')
    train = train[train['sign'].isin(WORDS)].copy()
    if train.empty:
        raise ValueError('No target signs found in train.csv')
    if train[list(required)].isna().any().any() or train['participant_id'].str.strip().eq('').any():
        raise ValueError('Target rows contain missing identifiers, labels or paths')
    if train['sequence_id'].duplicated().any() or train['path'].duplicated().any():
        raise ValueError('Duplicate target sequence IDs or parquet paths in train.csv')
    inconsistent = sorted(set(train['sign']) - set(source_labels))
    if inconsistent:
        raise ValueError(f'train.csv labels absent from source mapping: {inconsistent}')
    missing = [word for word in WORDS if word not in source_labels or word not in set(train['sign'])]
    print('Unavailable target labels: ' + (', '.join(missing) or 'none'), flush=True)
    assignment = split_participants(train, args.val_fraction, args.seed)
    train['split'] = train['participant_id'].map(assignment)
    participants = {split: sorted(p for p, group in assignment.items() if group == split)
                    for split in ('train', 'val')}
    mapping, failures, provenance = {}, [], {}
    counts = {split: {word: 0 for word in WORDS} for split in ('train', 'val')}
    available = {split: {word: int(((train['sign'] == word) & (train['split'] == split)).sum())
                         for word in WORDS} for split in counts}
    # Preserve verified existing samples during an interrupted expansion. This
    # keeps metadata consistent with files and avoids reconverting known outputs.
    prior_path = args.output_root / 'metadata.json'
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    source_rows = {str(row.sequence_id): row for row in train.itertuples(index=False)}
    for relative, label in prior.get('samples', {}).items():
        info = prior['sample_info'][relative]
        row = source_rows.get(str(info['sequence_id']))
        if row is None or row.split != info['split'] or row.participant_id != info['participant_id'] or row.sign != info['sign'] or str(row.path) != info['source']:
            raise ValueError('Existing dataset split/provenance differs; use a separate output root')
        expected = Path('data/processed') / row.split / row.sign / f'{row.sequence_id}.npy'
        if relative != expected.as_posix() or label != WORDS.index(row.sign):
            raise ValueError('Existing sample mapping is inconsistent')
        array = np.load(args.output_root / relative, allow_pickle=False)
        if array.shape != (64,180) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f'Invalid existing tensor: {relative}')
        mapping[relative] = label
        provenance[relative] = info
        counts[row.split][row.sign] += 1
    existing_ids = {info['sequence_id'] for info in provenance.values()}
    fatal_error = None
    for label, word in enumerate(WORDS):
        for split in counts:
            rows = train[(train['sign'] == word) & (train['split'] == split)]
            rows = rows.sort_values('sequence_id').sample(frac=1, random_state=args.seed)
            rows = rows[~rows['sequence_id'].isin(existing_ids)]
            for row, array, error in prepared_rows(rows):
                try:
                    sequence_id = str(row.sequence_id)
                    if not sequence_id.isdecimal():
                        raise ValueError('sequence_id must be numeric')
                    if error is not None:
                        raise error
                    if array.shape != (64, 180) or array.dtype != np.float32 or not np.isfinite(array).all():
                        raise ValueError('Invalid output tensor')
                    relative = Path('data/processed') / split / word / f'{sequence_id}.npy'
                    destination = args.output_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix('.npy.tmp')
                    with temporary.open('wb') as stream:
                        np.save(stream, array, allow_pickle=False)
                    temporary.replace(destination)
                    mapping[relative.as_posix()] = label
                    provenance[relative.as_posix()] = {
                        'source': str(row.path), 'sign': word, 'split': split,
                        'participant_id': row.participant_id, 'sequence_id': sequence_id,
                        'source_class_index': source_labels[word]}
                    counts[split][word] += 1
                except DownloadError as exc:
                    fatal_error = exc
                    break
                except (OSError, ValueError) as exc:
                    failures.append({'path': str(row.path), 'error': str(exc)})
            if fatal_error:
                break
        print(f"{label:2d} {word:10s}: train={counts['train'][word]} val={counts['val'][word]}", flush=True)
        if fatal_error:
            break
    complete = not fatal_error and not failures and len(mapping) == len(train)
    report = {'timestamp': datetime.now(timezone.utc).isoformat(), 'complete': bool(complete),
              'status': 'blocked' if fatal_error else ('complete' if complete else 'incomplete'),
              'counts': counts, 'total': len(mapping), 'missing_classes': missing,
              'available_counts': available, 'split_totals': {split: sum(values.values()) for split, values in counts.items()},
              'participants': participants, 'participant_overlap': len(set(participants['train']) & set(participants['val'])),
              'shape': [64, 180], 'dtype': 'float32', 'coordinates': 'xyz',
              'pose_indices': list(POSE), 'landmark_order': 'pose, left_hand 0..20, right_hand 0..20',
              'temporal_policy': 'uniform subsample above 64; trailing zero pad below 64',
              'seed': args.seed, 'val_fraction': args.val_fraction,
              'samples_per_class': None, 'expected_total': len(train),
              'extraction_complete': bool(complete), 'vocabulary_complete': not missing, 'failures': failures}
    if fatal_error:
        report['error'] = str(fatal_error).splitlines()[0]
    labels = {word: index for index, word in enumerate(WORDS)}
    metadata = {'schema_version': 2, 'class_to_index': labels, 'samples': mapping,
                'sample_info': provenance, 'participants': participants, 'stats': report}
    write_json(args.output_root / 'metadata.json', metadata)
    write_json(args.output_root / 'data/gislr_extraction_report.json', report)
    write_json(args.output_root / 'data/gislr_labels.json', labels)
    update_tracking(args.output_root, report)
    if fatal_error:
        raise fatal_error
    return 0 if complete else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog='Install requirements-gislr.txt first. --download requires Kaggle authentication and accepted asl-signs competition rules. Missing vocabulary is reported, never substituted. All target rows are processed without per-class caps. Exit 2 means unreadable/invalid samples. metadata.json schema v2 lists only this run; existing unrelated NPY files are retained.')
    parser.add_argument('--dataset-dir', type=Path, default=ROOT / 'data/raw_shards/gislr')
    parser.add_argument('--download', action='store_true', help='Download train.csv and selected missing parquet files only')
    parser.add_argument('--output-root', type=Path, default=ROOT)
    parser.add_argument('--val-fraction', type=float, default=0.2, help='Fraction of participants assigned to validation')
    parser.add_argument('--workers', type=int, choices=range(1, 17), default=4, help='Bounded concurrent parquet reads/downloads')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prompt-token', action='store_true', help='Read a token securely for this process only')
    args = parser.parse_args()
    if args.prompt_token:
        os.environ['KAGGLE_API_TOKEN'] = getpass.getpass('Kaggle API token: ')
    try:
        return run(args)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f'Extraction failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
