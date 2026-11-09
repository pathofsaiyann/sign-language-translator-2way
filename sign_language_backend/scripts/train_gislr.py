"""Train a 19-class GISLR BiLSTM using the existing participant-isolated split."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
import shutil

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

try:
    from ._shared.preprocessing import normalize_features, augment_features, NORMALIZATION
except ImportError:
    from _shared.preprocessing import normalize_features, augment_features, NORMALIZATION

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


class GISLRDataset(Dataset):
    def __init__(self, records, augment=False):
        self.augment = augment
        self.items = []
        for path, label in records:
            array = np.load(path, allow_pickle=False)
            if array.shape != (64, 180) or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f'Invalid tensor: {path}')
            array = normalize_features(array)
            frames = np.flatnonzero(np.any(array != 0, axis=1))
            if not len(frames):
                raise ValueError(f'All-zero tensor: {path}')
            self.items.append((torch.from_numpy(array.copy()), label, int(frames[-1] + 1)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        x, label, length = self.items[index]
        return (augment_features(x) if self.augment else x.clone()), label, length


def load_datasets(root):
    metadata = json.loads((root / 'metadata.json').read_text())
    if metadata.get('schema_version') != 2:
        raise ValueError('Expected extraction metadata schema v2')
    source_labels = metadata['class_to_index']
    present = set(metadata['samples'].values())
    words = [word for word, index in sorted(source_labels.items(), key=lambda pair: pair[1]) if index in present]
    if len(words) != 19 or len(set(source_labels.values())) != len(source_labels):
        raise ValueError('Expected exactly 19 present classes and unique source indices')
    labels = {word: index for index, word in enumerate(words)}
    remap = {source_labels[word]: index for word, index in labels.items()}
    records = {'train': [], 'val': []}
    participants = {'train': set(), 'val': set()}
    paths = set()
    for relative, old_label in sorted(metadata['samples'].items()):
        info = metadata['sample_info'][relative]
        split, word = info['split'], info['sign']
        if split not in records or word not in labels or source_labels[word] != old_label:
            raise ValueError(f'Inconsistent metadata: {relative}')
        expected = Path('data/processed') / split / word / f"{info['sequence_id']}.npy"
        if relative != expected.as_posix():
            raise ValueError(f'Unexpected sample path: {relative}')
        path = (root / relative).resolve()
        if not path.is_relative_to((root / 'data/processed' / split).resolve()) or path in paths:
            raise ValueError('Unsafe or repeated sample path')
        paths.add(path)
        participant = str(info['participant_id']).strip()
        if not participant:
            raise ValueError('Missing participant identifier')
        participants[split].add(participant)
        records[split].append((path, remap[old_label]))
    if participants['train'] & participants['val']:
        raise ValueError('Participant leakage between train and validation')
    for split, items in records.items():
        if set(label for _, label in items) != set(range(19)):
            raise ValueError(f'Missing class coverage in {split}')
        disk_paths = {p.resolve() for p in (root / 'data/processed' / split).rglob('*.npy')}
        if disk_paths != {path for path, _ in items}:
            raise ValueError(f'{split} disk files disagree with metadata')
    datasets = {split: GISLRDataset(items, augment=split == 'train') for split, items in records.items()}
    return datasets, labels, participants


class GISLRBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(180, 128, num_layers=2, dropout=.4,
                            bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(256, 19)

    def forward(self, x, lengths=None):
        if lengths is None:
            positions = torch.arange(1, x.shape[1] + 1, device=x.device)
            lengths = (x.ne(0).any(dim=2) * positions).amax(dim=1).clamp(min=1)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(torch.cat((hidden[-2], hidden[-1]), dim=1))


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, correct, count = 0., 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y, lengths in loader:
            x, y = x.to(device), y.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x, lengths)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training/validation loss')
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += logits.argmax(1).eq(y).sum().item()
            count += len(y)
    return total_loss / count, correct / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error('epochs and threads must be positive')
    root = args.root.resolve()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    datasets, labels, participants = load_datasets(root)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {split: DataLoader(dataset, batch_size=32, shuffle=split == 'train',
                                generator=generator if split == 'train' else None)
               for split, dataset in datasets.items()}
    model = GISLRBiLSTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=5)
    criterion = nn.CrossEntropyLoss(label_smoothing=.1)
    model_dir = root / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    backup = root / 'data/training_runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup.mkdir(parents=True)
    for source in (model_dir / 'model_gislr_19class.pth', model_dir / 'labels_19class.json',
                   model_dir / 'model_gislr_19class.config.json', root / 'data/training_history.json'):
        if source.exists():
            shutil.copy2(source, backup / source.name)
    save_json(model_dir / 'labels_19class.json', labels)
    history = {key: [] for key in ('epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'learning_rate')}
    history.update(config={'input_shape': [64,180], 'hidden_size':128, 'num_layers':2,
                           'bidirectional':True, 'dropout':.4, 'batch_size':32,
                           'optimizer':'AdamW', 'lr':.001, 'weight_decay':.001, 'label_smoothing':.1,
                           'preprocessing':NORMALIZATION,
                           'augmentation':{'noise_std':.015,'frame_mask_positions':[7,9],
                                           'scale':[.9,1.1],'translation':[-.05,.05]},
                           'scheduler':{'name':'ReduceLROnPlateau','mode':'max','factor':.5,'patience':5},
                           'seed':args.seed, 'epochs':args.epochs, 'device':str(device),
                           'padding':'packed sequences; last nonzero frame defines length',
                           'metadata_sha256': hashlib.sha256((root / 'metadata.json').read_bytes()).hexdigest()},
                   previous_artifacts=str(backup.relative_to(root)), class_to_index=labels, split_counts={s:len(d) for s,d in datasets.items()},
                   participants={s:sorted(ids) for s,ids in participants.items()}, complete=False)
    best_accuracy, best_loss, best_epoch = -1., float('inf'), 0
    started = time.monotonic()
    print(f'Device={device}; train={len(datasets["train"])} val={len(datasets["val"])}; classes={len(labels)}', flush=True)
    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = run_epoch(model, loaders['train'], criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, loaders['val'], criterion, device)
        scheduler.step(val_acc)
        if val_acc > best_accuracy or (val_acc == best_accuracy and val_loss < best_loss):
            best_accuracy, best_loss, best_epoch = val_acc, val_loss, epoch
            temporary = model_dir / 'model_gislr_19class.pth.tmp'
            torch.save({key:value.detach().cpu() for key,value in model.state_dict().items()}, temporary)
            temporary.replace(model_dir / 'model_gislr_19class.pth')
            save_json(model_dir / 'model_gislr_19class.config.json',
                      {'preprocessing': NORMALIZATION, 'dropout': .4, 'class_to_index': labels,
                       'best_epoch': best_epoch})
        for key, value in zip(('epoch','train_loss','train_accuracy','val_loss','val_accuracy','learning_rate'),
                              (epoch,train_loss,train_acc,val_loss,val_acc,lr)):
            history[key].append(value)
        history.update(best_epoch=best_epoch, best_val_accuracy=best_accuracy,
                       best_val_loss=best_loss, elapsed_seconds=time.monotonic()-started,
                       complete=epoch == args.epochs)
        save_json(root / 'data/training_history.json', history)
        print(f'Epoch {epoch:02d}/{args.epochs} | Train Loss {train_loss:.4f} | Train Acc {train_acc:.2%} | Val Loss {val_loss:.4f} | Val Acc {val_acc:.2%} | LR {lr:g}', flush=True)
    # Reload the exported best state and verify it reproduces recorded metrics.
    model.load_state_dict(torch.load(model_dir / 'model_gislr_19class.pth', map_location=device, weights_only=True))
    check_loss, check_acc = run_epoch(model, loaders['val'], criterion, device)
    if abs(check_acc-best_accuracy) > 1e-8 or abs(check_loss-best_loss) > 1e-5:
        raise RuntimeError('Exported best checkpoint metrics do not match history')
    history['checkpoint_verified'] = True
    save_json(root / 'data/training_history.json', history)
    summary = (f'\n## GISLR 19-class training — {datetime.now(timezone.utc).isoformat()}\n'
               f'- Completed {args.epochs} epochs on {device}; {len(datasets["train"])} train / {len(datasets["val"])} val samples; batch size 32.\n'
               '- Remapped the 19 present metadata classes to contiguous output indices 0–18.\n'
               '- Participant-isolated extraction split retained; packed sequences exclude trailing zero padding.\n'
               '- BiLSTM: input 180, hidden 128, 2 bidirectional layers, dropout .4, linear 256 -> 19.\n'
               '- CrossEntropyLoss label smoothing .1; AdamW lr .001 / weight decay .001; ReduceLROnPlateau max/factor .5/patience 5.\n'
               f'- Best epoch {best_epoch}: val accuracy {best_accuracy:.4%}, val loss {best_loss:.6f}.\n'
               f'- Final: train accuracy {train_acc:.4%}, loss {train_loss:.6f}; val accuracy {val_acc:.4%}, loss {val_loss:.6f}.\n'
               '- Best state_dict: models/model_gislr_19class.pth; sign-to-index mapping: models/labels_19class.json.\n'
               '- History arrays/config: data/training_history.json. Best checkpoint reload/validation verified.\n'
               '- Shared shoulder XYZ normalization; training-only noise/scaling/translation/frame masking.\n'
               '- Server uses matching normalization via checkpoint config; restart after training. Live accuracy untested.\n')
    for name in ('context.md','progress.md'):
        with (root / name).open('a') as stream:
            stream.write(summary)
    print(f'Best epoch {best_epoch}, validation accuracy {best_accuracy:.2%}; checkpoint reload verified.', flush=True)


if __name__ == '__main__':
    main()
