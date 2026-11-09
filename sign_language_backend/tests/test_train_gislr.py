import importlib.util
from pathlib import Path
import unittest
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "scripts"))
import torch

spec = importlib.util.spec_from_file_location('train_gislr', Path(__file__).resolve().parents[1] / 'scripts/train_gislr.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


class TrainingTests(unittest.TestCase):
    def test_real_metadata_remapping_and_split(self):
        datasets, labels, participants = g.load_datasets(g.ROOT)
        self.assertEqual(len(labels),19)
        self.assertEqual(set(labels.values()),set(range(19)))
        self.assertEqual(labels['bad'],6)
        self.assertEqual(labels['man'],18)
        self.assertGreaterEqual(len(datasets['train']),608)
        self.assertGreaterEqual(len(datasets['val']),152)
        self.assertFalse(participants['train'] & participants['val'])
        for dataset in datasets.values():
            self.assertEqual(set(item[1] for item in dataset.items),set(range(19)))

    def test_shoulder_normalization_missing_and_invariance(self):
        raw = np.zeros((64,60,3),dtype=np.float32)
        raw[:20,11]=[1,2,3];raw[:20,12]=[3,2,3];raw[:20,18]=[5,4,3]
        norm=g.normalize_features(raw.reshape(64,180)).reshape(64,60,3)
        np.testing.assert_allclose(norm[0,11],[-.5,0,0])
        np.testing.assert_allclose(norm[0,12],[.5,0,0])
        np.testing.assert_allclose(norm[0,18],[1.5,1,0])
        transformed=raw.copy();active=np.any(raw!=0,axis=-1)
        transformed[active]=raw[active]*1.1+np.array([2,3,4])
        np.testing.assert_allclose(g.normalize_features(transformed.reshape(64,180)),norm.reshape(64,180),atol=1e-6)
        self.assertFalse(norm[20:].any());self.assertFalse(norm[:,0].any())
        raw[0,12]=raw[0,11];raw[1,11]=0;raw[2,11]=np.nan
        invalid=g.normalize_features(raw.reshape(64,180))
        self.assertTrue(np.isfinite(invalid).all());self.assertFalse(invalid[:3].any())

    def test_training_augmentation_and_validation_determinism(self):
        datasets,_,_=g.load_datasets(g.ROOT)
        train,val=datasets['train'],datasets['val']
        raw=train.items[0][0].clone()
        a,_,length=train[0];b,_,_=train[0]
        self.assertFalse(torch.equal(a,b));torch.testing.assert_close(train.items[0][0],raw)
        missing=raw.reshape(64,60,3).eq(0).all(-1)
        self.assertFalse(a.reshape(64,60,3)[missing].any())
        self.assertTrue(torch.isfinite(a).all())
        torch.testing.assert_close(val[0][0],val[0][0])
        dense=torch.ones(64,180)
        masked=g.augment_features(dense).eq(0).all(-1).sum().item()
        self.assertTrue(7<=masked<=9)

    def test_padding_and_backward(self):
        torch.set_num_threads(2)
        model = g.GISLRBiLSTM().eval()
        x = torch.randn(2,64,180)
        x[0,10:] = 0
        lengths = torch.tensor([10,64])
        logits = model(x,lengths)
        self.assertEqual(logits.shape,(2,19))
        changed = x.clone()
        changed[0,10:] = 100
        torch.testing.assert_close(model(changed,lengths),logits)
        torch.testing.assert_close(model(x),logits)
        loss = torch.nn.CrossEntropyLoss()(logits,torch.tensor([0,18]))
        loss.backward()
        self.assertTrue(torch.isfinite(model.classifier.weight.grad).all())


if __name__ == '__main__':
    unittest.main()
