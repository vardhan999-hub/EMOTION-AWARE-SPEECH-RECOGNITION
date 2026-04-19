# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from ser_webapp.preprocess import extract_features
from ser_webapp.config import EMOTIONS

emo2idx = {e: i for i, e in enumerate(EMOTIONS)}


class SERDataset(Dataset):
    """
    PyTorch Dataset for Speech Emotion Recognition.
    Reads a CSV manifest with columns: epath, emotion
    """
    def __init__(self, manifest_csv: str):
        self.df = pd.read_csv(manifest_csv)
        # Normalise emotion labels to lowercase
        self.df['emotion'] = self.df['emotion'].str.lower().str.strip()
        # Drop rows with unknown emotions
        self.df = self.df[self.df['emotion'].isin(EMOTIONS)].reset_index(drop=True)
        print(f"[dataset] Loaded {len(self.df)} samples from {manifest_csv}")
        print(f"[dataset] Class distribution:\n{self.df['emotion'].value_counts().to_string()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        path    = row['epath']
        emotion = row['emotion']
        features = extract_features(path)

        # Skip corrupt files — load next valid sample
        if features is None:
            return self.__getitem__((idx + 1) % len(self.df))

        return {
            "mel":    torch.tensor(features["log_mel"], dtype=torch.float32),
            "mfcc":   torch.tensor(features["mfcc"],    dtype=torch.float32),
            "delta":  torch.tensor(features["delta"],   dtype=torch.float32),
            "delta2": torch.tensor(features["delta2"],  dtype=torch.float32),
            "label":  emo2idx.get(emotion, 0),
            "path":   path,
        }


def pad_collate(batch):
    """
    Custom collate: zero-pads variable-length sequences along time axis.
    mel  → (B, 1, n_mels, T_max)
    mfcc → (B, n_mfcc, T_max)
    """
    max_t    = max(item['mel'].shape[1] for item in batch)
    B        = len(batch)
    n_mels   = batch[0]['mel'].shape[0]
    n_mfcc   = batch[0]['mfcc'].shape[0]

    mel_batch  = torch.zeros(B, 1, n_mels, max_t)
    mfcc_batch = torch.zeros(B, n_mfcc, max_t)
    labels     = torch.zeros(B, dtype=torch.long)
    paths      = []

    for i, item in enumerate(batch):
        t = item['mel'].shape[1]
        mel_batch[i,  0, :, :t] = item['mel']
        mfcc_batch[i,    :, :t] = item['mfcc']
        labels[i]               = item['label']
        paths.append(item['path'])

    return {
        "mel":    mel_batch,
        "mfcc":   mfcc_batch,
        "labels": labels,
        "paths":  paths,
    }
