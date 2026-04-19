# preprocess.py
import os
import librosa
import numpy as np
import pandas as pd
from .config import SAMPLE_RATE, N_MELS, N_MFCC, HOP_LENGTH, WIN_LENGTH


def extract_features(file_path, sr=SAMPLE_RATE, n_mels=N_MELS, n_mfcc=N_MFCC):
    """
    Extract Mel spectrogram + MFCC + delta features from a WAV file.
    Returns a dict of float32 numpy arrays, or None on failure.
    """
    try:
        y, orig_sr = librosa.load(file_path, sr=None, mono=True)

        # Resample if needed
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

        # Trim leading/trailing silence
        y, _ = librosa.effects.trim(y, top_db=20)

        # Ensure minimum length (0.5 s)
        min_len = sr // 2
        if len(y) < min_len:
            y = np.pad(y, (0, min_len - len(y)))

        # ── Mel spectrogram ──────────────────────────────────────
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_mels=n_mels,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            fmin=80, fmax=8000
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)  # (n_mels, T)

        # ── MFCC + deltas ────────────────────────────────────────
        mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                        hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
        delta   = librosa.feature.delta(mfcc)
        delta2  = librosa.feature.delta(mfcc, order=2)

        # Normalize each feature
        def normalize(arr):
            mu  = arr.mean()
            std = arr.std() + 1e-9
            return ((arr - mu) / std).astype(np.float32)

        return {
            "log_mel": normalize(log_mel),
            "mfcc":    normalize(mfcc),
            "delta":   normalize(delta),
            "delta2":  normalize(delta2),
        }

    except Exception as e:
        print(f"[preprocess] Error processing {file_path}: {e}")
        return None


def scan_and_build_csv(csv_path):
    """
    Scan EMODB, RAVDESS, and TESS directories and build a unified CSV:
    Columns: epath, emotion
    """
    from .config import EMODB_DIR, RAVDESS_DIR, TESS_DIR

    rows = []

    # ── 1. EMODB ─────────────────────────────────────────────────
    emodb_map = {
        'W': 'angry',
        'L': 'neutral',   # boredom → neutral
        'E': 'disgust',
        'A': 'fear',
        'F': 'happy',
        'T': 'sad',
        'N': 'neutral',
    }
    for root, _, files in os.walk(EMODB_DIR):
        for f in files:
            if f.lower().endswith(".wav") and len(f) > 5:
                code    = f[5].upper()
                emotion = emodb_map.get(code, "neutral")
                rows.append({"epath": os.path.join(root, f), "emotion": emotion})

    # ── 2. RAVDESS ───────────────────────────────────────────────
    ravdess_map = {
        1: "neutral", 2: "neutral",
        3: "happy",   4: "sad",
        5: "angry",   6: "fear",
        7: "disgust", 8: "surprise",
    }
    for root, _, files in os.walk(RAVDESS_DIR):
        for f in files:
            if f.lower().endswith(".wav"):
                parts = f.split("-")
                try:
                    emotion = ravdess_map.get(int(parts[2]), "neutral")
                except (IndexError, ValueError):
                    emotion = "neutral"
                rows.append({"epath": os.path.join(root, f), "emotion": emotion})

    # ── 3. TESS ──────────────────────────────────────────────────
    tess_map = {
        "angry": "angry", "disgust": "disgust",
        "fear":  "fear",  "happy":   "happy",
        "ps":    "surprise", "sad":  "sad",
        "neutral": "neutral",
    }
    for root, _, files in os.walk(TESS_DIR):
        folder = os.path.basename(root).lower()
        emotion = next((v for k, v in tess_map.items() if k in folder), "neutral")
        for f in files:
            if f.lower().endswith(".wav"):
                rows.append({"epath": os.path.join(root, f), "emotion": emotion})

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"[preprocess] CSV saved → {csv_path}  ({len(df)} samples)")
    return df
