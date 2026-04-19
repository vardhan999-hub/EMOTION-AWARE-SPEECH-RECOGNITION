# config.py
import os
import torch

# ── Directory paths ──────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
EMODB_DIR   = os.path.join(DATA_DIR, "emodb")
RAVDESS_DIR = os.path.join(DATA_DIR, "ravdess")
TESS_DIR    = os.path.join(DATA_DIR, "tess")

# ── Audio settings ───────────────────────────────────────────────
SAMPLE_RATE = 16000
N_MFCC      = 40
N_MELS      = 64
HOP_LENGTH  = 256
WIN_LENGTH  = 512

# ── Training hyperparameters ─────────────────────────────────────
BATCH_SIZE  = 32
NUM_WORKERS = 4
NUM_EPOCHS  = 50
LR          = 1e-3

# ── Device ───────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Emotion classes ──────────────────────────────────────────────
EMOTIONS = ["neutral", "angry", "sad", "fear", "happy", "disgust", "surprise"]

# ── Emotion emoji map (for UI) ───────────────────────────────────
EMOTION_EMOJI = {
    "neutral":  "😐",
    "angry":    "😠",
    "sad":      "😢",
    "fear":     "😨",
    "happy":    "😊",
    "disgust":  "🤢",
    "surprise": "😲",
}

# ── Emotion color map (for UI) ───────────────────────────────────
EMOTION_COLOR = {
    "neutral":  "#6c757d",
    "angry":    "#dc3545",
    "sad":      "#4a90d9",
    "fear":     "#6f42c1",
    "happy":    "#28a745",
    "disgust":  "#fd7e14",
    "surprise": "#ffc107",
}
