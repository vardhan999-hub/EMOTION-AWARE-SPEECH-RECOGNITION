# augment_emodb.py
"""
Augments rare-emotion samples from EmoDB to balance the dataset.
Adds augmented files to dataset_manifest.csv.

Usage:
    python augment_emodb.py
"""
import os
import glob
import librosa
import numpy as np
import soundfile as sf
import pandas as pd

# ── Config ────────────────────────────────────────────────────────
EMODB_PATH    = "data/emodb"
AUG_OUT_DIR   = "augmented_EmoDB"
MANIFEST_FILE = "dataset_manifest.csv"
RARE_EMOTIONS = ["angry", "fear", "disgust", "sad", "surprise"]

EMODB_EMO_MAP = {
    'W': 'angry', 'L': 'neutral', 'E': 'disgust',
    'A': 'fear',  'F': 'happy',   'T': 'sad', 'N': 'neutral',
}

os.makedirs(AUG_OUT_DIR, exist_ok=True)


def augment_audio(file_path: str, out_dir: str) -> list[str]:
    """Creates 3 augmented versions of a WAV file."""
    y, sr     = librosa.load(file_path, sr=None)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    paths     = []

    # 1. Time stretch
    y_stretch  = librosa.effects.time_stretch(y, rate=1.1)
    p          = os.path.join(out_dir, f"{base_name}_stretch.wav")
    sf.write(p, y_stretch, sr); paths.append(p)

    # 2. Pitch shift (+2 semitones)
    y_pitch    = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    p          = os.path.join(out_dir, f"{base_name}_pitch.wav")
    sf.write(p, y_pitch, sr); paths.append(p)

    # 3. Add Gaussian noise
    y_noise    = y + 0.005 * np.random.randn(len(y))
    p          = os.path.join(out_dir, f"{base_name}_noise.wav")
    sf.write(p, y_noise, sr); paths.append(p)

    return paths


def main():
    if not os.path.exists(MANIFEST_FILE):
        print(f"[augment] Manifest not found: {MANIFEST_FILE}")
        return

    manifest_df = pd.read_csv(MANIFEST_FILE)
    new_rows    = []

    emo_files = glob.glob(os.path.join(EMODB_PATH, "**", "*.wav"), recursive=True)
    print(f"[augment] Found {len(emo_files)} EmoDB files")

    for f in emo_files:
        fname   = os.path.basename(f)
        if len(fname) <= 5:
            continue
        code    = fname[5].upper()
        emotion = EMODB_EMO_MAP.get(code)
        if emotion not in RARE_EMOTIONS:
            continue

        try:
            aug_files = augment_audio(f, AUG_OUT_DIR)
            for aug in aug_files:
                new_rows.append({"epath": aug, "emotion": emotion})
        except Exception as e:
            print(f"[augment] Skipping {f}: {e}")

    if new_rows:
        new_df   = pd.DataFrame(new_rows)
        updated  = pd.concat([manifest_df, new_df], ignore_index=True)
        updated.to_csv(MANIFEST_FILE, index=False)
        print(f"[augment] Added {len(new_rows)} augmented samples to {MANIFEST_FILE}")
    else:
        print("[augment] No files augmented.")


if __name__ == "__main__":
    main()
