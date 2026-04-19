# inference_demo.py
"""
Run single-file emotion inference from command line.

Usage:
    python inference_demo.py path/to/audio.wav
    python inference_demo.py path/to/audio.wav --model checkpoints/best_model.pth
"""
import os
import sys
import argparse
import torch

from ser_webapp.models import HybridSER
from ser_webapp.preprocess import extract_features
from ser_webapp.config import DEVICE, EMOTIONS, EMOTION_EMOJI, N_MELS, N_MFCC


def load_model(path="checkpoints/best_model.pth"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}. Run train.py first.")
    chk   = torch.load(path, map_location=DEVICE)
    model = HybridSER(n_mels=N_MELS, n_mfcc=N_MFCC, num_classes=len(EMOTIONS)).to(DEVICE)
    model.load_state_dict(chk["model_state"])
    model.eval()
    return model


def predict(audio_path: str, model_path="checkpoints/best_model.pth"):
    if not os.path.exists(audio_path):
        print(f"[inference] File not found: {audio_path}")
        sys.exit(1)

    model    = load_model(model_path)
    features = extract_features(audio_path)

    if features is None:
        print("[inference] Feature extraction failed.")
        sys.exit(1)

    mel  = torch.tensor(features["log_mel"]).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,H,T)
    mfcc = torch.tensor(features["mfcc"]).unsqueeze(0).to(DEVICE)                  # (1,40,T)

    with torch.no_grad():
        logits = model(mel, mfcc)
        probs  = torch.softmax(logits, dim=1)[0]

    pred_idx  = probs.argmax().item()
    emotion   = EMOTIONS[pred_idx]
    emoji     = EMOTION_EMOJI.get(emotion, "")
    confidence = float(probs[pred_idx]) * 100

    print("\n" + "─" * 48)
    print(f"  File      : {os.path.basename(audio_path)}")
    print(f"  Emotion   : {emoji}  {emotion.upper()}")
    print(f"  Confidence: {confidence:.2f}%")
    print("─" * 48)
    print("  All probabilities:")
    for i, e in enumerate(EMOTIONS):
        bar = '█' * int(float(probs[i]) * 30)
        print(f"    {e:<10}  {bar:<30}  {float(probs[i])*100:5.1f}%")
    print("─" * 48 + "\n")

    return emotion, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio",   help="Path to audio file (.wav, .mp3, etc.)")
    parser.add_argument("--model", default="checkpoints/best_model.pth")
    args = parser.parse_args()
    predict(args.audio, args.model)
