# app.py  —  Flask backend for Emotion-Aware SER
import os
import io
import subprocess
import shutil
import tempfile
import traceback
import torch
import librosa
import soundfile as sf
from flask import Flask, render_template, request, jsonify

from .config   import DEVICE, EMOTIONS, EMOTION_EMOJI, EMOTION_COLOR, N_MELS, N_MFCC
from .models   import HybridSER
from .preprocess import extract_features
from .response_generator import EmotionResponder

# ── App setup ────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024   # 32 MB

# ── Load model once at startup ───────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model.pth")

model     = HybridSER(n_mels=N_MELS, n_mfcc=N_MFCC, num_classes=len(EMOTIONS)).to(DEVICE)
responder = EmotionResponder(use_model=True)

if os.path.exists(MODEL_PATH):
    chk = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(chk["model_state"])
    print(f"[app] Model loaded from {MODEL_PATH}")
else:
    print(f"[app] WARNING: No model at {MODEL_PATH}. Using untrained weights.")

model.eval()


# ── Core prediction function ──────────────────────────────────────
def predict_emotion(file_bytes: bytes, original_filename: str = "audio.webm") -> dict:
    """
    Accepts audio in ANY browser format (webm, wav, mp3, ogg).
    Converts to clean 16kHz WAV → extracts features → runs model.
    """
    ext = os.path.splitext(original_filename)[-1].lower()
    if ext not in ['.wav', '.mp3', '.ogg', '.flac', '.webm', '.m4a']:
        ext = '.webm'

    tmp_input = None
    tmp_wav   = None

    try:
        # Step 1: Save raw bytes to temp file
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(file_bytes)
            tmp_input = f.name

        # Step 2: Convert to 16kHz mono WAV
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            tmp_wav = f.name

        converted = False

        # Try ffmpeg first (handles webm perfectly)
        if shutil.which('ffmpeg'):
            r = subprocess.run(
                ['ffmpeg', '-y', '-i', tmp_input,
                 '-ar', '16000', '-ac', '1', '-f', 'wav', tmp_wav],
                capture_output=True
            )
            converted = (r.returncode == 0)
            if not converted:
                print(f"[app] ffmpeg failed: {r.stderr.decode()}")

        # Fallback: librosa (works for wav/mp3/ogg)
        if not converted:
            y, sr = librosa.load(tmp_input, sr=16000, mono=True)
            sf.write(tmp_wav, y, 16000)

        # Step 3: Extract features
        features = extract_features(tmp_wav)
        if features is None:
            return {"error": "Feature extraction failed. Try recording again with clearer audio."}

        # Step 4: Model inference
        mel  = torch.tensor(features["log_mel"]).unsqueeze(0).unsqueeze(0).to(DEVICE)
        mfcc = torch.tensor(features["mfcc"]).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(mel, mfcc)
            probs  = torch.softmax(logits, dim=1)[0]

        pred_idx = probs.argmax().item()
        emotion  = EMOTIONS[pred_idx]
        conf     = float(probs[pred_idx]) * 100
        scores   = {EMOTIONS[i]: round(float(probs[i]) * 100, 2) for i in range(len(EMOTIONS))}

        return {
            "emotion":    emotion,
            "emoji":      EMOTION_EMOJI.get(emotion, ""),
            "color":      EMOTION_COLOR.get(emotion, "#6c757d"),
            "confidence": round(conf, 2),
            "scores":     scores,
        }

    except Exception as e:
        print(f"[app] ERROR:\n{traceback.format_exc()}")
        return {"error": str(e)}

    finally:
        for p in [tmp_input, tmp_wav]:
            if p and os.path.exists(p):
                try: os.unlink(p)
                except: pass


# ── Routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", emotions=EMOTIONS, emotion_emoji=EMOTION_EMOJI)


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received."}), 400

    audio_file = request.files["audio"]
    file_bytes = audio_file.read()
    filename   = audio_file.filename or "audio.webm"

    if len(file_bytes) == 0:
        return jsonify({"error": "Empty audio file received."}), 400

    print(f"[app] Received: {filename}  size={len(file_bytes)} bytes")

    result = predict_emotion(file_bytes, original_filename=filename)

    if "error" in result:
        return jsonify(result), 500

    result["response"] = responder.generate(result["emotion"])
    print(f"[app] Result: {result['emotion']}  conf={result['confidence']:.1f}%")
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": DEVICE, "emotions": EMOTIONS})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
