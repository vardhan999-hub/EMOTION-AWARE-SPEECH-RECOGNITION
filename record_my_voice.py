# record_my_voice.py
"""
Quick microphone recording utility.

Usage:
    python record_my_voice.py
    python record_my_voice.py --duration 10 --out my_recording.wav
"""
import argparse
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

def record(duration=5, sample_rate=16000, output="my_voice.wav"):
    print(f"[record] Recording {duration}s at {sample_rate} Hz...")
    print("[record] Speak now! ▶")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    # Convert float32 to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    write(output, sample_rate, audio_int16)
    print(f"[record] ✓ Saved to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int,   default=5)
    parser.add_argument("--rate",     type=int,   default=16000)
    parser.add_argument("--out",      type=str,   default="my_voice.wav")
    args = parser.parse_args()
    record(args.duration, args.rate, args.out)
