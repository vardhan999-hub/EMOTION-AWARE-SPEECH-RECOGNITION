# run.py  —  Launch the EmoSense AI web application
"""
Usage:
    python run.py
    python run.py --port 5000 --debug
"""
import argparse
import sys
import os

# Make sure ser_webapp is importable from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ser_webapp.app import app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EmoSense AI Web App")
    parser.add_argument("--port",  type=int,  default=5000)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--host",  type=str,  default="0.0.0.0")
    args = parser.parse_args()

    print("\n" + "═" * 52)
    print("  🎙  EmoSense AI — Emotion-Aware SER System")
    print("═" * 52)
    print(f"  ▶  Running at  http://localhost:{args.port}")
    print(f"  ▶  Debug mode: {args.debug}")
    print("  ▶  Press Ctrl+C to stop")
    print("═" * 52 + "\n")

    app.run(host=args.host, port=args.port, debug=args.debug)
