# EmoSense AI — Emotion-Aware Speech Recognition
### Hybrid CNN–BiLSTM with Intelligent Response Generation

**Final Year Project | Dept. of CSE | Seshadri Rao Gudlavalleru Engineering College**

---

## Project Structure

```
Emotion_Aware_Speech_Recognition/
│
├── ser_webapp/                  # Core package
│   ├── __init__.py
│   ├── config.py                # Hyperparameters, paths, emotion labels
│   ├── models.py                # Hybrid CNN-BiLSTM architecture
│   ├── preprocess.py            # Audio feature extraction
│   ├── response_generator.py    # DialoGPT emotion-aware responses
│   ├── app.py                   # Flask web application
│   └── templates/
│       └── index.html           # Frontend UI
│
├── dataset.py                   # PyTorch Dataset + DataLoader
├── train.py                     # Model training script
├── evaluate.py                  # Evaluation + confusion matrix
├── inference_demo.py            # CLI single-file inference
├── augment_emodb.py             # EmoDB data augmentation
├── balance_dataset.py           # Dataset class balancing
├── record_my_voice.py           # Microphone recording utility
├── run.py                       # Launch web app
├── requirements.txt
├── dataset_manifest.csv         # Auto-generated dataset index
└── checkpoints/
    └── best_model.pth           # Saved best model
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place datasets in data/ folder:
#    data/emodb/       → EmoDB .wav files
#    data/ravdess/     → RAVDESS .wav files
#    data/tess/        → TESS .wav files
```

---

## Usage

### Step 1 — Build dataset manifest
```bash
python -c "from ser_webapp.preprocess import scan_and_build_csv; scan_and_build_csv('dataset_manifest.csv')"
```

### Step 2 — (Optional) Augment & Balance
```bash
python augment_emodb.py
python balance_dataset.py
```

### Step 3 — Train
```bash
python train.py
# Optional: python train.py --epochs 100 --manifest dataset_manifest.csv
```

### Step 4 — Evaluate
```bash
python evaluate.py
```

### Step 5 — Inference (single file)
```bash
python inference_demo.py path/to/audio.wav
```

### Step 6 — Launch Web App
```bash
python run.py
# Open http://localhost:5000 in browser
```

---

## Architecture

```
Input Speech
     │
     ├─── Mel Spectrogram ──▶ CNN (3 conv layers) ──▶ TF-Attention ──▶ [256-dim]
     │                                                                       │
     └─── MFCC + Delta    ──▶ BiLSTM (2 layers)  ──▶ Attention    ──▶ [256-dim]
                                                                            │
                                                              Feature Fusion (DNN)
                                                                            │
                                                             Softmax → Emotion Class
                                                                            │
                                                      DialoGPT → Empathetic Response
```

---

## Emotion Classes
`neutral` · `angry` · `sad` · `fear` · `happy` · `disgust` · `surprise`

## Datasets
- **RAVDESS** — 24 actors, 8 emotion classes
- **EmoDB** — German emotional speech database
- **TESS** — Toronto Emotional Speech Set

---

*Built with PyTorch · Flask · Librosa · HuggingFace Transformers*
