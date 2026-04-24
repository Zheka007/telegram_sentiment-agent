from pathlib import Path
import sys
import json
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'sentiment_model.joblib'

if not MODEL_PATH.exists():
    raise RuntimeError("Model not found. Run train.py first.")

model = joblib.load(MODEL_PATH)

if len(sys.argv) < 2:
    print(json.dumps({"error": "No text provided"}, ensure_ascii=False))
    sys.exit(1)

text = " ".join(sys.argv[1:]).strip()

if not text:
    print(json.dumps({"error": "Empty text"}, ensure_ascii=False))
    sys.exit(1)

label = model.predict([text])[0]
probs = model.predict_proba([text])[0]
confidence = float(max(probs))

result = {
    "label": str(label),
    "confidence": round(confidence, 4)
}

print(json.dumps(result, ensure_ascii=False))