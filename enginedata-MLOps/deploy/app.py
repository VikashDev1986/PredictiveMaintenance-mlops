import os, json, threading
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from huggingface_hub import hf_hub_download

app = Flask(__name__)

HF_MODEL_REPO     = os.environ.get("HF_MODEL_REPO", "vikashHugFace/engine-failure-prediction-model")
_model            = None
_features         = None
_model_load_error = None

def _load_model():
    global _model, _features, _model_load_error
    try:
        model_path = hf_hub_download(
            repo_id=HF_MODEL_REPO, filename="best_model.pkl", cache_dir="/tmp/hf_cache")
        meta_path = hf_hub_download(
            repo_id=HF_MODEL_REPO, filename="metadata.json", cache_dir="/tmp/hf_cache")
        _model = joblib.load(model_path)
        with open(meta_path) as f:
            _features = json.load(f)["features"]
        print("Model loaded successfully.")
    except Exception as e:
        _model_load_error = str(e)
        print(f"Model load error: {e}")

# Load model in background — Flask starts immediately for health checks
threading.Thread(target=_load_model, daemon=True).start()

@app.route('/predict', methods=['POST'])
def predict():
    if _model is None:
        if _model_load_error:
            return jsonify({'error': f'Model failed to load: {_model_load_error}'}), 503
        return jsonify({'status': 'Model is still loading, please retry shortly'}), 503
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame(data)
        missing = [c for c in _features if c not in df.columns]
        if missing:
            return jsonify({'error': f'Missing features: {missing}'}), 400
        df = df[_features]
        preds  = _model.predict(df)
        probas = _model.predict_proba(df)[:, 1]
        return jsonify([
            {'prediction': int(p), 'probability': float(b)}
            for p, b in zip(preds, probas)
        ])
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def health_check():
    status = 'loading' if _model is None else 'ready'
    return jsonify({'status': status, 'model_repo': HF_MODEL_REPO})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)