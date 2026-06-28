import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# Configuration from model registration (HF_MODEL_REPO and MODEL_PATH are global)
# HF_MODEL_REPO is from cell GwehjAB2mMTE
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "vikashHugFace/engine-failure-prediction-model")
MODEL_FILE    = "best_model.pkl"

# Download model from Hugging Face Hub (or load if already present locally)
model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=MODEL_FILE, cache_dir="/tmp/hf_cache")
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame(data)

        # Retrieve features used during training. This assumes 'metadata' was saved and contains 'features'.
        # For this example, we'll hardcode them based on the earlier EDA and training steps.
        # In a real-world scenario, load from a metadata.json or similar.
        expected_features = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']

        # Ensure columns are in the correct order as during training
        df = df[expected_features] # Reorder columns if necessary

        predictions = model.predict(df)
        # For classification, get probabilities for the positive class (1)
        probabilities = model.predict_proba(df)[:, 1]

        results = [
            {'prediction': int(pred), 'probability': float(prob)}
            for pred, prob in zip(predictions, probabilities)
        ]

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def health_check():
    return 'Model API is running!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)