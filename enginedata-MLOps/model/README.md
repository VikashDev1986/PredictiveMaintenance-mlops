---
    tags:
      - classification
      - engine-failure-prediction
      - sklearn
    metrics:
      - accuracy
      - precision
      - recall
      - f1
      - roc_auc
    ---

    # Engine Failure Prediction Model

    **Algorithm:** Decision Tree
    **Task:** Binary Classification — predict `Engine Condition`

    ## Performance (held-out test set)

    | Metric    | Value  |
    |-----------|--------|
    | Accuracy  | 0.6404 |
    | Precision | 0.6871 |
    | Recall    | 0.7889 |
    | F1-Score  | 0.7345 |
    | ROC AUC   | 0.6687 |

    ## Best Hyperparameters

    ```json
    {
  "model__max_depth": 3,
  "model__min_samples_leaf": 1,
  "model__min_samples_split": 2
}
    ```

    ## Usage

    ```python
    import joblib
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id="vikashHugFace/engine-failure-prediction-model", filename="best_model.pkl")
    model = joblib.load(path)
    predictions = model.predict(X_new)
    ```