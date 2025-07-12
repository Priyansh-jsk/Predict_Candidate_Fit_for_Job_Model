import numpy as np
from preprocess import build_feature_vector
from embeddings import encode_text 
import shap

def predict_candidate_fit(model_name, model, explainer, candidate, job):
    """
    model_name: 'xgboost', 'random_forest', or 'logistic_regression'
    model: trained model
    explainer: SHAP explainer or None
    candidate: dict
    job: dict
    """

    if model_name == "logistic_regression":
        combined_text = f"Job: {job['description']} Candidate: {candidate['summary']}"
        emb = encode_text(combined_text)
        prob = model.predict_proba(emb)[0][1]
        return round(prob, 2), ["Text model — SHAP not supported"]

    # Structured model
    input_vector, feature_names = build_feature_vector(candidate, job)
    prob = model.predict_proba(input_vector)[0][1]

    if explainer:
        shap_values = explainer(input_vector)
        top_reasons = sorted(
            zip(feature_names, shap_values.values[0]),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
    else:
        top_reasons = ["SHAP not available"]

    return round(prob, 2), top_reasons
