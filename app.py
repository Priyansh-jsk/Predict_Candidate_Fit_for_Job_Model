import streamlit as st
import pandas as pd
from preprocess import process_dataframe
from predictor import predict_candidate_fit
from embeddings import add_embeddings
from model_utils import train_xgboost, train_random_forest, train_logistic_regression
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sentence_transformers import SentenceTransformer


# st.set_page_config(layout="wide")
st.title("Predict Candidate Fit for a Job Role")

model_text = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')


uploaded_file = st.file_uploader("Upload Candidate Dataset", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = process_dataframe(df)

    features = ['skill_match', 'experience_diff', 'salary_within_range', 'salary_diff', 'location_match', 'education_level_encoded']
    X = df[features]
    y = df['is_fit']

    st.subheader("📊 Dataset Overview")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("### Class Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='is_fit', data=df, ax=ax1)
    st.pyplot(fig1)

    st.markdown("### Education Level")
    fig2, ax2 = plt.subplots()
    df['education_level'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax2)
    ax2.set_ylabel('')
    st.pyplot(fig2)

    st.markdown("### 💰 Salary Distribution by Fit")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df, x='is_fit', y='expected_salary', ax=ax3)
    st.pyplot(fig3)

    st.markdown("---")
    st.subheader("Choose Model to Train")

    model_type = st.selectbox("Choose Model", ["XGBoost", "Random Forest", "Logistic Regression"])

    if model_type == "XGBoost":
        model, X_test, y_test = train_xgboost(X, y)
        preds = model.predict(X_test)
        st.text("XGBoost Classification Report")
        st.code(classification_report(y_test, preds))

        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        st.pyplot(fig)

    elif model_type == "Random Forest":
        model, X_test, y_test = train_random_forest(X, y)
        preds = model.predict(X_test)
        st.text("Random Forest Classification Report")
        st.code(classification_report(y_test, preds))

    elif model_type == "Logistic Regression":
        X_emb = add_embeddings(df)
        model, X_test, y_test = train_logistic_regression(X_emb, y)
        preds = model.predict(X_test)
        st.text("Logistic Regression Classification Report")
        st.code(classification_report(y_test, preds))
    

    st.markdown("---")
    st.subheader("📥 Predict Single Candidate Fit")

    with st.expander("Input Candidate and Job Details"):

        c_name = st.text_input("Candidate Name")
        c_skills = st.text_input("Candidate Skills (comma separated)")
        c_exp = st.number_input("Years of Experience", 0, 50)
        c_edu = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
        c_location = st.text_input("Candidate Location")
        c_salary = st.number_input("Expected Salary")

        j_title = st.text_input("Job Title")
        j_skills = st.text_input("Required Skills (comma separated)")
        j_desc = st.text_area("Job Description")
        j_exp = st.number_input("Min Experience Required", 0, 50)
        j_loc = st.text_input("Job Location")
        j_salary_min = st.number_input("Budgeted Salary Min")
        j_salary_max = st.number_input("Budgeted Salary Max")

    if st.button("🔍 Predict Fit Score"):
        candidate = {
            "skills": c_skills,
            "experience": c_exp,
            "expected_salary": c_salary,
            "location": c_location,
            "education": c_edu,
            "summary": f"{c_name} with {c_exp} years experience in {c_skills}"
        }

        job = {
            "skills": j_skills,
            "min_experience": j_exp,
            "budget_min": j_salary_min,
            "budget_max": j_salary_max,
            "location": j_loc,
            "description": j_desc
        }

        # Reuse trained model and SHAP from above
        model_key = model_type.lower().replace(" ", "_")
        if model_type == "XGBoost":
            explainer_model = shap.Explainer(model)
        elif model_type == "Random Forest":
            explainer_model = shap.Explainer(model)
        else:
            explainer_model = None

        score, reasons = predict_candidate_fit(model_key, model, explainer_model, candidate, job)

        st.success(f"✅ Predicted Fit Score: {score}")
        st.markdown("#### 🔍 Top Reasons")
        for r in reasons:
            if isinstance(r, str):
                st.markdown(f"- {r}")
            else:
                st.markdown(f"- **{r[0]}**: {round(r[1], 3)}")