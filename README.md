# Predict Candidate-Job Fit

Build a ml model to predict whether a candidate is a good fit for a job based on structured data like skills, experience, salary expectations, and job requirements.

📂 Dataset Size: 5000 records

**Features**: skills, experience, education, salary, location, job requirements

**Target**: is_fit → 1 if the candidate is suitable for the job, else 0

# Technical Stack

- Data Processing: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- ML Models: XGBoost, BERT, Random Forest
- Explainability: SHAP
- Text Processing: Sentence Transformers
- Feature Engineering: Scikit-learn

**Data Preprocessing**

- Text normalization (lowercase)

- Null handling (certifications cleaned)

- Split comma-separated fields (skills, certs)

**Feature Engineering**

- _skill_match_ratio:_ Calculates overlap between candidate skills and required skills
  
- _experience_diff:_ Difference between candidates years of experience and minimum required

**Positive values**: Over-qualified candidates, **Negative values**: Under-qualified candidates

- _salary_within_range_: 1 if expected salary within budget.
  
- _salary_diff_: Absolute difference from median of salary range
  
- _location_match_:	1 if locations match
  
- _education_level_encoded_: Encoding of education levels- High School: 0, , Bachelor's: 1, Master's: 2, PhD: 3

**Model Training**

Tested with models: XGBoost, Logistic Regression, Random Forest 

# Results of all models-

<img width="613" height="678" alt="xgboost" src="https://github.com/user-attachments/assets/d879c265-1e91-454b-b6cd-8a94d677d3ed" />

<img width="558" height="327" alt="randomforest" src="https://github.com/user-attachments/assets/350da751-ab06-403c-a992-7b5e41e9b44e" />

<img width="552" height="460" alt="logisticR" src="https://github.com/user-attachments/assets/6501c5a9-f0a4-446d-9334-c4a6460ede97" />


**Predict Score-**

<img width="831" height="358" alt="image" src="https://github.com/user-attachments/assets/fdd0a365-f23f-42a1-8934-3d6aa0f9709f" />

**Trade-offs & Limitations**

_Current Limitations:_

- Simple skill matching
- Binary location matching
- Text features (job descriptions)

_Potential Improvements:_

- Semantic skill matching using word embeddings
- Geographic distance-based scoring
- Hyperparameter tuning and ensemble methods
