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

Tested with models: XGBoost, BERT, Random Forest 

**Trade-offs & Limitations**

_Current Limitations:_

- Simple skill matching
- Binary location matching
- Text features (job descriptions)

_Potential Improvements:_

- Semantic skill matching using word embeddings
- Geographic distance-based scoring
- Hyperparameter tuning and ensemble methods
