import pandas as pd
import numpy as np

def preprocess_skills(text):
    return set(map(str.strip, text.lower().split(',')))

def process_dataframe(df):
    df[['candidate_skills_set', 'required_skills_set']] = df[['candidate_skills', 'required_skills']].applymap(preprocess_skills)

    df['skill_match'] = df.apply(lambda x: len(x['candidate_skills_set'] & x['required_skills_set']) / max(len(x['required_skills_set']), 1), axis=1)
    df['experience_diff'] = df['years_experience'] - df['min_experience']
    df['salary_within_range'] = df.apply(lambda x: 1 if x['budgeted_salary_min'] <= x['expected_salary'] <= x['budgeted_salary_max'] else 0, axis=1)
    df['salary_diff'] = abs(df['expected_salary'] - (df['budgeted_salary_min'] + df['budgeted_salary_max']) / 2)
    df['location_match'] = (df['candidate_location'].str.lower() == df['job_location'].str.lower()).astype(int)

    edu_encode = {'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3}
    df['education_level_encoded'] = df['education_level'].map(edu_encode)

    df.drop(columns=['candidate_skills_set', 'required_skills_set'], inplace=True)
    
    return df

def build_feature_vector(candidate, job):
    candidate_skills = set(map(str.strip, candidate["skills"].lower().split(",")))
    required_skills = set(map(str.strip, job["skills"].lower().split(",")))

    skill_match = len(candidate_skills & required_skills) / max(len(required_skills), 1)
    experience_diff = candidate["experience"] - job["min_experience"]
    salary_within_range = 1 if job["budget_min"] <= candidate["expected_salary"] <= job["budget_max"] else 0
    salary_diff = abs(candidate["expected_salary"] - (job["budget_min"] + job["budget_max"]) / 2)
    location_match = int(candidate["location"].lower() == job["location"].lower())

    edu_encode = {'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3}
    education_level_encoded = edu_encode.get(candidate["education"], 0)

    feature_names = [
        "skill_match", "experience_diff", "salary_within_range",
        "salary_diff", "location_match", "education_level_encoded"
    ]

    feature_vector = np.array([[skill_match, experience_diff, salary_within_range,
                                salary_diff, location_match, education_level_encoded]])

    return feature_vector, feature_names