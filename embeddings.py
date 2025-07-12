from sentence_transformers import SentenceTransformer
import numpy as np

# model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')


def build_candidate_summary(row):
    return (
        f"Candidate has {row['years_experience']} years of experience, "
        f"has held job titles such as {row['past_job_titles']}, "
        f"has skills: {row['candidate_skills']}, "
        f"expects a salary of {row['expected_salary']}, "
        f"is located in {row['candidate_location']}, "
        f"and has an education level of {row['education_level']}."
    )

def add_embeddings(df):
    df['candidate_summary'] = df.apply(build_candidate_summary, axis=1)
    df['combined_text'] = df.apply(lambda row: f"Job: {row['job_description']} Candidate: {row['candidate_summary']}", axis=1)

    embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)
    return np.array(embeddings)

def encode_text(text):
    return model.encode([text])