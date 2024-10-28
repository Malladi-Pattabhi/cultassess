
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2Tokenizer, GPT2Model
import torch
import numpy as np
import google.generativeai as genai

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gen-lang-client-0711936357-ec9fd589702f.json"

# Directly assign the API key
api_key = "AIzaSyBAVK69eBapQDikMaKdj7WFwBnR1-PIs6A"

# File paths
file_pathj = 'job_descriptions.xlsx'
file_pathr = 'resumes.xlsx'

# Load data
job_data = pd.read_excel(file_pathj)
candidate_data = pd.read_excel(file_pathr)

behavioral_questions = [
    "Describe a time you worked in a team.",
    "How do you handle conflict?",
    "What motivates you to work hard?",
    "Describe a challenging project you worked on.",
    "How do you prioritize tasks when under pressure?"
]

# Load the model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert job descriptions and resumes to embeddings
job_embeddings = embedding_model.encode(job_data['job_description'].tolist())
resume_embeddings = embedding_model.encode(candidate_data['Resume'].tolist())
behavioral_embeddings = embedding_model.encode(behavioral_questions)

# Calculate similarity scores and culture fit
culture_fit_scores = []
culture_pillars = {
    'Core Values': 5,
    'Behavior/Attitude': 4,
    'Mission': 4,
    'Vision': 4,
    'Practices': 3,
    'Goals': 3,
    'Work-Life Balance/Working Style': 3,
    'Diversity and Inclusion': 3,
    'Strategy': 2,
    'Policy': 2
}
for i, job_embedding in enumerate(job_embeddings):
    for j, resume_embedding in enumerate(resume_embeddings):
        culture_fit_score = cosine_similarity([job_embedding], [resume_embedding]).flatten()[0]
        question_match_scores = cosine_similarity([resume_embedding], behavioral_embeddings).flatten()

        total_score = culture_fit_score * culture_pillars['Core Values']
        for idx, question_score in enumerate(question_match_scores):
            if idx < 2:
                total_score += question_score * culture_pillars['Behavior/Attitude']
            else:
                total_score += question_score * culture_pillars['Mission']
        culture_fit_scores.append({
            'Job_ID': i,
            'Resume_ID': j,
            'Culture_Fit_Score': total_score / sum(culture_pillars.values())
        })

fit_score_df = pd.DataFrame(culture_fit_scores)

# Load GPT-2 model for embedding alternative
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2Model.from_pretrained("gpt2")

def get_gpt_embedding(text):
    """
    Generate an embedding for a given text using GPT.
    """
    inputs = tokenizer(text, return_tensors="pt")
    outputs = gpt2_model(**inputs)
    embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().detach().numpy()
    return embedding

# Using GPT-2 embeddings for culture fit scoring example
new_job_description = "We are looking for a team-oriented individual who values collaboration and has strong critical thinking skills."
new_resume_text = "Experienced professional with a passion for teamwork and problem-solving. Known for strong leadership and adaptability in fast-paced environments."

new_job_embedding = get_gpt_embedding(new_job_description)
new_resume_embedding = get_gpt_embedding(new_resume_text)
fit_score = cosine_similarity([new_job_embedding], [new_resume_embedding]).flatten()[0]

# Generating prompt-based reasoning output
reasoning_output = f"Based on the job description, the candidate aligns with team orientation and collaborative values. Fit Score: {round(fit_score * 100, 2)}."

print("Culture Fit Score:", round(fit_score * 100, 2))
print("Reasoning:", reasoning_output)

# Generative AI model configuration and example prompt usage
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Example input data
job_description = "Looking for a self-motivated, team-oriented individual with strong communication and leadership skills."
resume_text = """
Experienced team player with strong communication and leadership abilities. Proven track record in collaborative environments.
"""
behavioral_answers = ["I work well with teams by valuing each member's input.", "In conflicts, I aim for a balanced resolution that supports the team."]

# Create a prompt
prompt = f"""
Job Description: {job_description}

Resume: {resume_text}

Behavioral Questions and Answers:
1. Describe a time you worked in a team: {behavioral_answers[0]}
2. How do you handle conflict?: {behavioral_answers[1]}

Please assess the culture fit based on alignment with company culture pillars and provide a score and reasoning.
"""

# Generate content response
try:
    response = genai.generate_content(
        model="gemini-1.5-flash",
        prompt=prompt,
        generation_config=generation_config
    )
    print("Culture Fit Assessment:", response)
except Exception as e:
    print("Error during API call:", e)
