import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("AI-Powered Resume Analyzer 🧠")
st.markdown("Upload your resume and job description to check match % and skills!")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text.lower()

def clean_text(text):
    # Remove non-alphabetic characters and extra spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

resume_file = st.file_uploader("📄 Upload your Resume (PDF)", type=["pdf"])
job_description = st.text_area("📝 Paste the Job Description here", height=300)

if st.button("Analyze"):
    if resume_file is None or job_description.strip() == "":
        st.warning("⚠️ Please upload a resume and paste a job description.")
    else:
        # Extract and clean resume text
        resume_text = extract_text_from_pdf(resume_file)
        resume_text = clean_text(resume_text)

        # Clean job description text
        jd_text = clean_text(job_description)

        # Vectorize and calculate similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform([resume_text, jd_text])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

        match_percentage = round(similarity * 100, 2)
        st.success(f"✅ Match Percentage: {match_percentage}%")

        if match_percentage > 50:
            st.balloons()
        else:
            st.info("Try improving your resume or tailoring it better to the job description.")
