import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Styling
st.set_page_config(page_title="Cloud Resume Matcher", page_icon="📄", layout="centered")

st.title("📄 Serverless Resume & Job Matcher")
st.write("A cloud-native microservice assessing resume-to-job fit instantly.")

# Layout columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Candidate Resume")
    resume_text = st.text_area(
        "Paste resume skills & experience:", 
        "Experienced in Python, Machine Learning, Data Science, SQL, Pandas, Streamlit, and Git."
    )

with col2:
    st.subheader("Job Description")
    jd_text = st.text_area(
        "Paste job description requirements:", 
        "We need a developer proficient in Python, Machine Learning, SQL, and cloud deployment."
    )

if st.button("Calculate Cloud Match Score"):
    if resume_text.strip() == "" or jd_text.strip() == "":
        st.warning("Please fill in both fields before analyzing.")
    else:
        with st.spinner("Computing semantic similarity via cloud microservice..."):
            # TF-IDF and Cosine Similarity calculation
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
            score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
            
            st.success("Analysis Complete!")
            st.metric(label="Skill Match Percentage", value=f"{score:.2f}%")
            
            if score > 75:
                st.balloons()
                st.info("Strong match! Highly suitable for the role.")
            elif score > 40:
                st.warning("Moderate match. Some core skills overlap.")
            else:
                st.error("Low match. Significant skill gap detected.")
