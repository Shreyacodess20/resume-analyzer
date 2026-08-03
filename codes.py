import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag


nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')

st.set_page_config(page_title="Resume Job Match Scorer",page_icon="📄",layout="wide")
st.markdown("""
Upload your resume (PDF) and paste a job description to see how well they match! 


The application uses **TF-IDF vectorization and cosine similarity** to analyze your resume against job requirements.""")

with st.sidebar:
    st.header("About:")
    st.info("""
    This tool helps you to:
    
    - Measure how well your 
           resume matches a job description.

    - Identify important keywords
           in your resume and job description.

    - Improve your resume based
           on missing terms.
    """)

    st.header("How it works:")
    st.write("""
    1. Upload your resume in PDF format. 
    2. Paste the job description.
    3. Click **Analyze Match**.
    4. Review score and suggestion.
 """)


def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""


def clean_text(text):
    text=text.lower()
    text=re.sub(r', [^a-zA-Z]','', text)
    text = re.sub(r'\s+', ' ',text) 
    text = re.sub(r'[^\w\s]',' ',text) 
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_text = [word for word in words if word not in stop_words]
    return ' '.join(filtered_text)

def calculate_similarity(resume_text, job_description):
    resume_processed = remove_stopwords(clean_text(resume_text))
    job_processed = remove_stopwords(clean_text(job_description))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_processed, job_processed])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]*100
    return round(score, 2),resume_processed, job_processed

def extract_keywords(text, num_keywords=10):
    words = word_tokenize(text)
    words = [word for word in words if len(word)>2]
    tagged_words= pos_tag(words)
    nouns=[word for word, pos in tagged_words if pos.startswith('NN') or pos.startswith('JJ')]
    frequency=Counter(nouns)
    most_common_keywords = frequency.most_common(num_keywords)
    return most_common_keywords

def main():
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    job_description = st.text_area("Paste the job description here:", height=200)


    if st.button("Analyze Match"):
        if not uploaded_file:
            st.warning("Please upload your resume file")
            return
        if not job_description:
            st.warning("Please paste your job description")
            return

        with st.spinner("Analyzing your resume...."):
            resume_text = extract_text_from_pdf(uploaded_file)
            if not resume_text:
                st.error("Failed to extract text from the uploaded PDF. Please check the file and try again.")
                return
            
            score, resume_processed, job_processed = calculate_similarity(resume_text, job_description)

    
            st.subheader("Results:")
            st.metric("Match Score",f"{score:.2f}%")

            fig,ax= plt.subplots(figsize=(6,0.5))
            colours=["#ff4b4b","#ffa726","#0f9d58"]
            colour_index=min(int(score/33),2)
            ax.barh([0], [score], color=colours[colour_index])
            ax.set_xlim(0,100)
            ax.set_xlabel("Match Percentage")
            ax.set_yticks([])
            ax.set_title("Resume Job Match")
            st.pyplot(fig)


            if score<40:
                st.warning("Your resume has a low match score. Consider revising your resume to better align with the job description.")
            elif score<70:
                st.info("Your resume has a moderate match score. You may want to highlight relevant skills and experiences.")
            else:
                st.success("Great! Your resume has a high match score. It strongly aligns with the job description.")
    
    
if __name__=="__main__":
    main()
 


        

        


                     

        
