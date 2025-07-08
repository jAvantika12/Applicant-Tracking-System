from dotenv import load_dotenv
import base64
import streamlit as st
import os
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import google.generativeai as genai

# Load API key from environment variables
load_dotenv()
genai.configure(api_key = "Your_API")

# Initialize sentence transformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate response from Gemini AI model
def get_gemini_response(input_text, pdf_content, prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([input_text, pdf_content, prompt])
    return response.text

# Function to process the uploaded PDF and generate FAISS embeddings
def process_pdf_with_faiss(uploaded_file):
    if uploaded_file is not None:
        # Extract text from the PDF
        pdf_reader = PdfReader(uploaded_file)
        text = " ".join([page.extract_text() for page in pdf_reader.pages])
        
        # Chunk the text into manageable sizes
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        
        # Generate embeddings for each chunk
        embeddings = embedder.encode(chunks, convert_to_tensor=True)
        
        # Create a FAISS index for the embeddings
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.cpu().numpy())
        
        return chunks, index
    else:
        raise FileNotFoundError("No file uploaded")

# Streamlit App
st.set_page_config(page_title="Applicant Tracking System", page_icon=":Briefcase:", layout="wide", menu_items=None)

# Custom CSS for enhanced design
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}

  /* Sidebar styles */
  .sidebar-container {
    background-color: #f8f8f8;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
  }
  .sidebar-title {
    font-family: 'Arial', sans-serif;
    color: #333333;
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 20px;
  }
  .sidebar-text {
    font-family: 'Arial', sans-serif;
    color: #666666;
    font-size: 16px;
  }

  /* Footer styles */
  .footer {
    background-color: #333333;
    color: #ffffff;
    padding: 10px 0;
    width: 100%;
    text-align: center;
    font-size: 14px;
    position: fixed;
    bottom: 0;
    left: 0;
    z-index: 1;
  }

  /* Main content styles */
  .main-content {
    margin-left: 300px; /* Sidebar width */
    padding-bottom: 60px; /* Footer height */
  }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("AI Group Of Companies Presents")
    st.text("The Applicant Tracking System")
    input_text = st.text_area("Job Description: ", key="input")
    uploaded_file = st.file_uploader("Drop Your Resume Here-", type=["pdf"])
    with st.form(key='my_form'):
        submit1 = st.form_submit_button("See Results")
        submit3 = st.form_submit_button("Make A Match")

# Main content area
with st.container():
    col2 = st.columns([3])[0]
    st.markdown("""
    # Application Tracking System :
    An Applicant Tracking System (ATS) is a software application used by organizations to manage the recruitment process.
    
    # How Does an ATS Work?
    An ATS works by automating and centralizing the recruitment process, making it easier for HR professionals to handle multiple applicants, track application statuses and manage candidate communications.
    """)


if submit1 or submit3:
    if uploaded_file is not None:
        chunks, faiss_index = process_pdf_with_faiss(uploaded_file)
            
        # Generate embeddings for the job description
        job_desc_embedding = embedder.encode([input_text], convert_to_tensor=True).cpu().numpy()
            
        # Find matches using FAISS
        distances, indices = faiss_index.search(job_desc_embedding, k=5)
        relevant_chunks = [chunks[i] for i in indices[0]]
            
        # Prepare the content for analysis
        pdf_content = " ".join(relevant_chunks)
        input_prompt = """
            You are an experienced Technical Human Resource Manager. Analyze the provided resume content and job description.
            Focus on:
            1. Searchability
            2. Skills (Soft Skills and Hard Skills)
            3. Recruiter tips
            4. Formatting
            Provide insights based on the analysis.
        """ if submit1 else """
            Analyze the resume and calculate the percentage match with the job description.
            Provide:
            1. Percentage match
            2. Missing keywords
            3. Final thoughts.
            """
            
        # Generate response using Gemini AI
        response = get_gemini_response(input_text, pdf_content, input_prompt)
        st.subheader("Response:")
        st.write(response)
    else:
        st.warning("Please upload the resume to proceed.")

# Footer
st.markdown("""
<div class="footer">
    <p>&copy; Applicant Tracking System | Developed by <a href="https://github.com/imadityaim">Aditya Navakhande</p>
</div>
""", unsafe_allow_html=True)
