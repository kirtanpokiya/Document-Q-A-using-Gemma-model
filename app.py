import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

# Load the GROQ And OpenAI API KEY 
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #2f2f2f;
    }
    .title {
        font-family: 'Arial', sans-serif;
        color: #000;
    }
    .subtitle {
        font-family: 'Arial', sans-serif;
        color: #000;
        font-size: 20px;
    }
    .upload-section {
        background-color: #fff;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        color: #000;
    }
    .response-section {
        background-color: #fff;
        padding: 20px;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        color: #000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title'>Gemma Model Document Q&A</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='subtitle'>Upload your PDFs and ask questions based on their content.</h2>", unsafe_allow_html=True)

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions:{input}
"""
)

def vector_embedding(docs):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.docs = docs
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
uploaded_folder = st.text_input("Enter folder path to load PDFs (optional)")
st.markdown("</div>", unsafe_allow_html=True)

if st.button("Documents Embedding"):
    all_docs = []
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                loader = PyPDFLoader(tmp_file.name)
                docs = loader.load()
                all_docs.extend(docs)
                
    if uploaded_folder:
        loader = PyPDFDirectoryLoader(uploaded_folder)
        docs = loader.load()
        all_docs.extend(docs)
        
    if all_docs:
        vector_embedding(all_docs)
        st.success("Vector Store DB Is Ready")
    else:
        st.error("Please upload files or provide a folder path containing PDFs.")

prompt1 = st.text_input("Enter Your Question From Documents")

import time

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    response_time = time.process_time() - start
    
    st.markdown("<div class='response-section'>", unsafe_allow_html=True)
    st.write(f"Response time: {response_time} seconds")
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
    st.markdown("</div>", unsafe_allow_html=True)
