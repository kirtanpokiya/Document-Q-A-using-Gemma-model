# Gemma Model Document Q&A

## Overview
The Gemma Model Document Q&A project leverages advanced AI models and high-performance computing to create an intelligent document question-and-answer application. This platform allows users to upload PDFs and receive precise, contextually relevant answers to their queries.

## Features
- **Gemma-7b-it Model**: Utilizes Google's 7-billion parameter language model for accurate and contextually aware responses.
- **Groq Computing**: Ensures high performance and efficient data processing with Groq's tensor streaming processor.
- **User-Friendly Interface**: Easy-to-use UI for uploading documents and querying information.
- **Plagiarism Avoidance**: Implements mechanisms to paraphrase and ensure originality in responses.

## How It Works
1. **Document Ingestion**: Upload individual PDFs or entire folders of documents.
2. **Text Splitting**: Documents are split into manageable chunks using RecursiveCharacterTextSplitter.
3. **Vector Embedding**: Text chunks are converted into high-dimensional vectors for efficient similarity searches.
4. **Query Processing**: Input queries are processed to retrieve and generate paraphrased, contextually relevant answers.

## Technical Highlights
- **Gemma-7b-it Model**:
  - Parameters: 7 billion
  - Extensive pre-training on diverse datasets
- **Groq Computing**:
  - Unparalleled performance and scalability
- **Embeddings**:
  - GoogleGenerativeAIEmbeddings for text chunk conversion
- **Document Loaders**:
  - PyPDFLoader and PyPDFDirectoryLoader for efficient PDF parsing

## Setup Instructions

### Prerequisites
- Python 3.7+
- Streamlit
- Groq API Key
- Google API Key
