# RAG LLM with Google Cloud Platform (BigQuery, VertexAI & Gemini)
Author: Marco Pellegrino<br>
Year: 2024

## Overview
This notebook demonstrates a RAG (Retrieval-augmented Generation) LLM (Large Language Model) system for processing documents stored in Google Cloud Storage (GCS), generating embeddings, and storing them in BigQuery Vector Store for efficient similarity search. The system then uses Langchain for question answering (QA) based on the embedded document. Both a notebook and a Streamlit app are provided.
  
### Environment Configuration
Ensure your environment variables and configurations (`config.py`) are correctly set up for:
- Google Cloud project (`PROJECT_ID`)
- Google Cloud Storage bucket (`GCS_BUCKET`)
- BigQuery Vector Store dataset (`DATASET`)
- BigQuery Vector Store table (`TABLE`)

## Workflow
1. **Retrieve Document from Cloud Storage**: Copies a PDF document from GCS to a local directory (`data`).
   
2. **Ingest PDF file**: Loads the PDF file into memory using `PyPDFLoader` and enriches document metadata with source information.

3. **Chunk documents**: Splits documents into smaller chunks of text to facilitate efficient processing and storage. Metadata for each chunk includes a chunk index.

4. **BigQuery Vector Store**: Prepares a BigQuery dataset and table for storing embedded document chunks using a specified Vertex AI embedding model (`textembedding-gecko@latest`).

5. **Add documents to the vector store**: Uploads document chunks into the BigQuery table for indexing and retrieval.

6. **Chatbot interaction**: Sets up a retrieval system using Langchain to answer questions based on the stored document chunks. Uses a Vertex AI language model (`gemini-1.5-flash` or `gemini-pro`) for answering queries.

7. **Cleaning up**: Deletes the BigQuery dataset to remove all stored data after use.

## Usage
1. Clone the repository: `git clone git@github.com:marcopellegrinoit/LLM-RAG-GCP.git` and `cd LLM-RAG-GCP`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run web app locally: `cd app` and `python -m streamlit run app.py`, or deploy it on [Hugging Face](https://huggingface.co/)

## Credits
The work is based on the official GCP [documentation and tutorials](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/retrieval-augmented-generation/rag_qna_with_bq_and_featurestore.ipynb).