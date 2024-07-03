import streamlit as st
from langchain.chains import RetrievalQA
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_google_community import BigQueryVectorStore
import config

# create embedding model (the same used when embedding the documents)
embedding_model = VertexAIEmbeddings(
    model_name="textembedding-gecko@latest", project=config.PROJECT_ID
)

# retrieve vector store
bq_store = BigQueryVectorStore(
    project_id=config.PROJECT_ID,
    location=config.REGION,
    dataset_name=config.DATASET,
    table_name=config.TABLE,
    embedding=embedding_model,
)

# retrieve chain retrieve associated with the vector store
langchain_retriever = bq_store.as_retriever()

# define main LLM
llm = VertexAI(model_name="gemini-1.5-flash")

def main():
    st.title("Chatbot")
    st.write("RAG LLM from a document on CloudStorage, using BigQueryVectorStore and Gemini.")

    # Input field for user question
    search_query = st.text_input("Enter your question:")

    if st.button("Ask"):
        if search_query:
            with st.spinner('Searching for answer...'):
                # Perform question answering
                response = perform_question_answering(search_query)
                
                st.subheader("Answer:")
                st.write(response)
        else:
            st.warning("Please enter a question.")

def perform_question_answering(question):
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=langchain_retriever
    )
    response = retrieval_qa.invoke(question)
    return response["result"]

if __name__ == "__main__":
    main()