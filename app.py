import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS  # vectorstore db
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # vector embeddings
import time
from dotenv import load_dotenv

load_dotenv()

# Load the groq and google api key

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Medical Q&A Chatbot")

# llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-it")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")


# prompt template

prompt = ChatPromptTemplate.from_template(
    """
Answer the following questions based on the provided context only.
Provide the complete and concise answer to the question.
<context>
{context}
<context>
Question: {input}
"""
)
print("Model Loaded")


def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )
        # print(st.session_state.embeddings)
        st.session_state.loader = PyPDFDirectoryLoader("./data")
        print("done")
        st.session_state.docs = st.session_state.loader.load()  ##to load the documents
        print("done")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        print("done")
        st.session_state.final_documents = (
            st.session_state.text_splitter.split_documents(st.session_state.docs)
        )  ##to split the documents
        print("done")
        st.session_state.vectorstore = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )  ##to create the vectorstore
        print("done")
        st.session_state.vectors = st.session_state.vectorstore

        print("Vector Store Created")


prompt1 = st.text_input("What you want to ask from the documents?")

if st.button("Creating Vector Store"):
    vector_embedding()
    st.write("Vector Store Created")


if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriver = st.session_state.vectors.as_retriever()
    retriver_chain = create_retrieval_chain(retriver, document_chain)
    start = time.process_time()
    response = retriver_chain.invoke({"input": prompt1})
    st.write(response["answer"])

    with st.expander("Show Context"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------------")

    st.write("Time taken to process the request: ", time.time() - start, "seconds")
