import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    llm=GooglePalm()
    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain

def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i%2 == 0:
            st.write("**User:** "+ message.content)
        else:
            ans="**PDF Guru:** "+ message.content
            st.info(ans)

def submit():
    st.session_state.user_question = st.session_state.widget
    st.session_state.widget = ""

def main():
    st.set_page_config("Talk2PDF 📊")
    st.header("Talk2PDF: Conversations with PDF Guru 👁️‍🗨️")
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""
    user_question = st.text_input("Hey, Ask a Question from your PDF Files",key="widget", on_change=submit)
    user_question = st.session_state.user_question
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    if st.button("Find My Answer"):
        if user_question:
            user_input(user_question) 
        else:
            st.write("Enter your question first") 
    with st.sidebar:
        st.title("Settings")
        st.markdown("To get your API key, visit [Generative AI - PALM](https://developers.generativeai.google/products/palm)")
        api_key = st.text_input("Enter your API Key:", type="password")
        if st.button("Submit"):
            if api_key:
                st.success("API Key submitted successfully!")
                api_key1 = st.secrets["google_api_key"]
                os.environ['GOOGLE_API_KEY'] = api_key1
            else:
                st.warning("Please enter a valid API Key.")
        st.subheader("Share your Documents for Upload")
        pdf_docs = st.file_uploader("Submit your PDF files and tap the 'Execute' button.", accept_multiple_files=True)
        if st.button("Execute"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("Done")

if __name__ == "__main__":
    main()

