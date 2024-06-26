import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

api_key1 = st.secrets["google_api_key"]
os.environ['GOOGLE_API_KEY'] = api_key1
genai.configure(api_key=api_key1)

def extract_text_from_pdfs(pdf_files):
    text_content = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text_content += page.extract_text()
    return text_content

def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def create_conversational_chain():
    prompt_template = """
    Answer the question in detail based on the provided context. If the answer is not in the context, respond with "answer is not available in the context".\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

def reset_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

def handle_user_input(user_query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_query)
    chain = create_conversational_chain()
    response = chain({"input_documents": docs, "question": user_query}, return_only_outputs=True)
    return response

def main():
    st.set_page_config(page_title="Talk2PDF 📊")

    with st.sidebar:
        st.title("Configuration")
        api_key_input = st.sidebar.text_input("Enter your Gemini API Key (if available)", type="password", placeholder="If you have!")
        temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.8, 0.1)
        model_selection = st.sidebar.selectbox("Select Model", ("gemini-1.5", "gemini-1.5-pro", "gemini-1.0-pro", "gemini-1.5-flash"))

        pdf_files = st.file_uploader("Upload your PDF Files: ", accept_multiple_files=True)
        st.markdown("App built by Subhayu Dutta")

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = extract_text_from_pdfs(pdf_files)
                text_chunks = split_text_into_chunks(raw_text)
                create_vector_store(text_chunks)
                st.success("Processing complete")

    st.title("Talk2PDF: Conversations with PDF Guru 👁️‍🗨️")
    st.sidebar.button('Clear Chat History', on_click=reset_chat_history)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if user_message := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.write(user_message)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = handle_user_input(user_message)
                full_response = ''.join(response['output_text'])
                st.write(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()