import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# Load API key from environment
if "MISTRAL_API_KEY" not in os.environ:
    st.error("Mistral API key is missing. Please set it in your environment variables.")
    st.stop()

# Initialize model and embeddings
model = ChatMistralAI(model="mistral-large-latest", temperature=1)
embeddings = MistralAIEmbeddings()
parser = StrOutputParser()

# Define prompt template
custom_instruction = """
You are a helpful AI assistant. Answer the questions based on the provided context.
If you don't know the answer, reply with "I don't know."

Context: {context}
Question: {question}
"""

prompt = PromptTemplate.from_template(custom_instruction)

# Streamlit UI
def main():
    st.set_page_config(page_title="RAG Chatbot", layout="wide")
    st.title("📄 RAG Chatbot")
    st.write("Upload a document, then ask questions based on it!")

    # File Upload
    uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    
    if uploaded_file:
        # file_path = os.path.join("uploads", uploaded_file.name)
        # os.makedirs("uploads", exist_ok=True)
        # with open(file_path, "wb") as f:
        #     f.write(uploaded_file.getbuffer())
        # st.success(f"File {uploaded_file.name} uploaded successfully!")
        
        # Load and process the document
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(uploaded_file.name)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(uploaded_file.name)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(uploaded_file.name)
        else:
            st.error("Unsupported file format.")
            return
        
        pages = loader.load_and_split()
        
        # Create vector store
        vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        
        # Define the RAG chain
        chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question")
            }
            | prompt
            | model
            | parser
        )
        
        st.session_state.chain = chain  # Store chain in session state

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat messages
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["message"])
    
    # User input
    user_query = st.chat_input("Ask a question based on the document...")
    if user_query and "chain" in st.session_state:
        st.session_state.chat_history.append({"role": "user", "message": user_query})
        
        # Get response from RAG model
        response = st.session_state.chain.invoke({"question": user_query})
        
        # Append bot response
        st.session_state.chat_history.append({"role": "assistant", "message": response})
        
        st.rerun()

if __name__ == "__main__":
    main()