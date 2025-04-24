import streamlit as st
import os
import sys
from mistralai import Mistral
from langchain_mistralai import MistralAIEmbeddings,ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain import PromptTemplate
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser

api_key=os.environ["MISTRAL_API_KEY"]
client=Mistral(api_key=api_key)
embeddings=MistralAIEmbeddings()
parser=StrOutputParser()

if "mistral_model" not in st.session_state:
    st.session_state.mistral_model=ChatMistralAI(model="mistral-large-latest", temperature=1)

if "messages" not in st.session_state:
    st.session_state.messages=[]

final_pages=[]

custom_instruction=""" 
You are a helpful AI assistant. Answer the questions based on the provided context.
If you don't know the answer, reply with "I don't know."

Context: {context}
Question: {question}
"""     
prompt=PromptTemplate.from_template(custom_instruction)

st.title("SummarAIze")
uploaded_files=st.file_uploader("Upload documents(PDF, DOCX, TXT) to start querying!",accept_multiple_files=True,type=["pdf", "docx", "txt"])
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if len(uploaded_files)>=1:
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            loader=PyPDFLoader(file.name)
            final_pages+=loader.load_and_split()
        elif file.name.endswith(".docx"):
            loader=Docx2txtLoader(file.name) 
            final_pages+=loader.load_and_split()
        elif file.name.endswith(".txt"):
            loader= TextLoader(file.name)
            final_pages+=loader.load_and_split()
        else:
            st.error("Unsupported file format.")
            sys.exit() 

    vectorStore=DocArrayInMemorySearch.from_documents(final_pages,embedding=embeddings) 
    retriever=vectorStore.as_retriever()        
        
    chain=(
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question")
    }
    | prompt
    | st.session_state.mistral_model
    | parser  
    )
    st.session_state.chain=chain        


def stream_read():
    for chunk in stream:
        yield chunk.data.choices[0].delta.content + " "

if user_input := st.chat_input("what is up?"):
    st.session_state.messages.append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    model_response=st.session_state.chain.invoke({"question":user_input}) 
       
    # with st.chat_message("assistant"):
    #     stream=client.chat.stream(
    #         model=st.session_state.mistral_model,
    #         # messages=[
    #         #     {"role":m["role"],"content":m["content"]}
    #         #     for m in st.session_state.messages
    #         # ],
            
    #     )   

        #response=st.write_stream(stream_read())
    st.session_state.messages.append({"role":"assistant","content":model_response})
    with st.chat_message("assistant"):
        st.markdown(model_response)     

