# Delving into Chatbots  


import os
import getpass
from apikey import apikey
import streamlit as st 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder 
from langchain_core.messages import HumanMessage , AIMessage 



os.environ["OPENAI_API_KEY"] = apikey

# llm
llm = ChatOpenAI(temperature=0.9)


# prompt template 
prompt_template = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user" , "{input}" ),
    ("user" , "Given the above conversation , generate a search query to look up in order to get information relevant to the conversation")])

# output-parser
output_parser = StrOutputParser()

# external data 
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector_db = FAISS.from_documents(documents, embeddings)

retriever = vector_db.as_retriever() 
retriever_chain = create_history_aware_retriever(llm, retriever, prompt_template)


chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"),
                AIMessage(content="Yes!")]

st.title('Skillspire')
if chat_history: 
    st.write(retriever_chain.invoke({ "chat_history": chat_history ,
                                    "input": "Tell me More about it "}))