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
from langchain_community.document_loaders import PyPDFLoader

os.environ["OPENAI_API_KEY"] = apikey

# llm
llm = ChatOpenAI(temperature=0.9)

# external data
loader = PyPDFLoader("example_data/World-War-II-With-img.pdf", extract_images = True)
pages = loader.load()
st.write(pages[1].page_content)


# faiss-index is equivalent to vector_db in the app2.py
faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())




