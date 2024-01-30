import os
from apikey import apikey
import streamlit as st 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain

os.environ["OPENAI_API_KEY"] = apikey

# llm
llm = ChatOpenAI(temperature=0.9)

# external data 
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector_db = FAISS.from_documents(documents, embeddings)

# prompt_template
prompt_template2 = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# output-parser 
output_parser = StrOutputParser()

document_chain = create_stuff_documents_chain(llm, prompt= prompt_template2,output_parser=output_parser)





                                                    
