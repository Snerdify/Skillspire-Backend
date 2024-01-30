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
loader = PyPDFLoader("example_data/World-War-II.pdf")
pages = loader.load_and_split()

# faiss-index is equivalent to vector_db in the app2.py
faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())



# This code lets u visualize the question in the cmd
#docs = faiss_index.similarity_search("What was the world war time period?", k=2)
#for doc in docs:
 #   print(str(doc.metadata['page']) + ":" , doc.page_content[:100])






# prompt_template
prompt_template2 = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# output-parser 
output_parser = StrOutputParser()

document_chain = create_stuff_documents_chain(llm, prompt= prompt_template2,output_parser=output_parser)

retriever = faiss_index.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


#response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
# print(response["answer"])

st.title('Skillspire')
text = st.text_input('Interact with your model')
# context = Document(page_content="langsmith can let you visualize test results") # Assuming you want to use the first document as context
if text:
    st.write(retrieval_chain.invoke({ "input": "What is the world war time period?"}))





                                                    




