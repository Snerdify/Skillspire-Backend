
import os
from apikey import apikey
import streamlit as st 
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = apikey

# llm
llm = ChatOpenAI(temperature=0.9)

# prompt_template
from langchain_core.prompts import ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_messages([
    ("system" , "You are world class technical documentation writer") , 
    ("user", "{input}")
])

# output-parser 
output_parser = StrOutputParser()

# chain = llm + model + output_parser

chain = LLMChain(llm=llm , prompt = prompt_template, output_parser=output_parser)


# visualize   
st.title('Skillspire')
text = st.text_input('Interact with your model')
if text:
    st.write(chain.run({"input" : text  }))


# llm.invoke("Are angels real?")
# using the prompt template  
# chain = llm | prompt 
# combine the llm and prompt into a chain
# invoke the chain
# chain.invoke({"input" : "are angels real?"})
# parse the output to be a string










