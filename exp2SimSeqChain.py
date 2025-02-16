## Integrate our code OpenAI API
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain


import streamlit as st


os.environ["OPENAI_API_KEY"]=openai_key

# streamlit framework

st.title('Know about the Celebrity You Want')
input_text = st.text_input("Search whatever you want")


# Prompt Template

first_input_prompt=PromptTemplate(
    input_variables=['name']
    template="Tell me about the celebrity {name}"
)


## OpenAI LLM Model

llm = OpenAI(temperature=0.8)
chain=LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='person')


# 2nd Prompt Template

second_input_prompt=PromptTemplate(
    input_variables=['person']
    template="When was {person} born"
)

chain2=LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob')

parent_chain=SimpleSequentialChain(chains=[chain, chain2], verbose=True)


if input_text:
    st.write(parent_chain.run(input_text))

    

