import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the question asked."),
    ("user", "Question: {question}")
])

# Streamlit UI
st.title("LangChain Demo with Gemma Model")
input_text = st.text_input("What question do you have in mind?")

# Ollama model
llm = Ollama(model="gemma2:2b")

# Output parser
output_parser = StrOutputParser()

# Chain
chain = prompt | llm | output_parser

# Run only if input is provided
if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)