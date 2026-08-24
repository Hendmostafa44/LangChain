import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from langchain_core.prompts import PromptTemplate
load_dotenv()



## prompet template
first_prompt = PromptTemplate(
   input_variables=["name"],
   template="Tell me about {name}"
)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.8
)

secound_prompt = PromptTemplate(
   input_variables=["person"],
   template="Tell me when {person} was born"
)
st.title("Celebrity Info App")
chain= first_prompt | model    ## mean give the output of first prompt to model as input. this tool call LCEL
input_text = st.text_input("Enter the name of a celebrity:")
if input_text:
    response = chain.invoke({"name": input_text}) 
    chain = secound_prompt | model
    date_born = chain.invoke({"person": response})  ## el invoke lazem tt3ml dictionary 3shan el input_variables bta3t el prompt template
    st.info(response.content)  ## el buffer deh el output bta3 el model
    st.info(date_born.content)





