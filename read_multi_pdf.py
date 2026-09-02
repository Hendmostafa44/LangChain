import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import dotenv 
dotenv.load_dotenv()
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate

from langchain_community.vectorstores import FAISS
import streamlit as st

llm=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite")

prompt_template="""
You are a helpful assistant. Use the following pieces of context {context} to answer the question {input} at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."""

pdf=PdfReader("documents/budget_speech.pdf")

raw=""

for i , page in enumerate(pdf.pages):
    content=page.extract_text()
    if content:
        raw += content

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
)

chunks=text_splitter.split_text(raw)

vectorstore=FAISS.from_texts(chunks,GoogleGenerativeAIEmbeddings(model='gemini-embedding-001'))

retriever=vectorstore.as_retriever()

prompt=PromptTemplate(
    input_variables=["context","question"],
    template=prompt_template
)

combine_docs_chain = create_stuff_documents_chain(  ##### el goz' ely by5od el context w el question w by3ml el answer
    llm,
    prompt
)


retrieval_chain = create_retrieval_chain( # el goz' ely byrbt el retriever w el combine_docs_chain , el retriever by3ml search 3n el chunk el akrb ela el question w byb3to lel combine_docs_chain
    retriever,
    combine_docs_chain
)


response = retrieval_chain.invoke({"input": "What is this document about?"})  # create_retrieval_chain dymn 3yz el input variable esmha "input"  w hyb3t el input variable da lel retriever w  combine_docs_chain
print(response)




st.sidebar.header("Upload PDF Files")
pdf_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
st.sidebar.write("Uploaded PDF file:", pdf_file.name if pdf_file else "No file uploaded")
if pdf_file:
    if st.button("submit"):
        with st.spinner("Processing the PDF..."):
            st.success("PDF processed successfully!")


