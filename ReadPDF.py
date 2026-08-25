# PDF
#  ↓
# PdfReader
#  ↓
# Text
#  ↓
# CharacterTextSplitter
#  ↓
# Chunks
#  ↓
# GoogleGenerativeAIEmbeddings
#  ↓
# Vectors
#  ↓
# FAISS
#  ↓
# Similarity Search


from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()
from typing_extensions import Concatenate
pdf=PdfReader('document.pdf')
raw=''
for i , page in enumerate(pdf.pages): # read every page and put it in raw=""
    content=page.extract_text()
    if content:
        raw += content

text_splitter = CharacterTextSplitter(
    separator="\n",    # hyksm 3nd el new line 
    chunk_size=20,
    chunk_overlap=5,   # 3dd el 7rof el motada5la ben kol chunk (el mtwst)
)

chunks = text_splitter.split_text(raw) # call text_splitter and return array
print(chunks.__len__())
print(chunks[0])

embed = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')
document_search = FAISS.from_texts(chunks, embed)
print(document_search)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate


model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0
)


retriever = document_search.as_retriever()  # y3ml search 3n el chunk el akrb ela el question

# question = "What is this document about?"

# docs = retriever.invoke(question)

# for doc in docs:
#     print(doc.page_content)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer the question based only on the following context.

Context:
{context}

Question:
{question}

Answer:
"""
)

chain = (
    {
        "context": retriever,   #y3ml search 3n el chunk el akrb ela el question
        "question": RunnablePassthrough()  # ymrr el question zy ma hwa 
    }
    | prompt
    | model
)

response = chain.invoke("What is this document about?")

print(response.content)