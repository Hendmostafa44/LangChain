from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader  # use with pdfs in general
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv
load_dotenv()

def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)  ## read files can work in langchain flow and return LangChain Documents
    doc=file_loader.load()
    return doc

doc=read_doc("documents/")
print(doc.__len__())


text_splitter = RecursiveCharacterTextSplitter(   
    chunk_size=1000,
    chunk_overlap=200,   
)

doc = text_splitter.split_documents(doc) # call text_splitter and return array
print(doc.__len__())

embed = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')

vect=embed.embed_query("hi")
print(vect.__len__())  

## Vector search DB In Pinecone



