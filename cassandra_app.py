from langchain_community.vectorstores import Cassandra
#from langchain.indexes import VectorstoreIndexCreator
from langchain_classic.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from PyPDF2 import PdfReader
import os
import cassio 
from langchain_text_splitters import CharacterTextSplitter


doc=PdfReader("documents/budget_speech.pdf")

## read every page and add it to raw
raw=''
for i , page in enumerate(doc.pages):
    content = page.extract_text()
    if content :
        raw += content


### initialize the connection to your datatbase 

cassio.init(token=os.getenv('ASTRA_DB_APPLICATION_TOKEN'), database_id=os.getenv('ASTRA_DB_ID'))

llm= ChatGoogleGenerativeAI(
    model="gemini-1.5-flash"
)

embedding = GoogleGenerativeAIEmbeddings(
    model='models/gemini-embedding-001'
)

aster_vector_store= Cassandra(
    embedding=embedding,
    table_name="qa_mini_demo"
)
# make the chunks 
text_splitter= CharacterTextSplitter(
    separator="\n",
    chunk_size=1500,
    chunk_overlap=100
)

chunks=text_splitter.split_text(raw)

# put the embeddings in cassendra 
aster_vector_store.add_texts(chunks)

astra_vector_index=VectorStoreIndexWrapper(vectorstore=aster_vector_store)

while True:
    query = input("Ask a question: ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    if query.strip():
        answer = astra_vector_index.query(
            query,
            llm=llm
        )

        print("\nAnswer:")
        print(answer)
    else:
        print("Please enter a question.")


### el chunks ely esta5dmha 

print("FIRST DOCUMENTS BY RELEVANCE:")

for doc, score in aster_vector_store.similarity_search_with_score(
    query,
    k=4
):
    print(
        '  [%0.4f] "%s ..."'
        % (score, doc.page_content[:84])
    )


