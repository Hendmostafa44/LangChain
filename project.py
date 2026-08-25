from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
#from langchain_huggingface import HuggingFaceEndpoint
from huggingface_hub import InferenceClient
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser   ## AIMessage → text → Python list

load_dotenv()

model=ChatGoogleGenerativeAI(    
    model="gemini-3.5-flash-lite",
    temperature = 0.6
)
response =model.invoke(input="what is the capital of egypt ?")
print(response.content[0]["text"])

client = InferenceClient(
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

response  = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of Morocco?"
        }
    ]
)

print(response.choices[0].message.content)

from langchain_core.messages import HumanMessage, SystemMessage,AIMessage

re=model.invoke([
    SystemMessage(content="you are comedian ai assistant"),
    HumanMessage(content='Please make a comedy punchlines on ai')

])
print(re.content)

class CommaSeparatedOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(",")

templet= "you are a helpful assistant . when the use given any input , you should generate 5 word in comma seperate "
human_temple="{text}"
chatprompt=ChatPromptTemplate.from_messages([
    ("system",templet),
    ("human",human_temple)
])

chain = chatprompt | model | CommaSeparatedOutputParser()
res=chain.invoke({"text":"intelligent"})
print(res)