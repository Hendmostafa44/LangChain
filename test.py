from langchain_core.prompts import PromptTemplate , FewShotPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
prompt = PromptTemplate(
    input_variables=["financial_concept"],
    template="Explain the financial concept of {financial_concept} in simple terms. act as a financial expert "
)
#print(prompt.format(financial_concept="incoming tax ")) # use format to test the prompt template


model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.8
)

chain1 = prompt | model  ## mean give the output of first prompt to model as input. this tool call LCEL
#print(chain1.invoke({"financial_concept": "incoming tax"}) ) # try the model with input variable "incoming tax" and print the output


#######################################################################################################
#################Language Translation , use two inputs #########

prompt2 = PromptTemplate(
    input_variables=["text", "target_language"],
    template="Translate the following text into {target_language}: {text}"
)
chain2= prompt2 | model  

# print(prompt2.format(text="Hello, how are you?", target_language="Arabic")) 
# print(chain2.invoke({"text":"Hello, how are you ","target_language":"French"}))

examples=[
    {"word":"Happy","antonym":"sad"},
    {"word":"tall","antonym":"short"},
]


example_templete=""""
Word:{word}
Antonym:{antonym}
"""

Few_prompt=PromptTemplate(
    input_variables=["word","antonym"],
    template=example_templete
)


few_shot_prompt= FewShotPromptTemplate(
    examples=examples,
    example_prompt=Few_prompt,
    prefix="give the antonym for every input",
    suffix="Word : {input}\nAntonym :",
    input_variables=["input"]

)

############ the output of few_shot_prompt ###############
# Word:Happy
# Antonym:sad


# "
# Word:tall
# Antonym:short


# Word : big
# Antonym :


chain3= few_shot_prompt | model
print(few_shot_prompt.format(input="big"))
respo = chain3.invoke({"input":"big"})
print(respo.content)