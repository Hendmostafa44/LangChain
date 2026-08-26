## conversational Q & A chatpot
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()



st.set_page_config(page_title="conversational Q & A chatpot")  ## name of tab
st.header("hey, let's chat")



chat=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
)

## use session_ state 3l4an el chat yftkr ely etkal kbl kda fel chat

if "flowMessages" not in st.session_state:
    st.session_state['flowMessages']=[
        SystemMessage(content="You are a helpful assistant.")  ## y3ny lw de awel session ->  5ly awel rsala ela el chatbot hya de
    ]


def get_gimini_response(question):
    st.session_state["flowMessages"].append(HumanMessage(content=question))
    answer=chat.invoke( st.session_state["flowMessages"])  ##  nb3t lel chat el mohdsa kolha
    st.session_state["flowMessages"].append(AIMessage(content=answer.content))
    return answer.content[0]['text']



input=st.text_input("Input: ",key="input")


submit=st.button("Ask the question")

## If ask button is clicked

if submit:
    if input.strip():
        response = get_gimini_response(input)

        st.subheader("The Response is")
        st.write(response)

    else:
        st.warning("Please enter a question.")

