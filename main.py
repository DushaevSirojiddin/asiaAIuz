from google.cloud import translate_v2 as translate
import os, dotenv
import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
import openai

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
dotenv.load_dotenv()

# Function to translate text using Google Translate
def translate_text(target, text):
    translate_client = translate.Client()
    result = translate_client.translate(text, target_language=target)
    return result["translatedText"]

chatbot = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0, model_name="gpt-3.5-turbo", max_tokens=150
    ),
    chain_type="stuff",
    retriever=FAISS.from_documents(PyPDFLoader("qa101.pdf").load_and_split(), OpenAIEmbeddings())
        .as_retriever(search_type="similarity", search_kwargs={"k":1})
)

template = """
respond as succinctly as possible. {query}?
"""

prompt = PromptTemplate(
    input_variables=["query"],
    template=template,
)

# Setting page title and header
st.set_page_config(page_title="Asia", page_icon="S:\\uber\\asiaai.jpeg")
st.markdown("<h1 style='text-align: center;'>Asia - AI assistant for all your legal inquiries. 😬</h1>", unsafe_allow_html=True)

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

# Generate a response
def generate_response(prompt):
    # Translate the prompt from Uzbek to English
    print("prompt:", prompt)
    prompt_english = translate_text("en", prompt)
    print("prompt_english:", prompt_english)
    # Get the chatbot's response in English
    response_english = chatbot.run(prompt.format(query=prompt_english))
    print("response_english:", response_english)
    # Translate the response back to Uzbek
    response_uzbek = translate_text("uz", response_english)
    print("response_uzbek:", response_uzbek)
    # Append the response to the session state's messages
    st.session_state['messages'].append({"role": "assistant", "content": response_uzbek})
    return response_uzbek

# container for chat history
response_container = st.container()
# container for text box
container = st.container()
temp = "I want you act as an AI assistant bot developed by BIKAL, whose name is ASIA."
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        if len(st.session_state['messages']) == 0:
            output = generate_response(temp + user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
        # Translate the user's input from Uzbek to English, get the chatbot's response in English,
        # and translate the response back to Uzbek
        else:
            output = generate_response(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
