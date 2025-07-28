import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please answer the user's queries."),
    ("user", "Question:{question}")
])

# Function to generate response using OpenAI
def generate_openai_response(question, api_key, llm, temperature, max_tokens):
    openai.api_key = api_key
    llms = ChatOpenAI(model=llm, temperature=temperature, max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain = prompt | llms | output_parser
    return chain.invoke({'question': question})

# Function to generate response using Ollama
def generate_ollama_response(question, engine):
    llms = OllamaLLM(model=engine)
    output_parser = StrOutputParser()
    chain = prompt | llms | output_parser
    return chain.invoke({'question': question})

# Streamlit UI
st.title("🔍 Enhance Q&A Chatbot")
st.sidebar.title("⚙️ Settings")

# Select backend: OpenAI or Open Source
model_type = st.sidebar.radio("Choose Model Type:", ["OpenAI (Paid)", "Open Source (Ollama)"])

# Conditional sidebar inputs
if model_type == "OpenAI (Paid)":
    api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
    llm = st.sidebar.selectbox("🤖 OpenAI Model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])
    temperature = st.sidebar.slider("🌡️ Temperature", min_value=0.0, max_value=1.0, value=0.7)
    max_tokens = st.sidebar.slider("🔢 Max Tokens", min_value=50, max_value=300, value=150)
else:
    engine = st.sidebar.selectbox("🧠 Ollama Model", ["gemma2:2b", "mistral", "llama3"])

# Session state for input and response
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "last_response" not in st.session_state:
    st.session_state.last_response = ""

# Input field and submit button in a form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", placeholder="Type your question and press Enter or click Submit")
    submit_btn = st.form_submit_button(label="💬 Submit")

# Handle form submission
if submit_btn and user_input:
    st.session_state.last_input = user_input  # Save input to session state
    try:
        if model_type == "OpenAI (Paid)":
            if not api_key:
                st.warning("Please enter your OpenAI API key.")
            else:
                response = generate_openai_response(user_input, api_key, llm, temperature, max_tokens)
                st.session_state.last_response = response
        else:
            response = generate_ollama_response(user_input, engine)
            st.session_state.last_response = response
    except Exception as e:
        st.session_state.last_response = f"Error: {str(e)}"

# Display last interaction
if st.session_state.last_input and st.session_state.last_response:
    st.markdown(f"**🧑 You:** {st.session_state.last_input}")
    st.markdown(f"**🤖 Assistant:** {st.session_state.last_response}")
