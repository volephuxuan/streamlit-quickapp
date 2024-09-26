import streamlit as st
from langchain_openai.chat_models import ChatOpenAI

st.title = "Quick Start app"

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

def generate_response(input_text):
	model=ChatOpenAI(temperature=0.7, api_key = openai_api_key)
	st.info(model.invoke(input_text))

with st.form("my_form"):
	text=st.text_area(
		"Enter text:",
		"Làm thế nào để xây dựng ứng dụng RAG với Streamlit và Langchain?",
		)
	submitted= st.form_submit_button("Submit")
	if not openai_api_key.startswith("sk"):
		st.warning("Please enter your OpenAI API Key", icon=None)
	if submitted and openai_api_key.startswith("sk-"):
		generate_response(text)