from langchain_core.embeddings import Embeddings
import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
#from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

MODEL_NAME="gpt-4"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def generate_response(input_text):
	# Select embedding
	embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)
	# Vector store
	db = Chroma(embedding_function=embeddings)  
	# Search the DB
	results = db.similarity_search_with_relevance_scores(input_text, k=3)
	if len(results) == 0 or results[0][1] < 0.7:
		st.info("Không tìm thấy kết quả nào ")      
	else:
		context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
		prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
		prompt = prompt_template.format(context=context_text, question=input_text)
		#st.info(prompt)
		model=ChatOpenAI(model=MODEL_NAME,temperature=0.0,max_tokens=1000, api_key = openai_api_key)
		response=model.invoke(prompt)
		st.info(response.content)
		st.session_state.history.append((input_text,response.content))
		# Show chat history
		st.divider()
		st.markdown("##Chat History")
		for i, (q,r) in enumerate(st.session_state.history):
			st.markdown(f"**Query: {i+1}:** {q}")
			st.markdown(f"**Response: {i+1}:** {r}")
			st.markdown(f"-----")

	

# Page title
st.set_page_config(page_title="Personal Assistant Chatbot")
st.title = "FINAL PROJECT"
st.header="Personal AI Assisstant"
# Sidebar
with st.sidebar:	
	openai_api_key = st.text_input("OpenAI API Key","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", type="password")
	# File upload
	uploaded_file=st.file_uploader('Upload your file:',type='txt')
	#if st.button("Load Data"):
		#Load document if file is uploaded
	if st.button("Ingest data"):
		if uploaded_file is not None:
			# Read txt file
			documents=[uploaded_file.read().decode()]
	
			# Split documents into chunks
			#text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=100,length_function=len,add_start_index=True)
			text_splitter = CharacterTextSplitter(chunk_size=100,chunk_overlap=100)
			chunks = text_splitter.create_documents(documents)
	
			# Select embedding
			embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)
	
			# Create vector store from document
			db = Chroma.from_documents(chunks,embeddings)    
			st.write(f"Splitted {len(documents)} documents into {len(chunks)} chunks.")
# Inittial history 
if "history" not in st.session_state:
	st.session_state.history=[]

query = st.text_input("Bạn muốn hỏi gì?")
submitted= st.button("Query",disabled= not(uploaded_file))
if not openai_api_key.startswith("sk"):
	st.warning("Please enter your OpenAI API Key", icon=None)
if submitted and openai_api_key.startswith("sk-"):
	generate_response(query)
	