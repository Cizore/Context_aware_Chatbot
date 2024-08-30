import streamlit as st
from huggingface_hub import InferenceClient
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import chromadb
from langchain.memory import ConversationBufferMemory
from streamlit_extras.switch_page_button import switch_page

# Set page config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Set your Hugging Face token here
HF_TOKEN = "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"

# Initialize your models, databases, and other components here
@st.cache_resource
def init_chroma():
    persist_directory = "chroma_db"
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    chroma_collection = chroma_client.get_or_create_collection("my_collection")
    return chroma_client, chroma_collection

@st.cache_resource
def init_vectorstore():
    persist_directory = "chroma_db"
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name="my_collection")
    return vectorstore

# Initialize components
client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token=HF_TOKEN)
chroma_client, chroma_collection = init_chroma()
vectorstore = init_vectorstore()

# Initialize memory buffer
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def rag_query(query):
    # Retrieve relevant documents using similarity search
    retrieved_docs = vectorstore.similarity_search(query, k=3)

    # Prepare context for LLaMA
    if retrieved_docs:
        context = "\n".join([doc.page_content for doc in retrieved_docs])
    else:
        context = ""

    # Append new interaction to memory
    memory.chat_memory.add_user_message(query)

    # Retrieve past interactions for context
    past_interactions = memory.load_memory_variables({})[memory.memory_key]
    context_with_memory = f"{context}\n\nConversation History:\n{past_interactions}"

    # Debugging: Display context and past interactions
    # st.write("Debugging Info:")
    # st.write("Context Sent to Model:", context_with_memory)
    # st.write("Retrieved Documents:", [doc.page_content for doc in retrieved_docs])
    # st.write("Past Interactions:", past_interactions)

    # Generate response using LLaMA
    messages = [
        {"role": "user", "content": f"Context: {context_with_memory}\n\nQuestion: {query},it is not mandatory to use the context\n\nAnswer:"}
    ]

    # Get the response from the client
    response_content = client.chat_completion(messages=messages, max_tokens=500, stream=False)

    # Process the response content
    response = response_content.choices[0].message.content.split("Answer:")[-1].strip()

    # If the response is empty or very short, or if no relevant documents were found, use the LLM's default knowledge
    if not context or len(response.split()) < 35 or not retrieved_docs:
        messages = [{"role": "user", "content": query}]
        response_content = client.chat_completion(messages=messages, max_tokens=500, stream=False)
        response = response_content.choices[0].message.content

    # Append the response to memory
    memory.chat_memory.add_ai_message(response)

    return response

def process_feedback(query, response, feedback):
   # st.write(f"Feedback received: {'👍' if feedback else '👎'} for query: {query}")
    if feedback:
        # If thumbs up, store the response in memory buffer
        memory.chat_memory.add_ai_message(response)
    else:
        # If thumbs down, remove the response from memory buffer and regenerate the response
       # memory.chat_memory.messages = [msg for msg in memory.chat_memory.messages if msg.get("content") != response]
        new_query=f"{query}. Give better response"
        new_response = rag_query(new_query)
        st.markdown(new_response)
        memory.chat_memory.add_ai_message(new_response)

# Streamlit interface

st.title("Welcome to our RAG-Based Chatbot")
st.markdown("***")
st.info('''
        To use Our Mistral supported Chatbot, click Chat.
         
        To push data, click on Store Document.
        ''')

col1, col2 = st.columns(2)

with col1:
    chat = st.button("Chat")
    if chat:
        switch_page("chatbot")

with col2:
    rag = st.button("Store Document")
    if rag:
        switch_page("management")

st.markdown("<div style='text-align:center;'></div>", unsafe_allow_html=True)