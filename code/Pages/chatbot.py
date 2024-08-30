import streamlit as st
from app import rag_query, process_feedback


st.title("RAG Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
                col1, col2 = st.columns([1,15])
                with col1:
                    if st.button("👍", key=f"thumbs_up_{i}"):
                        process_feedback(st.session_state.messages[i-1]["content"], message["content"], True)
                with col2:
                    if st.button("👎", key=f"thumbs_down_{i}"):
                        process_feedback(st.session_state.messages[i-1]["content"], message["content"], False)
                   # st.session_state.messages.pop()  # Remove the last assistant message
                    #st.rerun()  # Rerun the app to regenerate the response

# React to user input
if prompt := st.chat_input("What is your question?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = rag_query(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Rerun the app to display the feedback buttons
    st.experimental_rerun()

# Sidebar for additional controls
with st.sidebar:
    st.header("Options")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()