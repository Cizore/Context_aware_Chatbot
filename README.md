## RAG Chatbot Application🤖

## Introduction
This project implements a Context-Aware Retrieval-Augmented Generation (RAG) chatbot using Streamlit. The chatbot combines cutting-edge natural language processing with efficient information retrieval to deliver intelligent, context-sensitive responses. It is powered by the Mistral-7B-Instruct-v0.3 language model, a state-of-the-art AI capable of understanding and generating human-like text. The system is integrated with ChromaDB, a high-performance vector database, enabling rapid and relevant information retrieval to augment the chatbot's responses. This architecture allows the chatbot to provide more accurate, informative, and contextually appropriate answers by leveraging both its language understanding capabilities and a vast knowledge base.

## Hugging Face Model Token
To obtain a token for Hugging Face, you need to create an account on their website (huggingface.co) if you haven't already. Once logged in, go to your account settings by clicking on your profile picture in the top right corner and selecting "Settings". In the settings menu, navigate to the "Access Tokens" section. Here, you can create a new token by clicking "New token", giving it a name, and selecting the appropriate permissions. After creation, you'll be shown the token - make sure to copy it immediately as it won't be displayed again for security reasons. Keep this token secure and use it in your applications to authenticate with Hugging Face's services.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [License](#license)

## Installation
To install and set up the project, follow these steps:

1. Clone the repository.
    ```bash
    git clone https://github.com/Cizore/Context_aware_Chatbot.git
    ```
2. Navigate to the project directory.
    ```bash
    cd Context_aware_Chatbot
    ```
3. Install the required dependencies.
    ```bash
    pip install -r requirements.txt
    ```
4. Set your Hugging Face token in the `app.py` file:
   ```python
   HF_TOKEN = st.secrets["HF_TOKEN"]
   ```

## Usage
1. Run the main application:
    ```bash
    streamlit run app.py
    ```
    ### OR
     -deployed on streamlit:https://team-qubits.streamlit.app/
3. Interact with the chatbot via the web interface.
4. Upload documents using the Document Management section and process them for use within the chatbot.

## Features

1. **Contextual Responses**: The chatbot retrieves relevant documents from a knowledge base and uses them to provide contextual responses to user queries.
2. **Conversational History**: The chatbot maintains a conversation history, allowing it to reference and build upon previous interactions.
3. **Document Management**: The application provides a document management interface, allowing users to upload and store new documents in the knowledge base.
4. **Feedback Mechanism**: Users can provide feedback on the chatbot's responses, which is used to improve the quality of future responses.

## Dependencies
The project relies on the following major dependencies:
- `streamlit`
- `huggingface_hub`
- `langchain`
- `chromadb`

## Configuration
- Store your Hugging Face token in the Streamlit secrets file as `HF_TOKEN`.
- Additional configuration options may be found within the `app.py` file.

