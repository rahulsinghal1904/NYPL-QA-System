from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.runnable import RunnableLambda
import requests
import json
from htmlTemplates import css, bot_template, user_template

# Custom template to guide the LLM model
custom_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow-Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

# Extracting text from PDF with metadata
def get_chunks_with_metadata(docs):
    chunks = []
    metadata = []
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        pdf_chunks = text_splitter.split_text(text)
        chunks.extend(pdf_chunks)
        metadata.extend([{"source": pdf.name}] * len(pdf_chunks))  # Keep track of source PDF
    return chunks, metadata

# Using embeddings model and FAISS with metadata
def get_vectorstore_with_metadata(chunks, metadata):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vectorstore = faiss.FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadata  # Include metadata
    )
    return vectorstore

# Querying Ollama's Llama 3 model
def query_ollama(prompt):
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    model_name = "llama3"

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": model_name, "prompt": prompt},
            stream=True,
        )
        response.raise_for_status()

        final_response = ""
        for line in response.iter_lines(decode_unicode=True):
            if line.strip():
                try:
                    data = json.loads(line)
                    if "response" in data and isinstance(data["response"], str):
                        final_response += data["response"]
                except json.JSONDecodeError:
                    print(f"Invalid JSON line: {line}")
        
        return final_response.strip()
    except requests.exceptions.RequestException as e:
        print(f"Error querying Ollama: {e}")
        raise

# Create the conversation chain with metadata support
def get_conversationchain_with_metadata(vectorstore):
    llm = RunnableLambda(
        lambda prompt, **kwargs: query_ollama(prompt.to_string() if hasattr(prompt, "to_string") else prompt)
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"metadata": True}),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        memory=memory,
        return_source_documents=True,  # Ensure source documents are returned
    )
    return conversation_chain

# Workflow state initialization
if "workflow_state" not in st.session_state:
    st.session_state.workflow_state = {
        "current_collection": None,  # Stores the current collection the user has selected
        "awaiting_confirmation": False  # Indicates if the system is waiting for user confirmation
    }

# Generating response and handling workflow logic
def handle_question(question):
    # Workflow state
    workflow_state = st.session_state.workflow_state

    if workflow_state["awaiting_confirmation"]:
        # Handle user's response to confirmation
        if question.lower() in ["yes", "proceed", "go ahead"]:
            st.write(f"Searching within the collection: {workflow_state['current_collection']}")
            # Filter the vectorstore by collection and reset workflow state
            workflow_state["awaiting_confirmation"] = False
            workflow_state["current_collection"] = None
        else:
            st.write("Workflow canceled. Returning to general search.")
            workflow_state["awaiting_confirmation"] = False
            workflow_state["current_collection"] = None
        return

    # Standard question handling
    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response["chat_history"]
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            # Retrieve metadata and determine workflow
            retriever_metadata = response.get("source_documents", [])
            if retriever_metadata:
                # Extract most relevant collection (source document name)
                sources = {doc.metadata.get("source", "Unknown Document") for doc in retriever_metadata}
                most_relevant_source = next(iter(sources))  # Take the first one (most relevant)
                if "which collection" in question.lower():  # Detect collection-specific questions
                    st.write(f"The item belongs to the collection: {most_relevant_source}.")
                    st.write("Do you want to proceed with this collection?")
                    workflow_state["current_collection"] = most_relevant_source
                    workflow_state["awaiting_confirmation"] = True
                else:
                    # Normal bot response
                    bot_response = f"{msg.content} (Source: {', '.join(sources)})"
                    st.write(bot_template.replace("{{MSG}}", bot_response), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="NYPL QA", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with NYPL Expert :books:")
    question = st.text_input("Ask me a question:")
    if question:
        handle_question(question)

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("Upload your PDF here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get text chunks with metadata
                text_chunks, metadata = get_chunks_with_metadata(docs)

                # Create vectorstore with metadata
                vectorstore = get_vectorstore_with_metadata(text_chunks, metadata)

                # Create conversation chain
                st.session_state.conversation = get_conversationchain_with_metadata(vectorstore)

if __name__ == '__main__':
    main()
