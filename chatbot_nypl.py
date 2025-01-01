import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.runnable import RunnableLambda
from PyPDF2 import PdfReader

from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import openai
from openai import OpenAIError

class CustomConversationMemory(ConversationBufferMemory):
    def save_context(self, inputs, outputs):
        # Only store the 'answer' key in memory
        filtered_outputs = {"answer": outputs["answer"]} if "answer" in outputs else outputs
        super().save_context(inputs, filtered_outputs)

load_dotenv()
openai.api_key = "" # Make sure this matches your .env variable name

# Global dictionaries to store vectorstores and user states keyed by user_id
global_vectorstores = {}
user_states = {}  # structure: {user_id: {"asked_for_collection_confirmation": bool,
                  #                           "awaiting_collection_choice": bool,
                  #                           "last_question": str,
                  #                           "collections_found": list }}

# Custom prompt template
custom_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow-Up Input: {question}
Standalone question:"""
CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

def get_chunks_with_metadata(docs):
    chunks, metadata = [], []
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        text = "".join(page.extract_text() for page in pdf_reader.pages)
        pdf_chunks = text_splitter.split_text(text)
        chunks.extend(pdf_chunks)
        metadata.extend([{"source": pdf.name}] * len(pdf_chunks))
    return chunks, metadata

def get_vectorstore_with_metadata(chunks, metadata):
    documents = [Document(page_content=text, metadata=meta) for text, meta in zip(chunks, metadata)]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embedding=embeddings)
    # Save vectorstore locally for reuse if needed
    vectorstore.save_local("/tmp/vectorstore")
    return vectorstore

def query_gpt(prompt):
    if not isinstance(prompt, str):
        raise ValueError("Prompt must be a string.")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message['content'].strip()
    except OpenAIError as e:
        raise RuntimeError(f"Error querying GPT-4: {e}")

def get_conversationchain_with_metadata(vectorstore):
    llm = RunnableLambda(lambda prompt, **kwargs: query_gpt(str(prompt)))
    memory = CustomConversationMemory(
        memory_key='chat_history',
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity"),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )
    return conversation_chain

def general_search(question):
    # This function performs a general search (no vectorstore)
    return query_gpt(question)

def handle_chatbot_action_test(request):
    user_id = request.POST.get("user_id", "default_user")

    # Initialize user state if not present
    if user_id not in user_states:
        user_states[user_id] = {
            "asked_for_collection_confirmation": False,
            "awaiting_collection_choice": False,
            "last_question": None,
            "collections_found": []
        }

    state = user_states[user_id]

    # Check if we have vectorstore for this user
    vectorstore = global_vectorstores.get(user_id)

    # Handle document upload if vectorstore not exists
    if not vectorstore:
        if "documents" in request.FILES:
            uploaded_files = request.FILES.getlist("documents")
            chunks, metadata = get_chunks_with_metadata(uploaded_files)
            vectorstore = get_vectorstore_with_metadata(chunks, metadata)
            global_vectorstores[user_id] = vectorstore
            return {"message": "Documents processed successfully! You can now ask a question."}
        else:
            return {"error": "No documents uploaded. Please upload documents first."}

    # Handle user input (question)
    if "question" in request.POST:
        user_input = request.POST.get("question", "").strip()
        if not user_input:
            return {"error": "No question provided."}

        # If we are awaiting a yes/no confirmation for a single collection
        if state["asked_for_collection_confirmation"]:
            if user_input.lower() in ["yes", "y"]:
                # User confirmed to use the single identified collection
                question = state["last_question"]
                # Proceed with retrieval
                conversation_chain = get_conversationchain_with_metadata(vectorstore)
                response = conversation_chain({"question": question})
                answer = response.get("answer", "")
                source_documents = response.get("source_documents", [])
                # Reset state
                state["asked_for_collection_confirmation"] = False
                state["last_question"] = None
                state["collections_found"] = []
                return {
                    "answer": answer,
                    "source_documents": source_documents
                }
            elif user_input.lower() in ["no", "n"]:
                # User declined to use the collection, do general search
                question = state["last_question"]
                answer = general_search(question)
                # Reset state
                state["asked_for_collection_confirmation"] = False
                state["last_question"] = None
                state["collections_found"] = []
                return {
                    "answer": answer,
                    "source_documents": []
                }
            else:
                # Unexpected response
                return {"message": "I didn't understand your response. Please say 'yes' or 'no'."}

        # If we are awaiting a collection choice from multiple options
        if state["awaiting_collection_choice"]:
            # User should respond with a collection name or say "no"
            if user_input.lower() == "no":
                # User doesn't want to use any collection, do general search
                question = state["last_question"]
                answer = general_search(question)
                # Reset state
                state["awaiting_collection_choice"] = False
                state["last_question"] = None
                state["collections_found"] = []
                return {
                    "answer": answer,
                    "source_documents": []
                }
            else:
                # Check if user's input matches one of the collections
                chosen_collection = None
                for c in state["collections_found"]:
                    if c.lower() in user_input.lower():
                        chosen_collection = c
                        break
                if chosen_collection:
                    # User chose a collection
                    question = state["last_question"]
                    # Proceed with retrieval
                    conversation_chain = get_conversationchain_with_metadata(vectorstore)
                    response = conversation_chain({"question": question})
                    answer = response.get("answer", "")
                    source_documents = response.get("source_documents", [])
                    # Reset state
                    state["awaiting_collection_choice"] = False
                    state["last_question"] = None
                    state["collections_found"] = []
                    return {
                        "answer": answer,
                        "source_documents": source_documents
                    }
                else:
                    return {
                        "message": "I didn't understand your choice. Please select one of the listed collections or say 'no'."
                    }

        # If we reach here, this is a new question scenario
        # Step 1: Use the vectorstore to find relevant documents (collections)
        docs = vectorstore.similarity_search(user_input, k=3)
        sources = {doc.metadata.get("source", "Unknown Document") for doc in docs if doc.metadata}

        # If no documents found, fallback to general search
        if not sources:
            answer = general_search(user_input)
            return {
                "answer": answer,
                "source_documents": []
            }

        # If exactly one collection found
        if len(sources) == 1:
            only_collection = list(sources)[0]
            only_collection = only_collection.split(".")[0]
            state["asked_for_collection_confirmation"] = True
            state["last_question"] = user_input
            state["collections_found"] = list(sources)
            return {"message": f"I found a collection that might help: {only_collection.upper()}. Would you like me to search in this collection? (yes/no)"}

        # If multiple collections found
        else:
            processed_sources = [source.replace('.pdf', '').upper() for source in sources]
    
            # Join the processed sources into a comma-separated string
            collections_list = ", ".join(processed_sources)
            
            state["awaiting_collection_choice"] = True
            state["last_question"] = user_input
            state["collections_found"] = list(sources)
            return {
                "message": f"I found multiple collections that might be relevant: {collections_list}. Please type the name of the collection you would like to search in, or say 'no' for a general search."
            }
    return {"error": "Invalid request. No action taken."}
