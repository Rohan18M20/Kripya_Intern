############### Summarizing Chatbot with RAG ###############



import os
import streamlit as st

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

# Step 1: Prepare documents

transcript_text = """
DeepMind is a British artificial intelligence research lab. They have made breakthroughs in deep reinforcement learning, protein folding, and more. Demis Hassabis is the co-founder and CEO.
The lab focuses on solving general intelligence problems and has created systems like AlphaGo, AlphaFold, and others.
"""

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.create_documents([transcript_text])

# Create embeddings
embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# Create Chroma vector store
vector_store = Chroma.from_documents(chunks, embeddings)

# Create retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Step 2: Initialize LLM

llm_endpoint = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm_endpoint)

# Step 3: Session state

if "full_chat_history" not in st.session_state:
    st.session_state.full_chat_history = []
if "summary" not in st.session_state:
    st.session_state.summary = "The conversation so far is: None."
if "chat_stopped" not in st.session_state:
    st.session_state.chat_stopped = False

st.title("RAG Chatbot")

# Display chat history
for msg in st.session_state.full_chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Step 4: User input & RAG logic

if not st.session_state.chat_stopped:
    user_input = st.chat_input("Type your message here (type 'exit' to stop)...")
    if user_input:
        if user_input.lower() == "exit":
            st.session_state.chat_stopped = True
            st.info("Conversation stopped. Refresh to start again.")
        else:
            # Append user message
            st.session_state.full_chat_history.append(HumanMessage(content=user_input))

            # Summarize conversation so far
            summarize_msgs = [
                SystemMessage(content="You are a summarization bot. Summarize the following conversation briefly."),
                *st.session_state.full_chat_history
            ]
            summary_result = model.invoke(summarize_msgs)
            st.session_state.summary = summary_result.content

            # Retrieve context from vector store
            retrieved_docs = retriever.invoke(user_input)
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

            # Prepare final prompt
            rag_prompt = PromptTemplate(
                template="""
                You are a helpful assistant.
                Answer ONLY using the below context. If insufficient, say "I don't know."

                Context:
                {context}

                Question: {question}
                """,
                input_variables=['context', 'question']
            )

            final_prompt_text = rag_prompt.invoke({"context": context_text, "question": user_input}).to_string()

            # Compose messages for actual reply
            messages_for_reply = [
                SystemMessage(content="You are a helpful AI assistant."),
                SystemMessage(content=f"Summary of previous conversation: {st.session_state.summary}"),
                HumanMessage(content=final_prompt_text)
            ]

            # Get reply
            reply = model.invoke(messages_for_reply)

            # Append AI reply
            st.session_state.full_chat_history.append(AIMessage(content=reply.content))

            # Rerun to display new messages
            st.rerun()
else:
    st.info("Conversation stopped. Refresh to start again.")

# Step 5: Show summary at bottom

if st.session_state.summary != "The conversation so far is: None.":
    with st.expander("Final conversation summary so far"):
        st.write(st.session_state.summary)
