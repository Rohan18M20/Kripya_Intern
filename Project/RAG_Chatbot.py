############### Hybrid RAG Chatbot ###############

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()  # load HF_TOKEN, etc.

# ──────────────────────────────────────────────────────────────────────────────
# LLM and embedding models
# ──────────────────────────────────────────────────────────────────────────────
llm_endpoint = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm_endpoint)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit session-state
# ──────────────────────────────────────────────────────────────────────────────
if "full_chat_history" not in st.session_state:
    st.session_state.full_chat_history = []
if "chat_stopped" not in st.session_state:
    st.session_state.chat_stopped = False

st.title("Hybrid RAG Chatbot (first 3 answers = LLM knowledge)")

# ──────────────────────────────────────────────────────────────────────────────
# Display prior messages
# ──────────────────────────────────────────────────────────────────────────────
for msg in st.session_state.full_chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# ──────────────────────────────────────────────────────────────────────────────
# helper: build vector store from *pairs* of User+Assistant turns
# ──────────────────────────────────────────────────────────────────────────────
def build_retriever(history_pairs):
    """Builds a retriever from Q-A pairs using optimized settings."""
    if not history_pairs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    docs = [
        chunk.strip()
        for pair in history_pairs
        for chunk in splitter.split_text(pair)
        if chunk.strip()
    ]

    if not docs:
        return None

    vectordb = Chroma.from_texts(docs, embeddings)

    # Use MMR-based retriever for better relevance + diversity
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "lambda_mult": 0.5
        }
    )
    return retriever

# ──────────────────────────────────────────────────────────────────────────────
# Main chat loop
# ──────────────────────────────────────────────────────────────────────────────
if not st.session_state.chat_stopped:
    user_input = st.chat_input("Type your message ('exit' to stop)…")

    if user_input:
        if user_input.lower() == "exit":
            st.session_state.chat_stopped = True
            st.info("Conversation stopped. Refresh to start again.")
        else:
            # Add user message to history
            st.session_state.full_chat_history.append(HumanMessage(content=user_input))

            # Count complete Q→A pairs
            total_msgs = len(st.session_state.full_chat_history)
            completed_pairs = total_msgs // 2

            use_rag = completed_pairs >= 3

            # ------------------------------------------------------------------
            # build retriever *only* when RAG should be active
            # ------------------------------------------------------------------
            if use_rag:
                
                pairs = []
                msgs = st.session_state.full_chat_history[:-1]
                i = 0
                while i < len(msgs) - 1:
                    if isinstance(msgs[i], HumanMessage) and isinstance(msgs[i+1], AIMessage):
                        pair_text = f"User: {msgs[i].content}\nAssistant: {msgs[i+1].content}"
                        pairs.append(pair_text)
                        i += 2
                    else:
                        i += 1

                retriever = build_retriever(pairs)

                # Retrieve context
                context_text = ""
                if retriever:
                    retrieved = retriever.invoke(user_input)
                    context_text = "\n\n".join(
                        doc.page_content.strip()
                        for doc in retrieved
                        if doc.page_content.strip()
                    )

                # Fallback if no context retrieved
                if not context_text.strip():
                    st.warning("No relevant past context found. Answering: I don't know.")
                    st.session_state.full_chat_history.append(AIMessage(content="I don't know."))
                    st.rerun()

                # Strict-RAG prompt
                rag_prompt = PromptTemplate(
                    template="""
You are a helpful assistant.
Answer ONLY using the context below. If the context is insufficient, respond with: "I don't know."

Context:
{context}

Question: {question}
""",
                    input_variables=["context", "question"]
                )
                final_prompt = rag_prompt.invoke(
                    {"context": context_text, "question": user_input}
                ).to_string()

                messages_for_llm = [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=final_prompt)
                ]
            else:
                # Free-knowledge mode
                messages_for_llm = [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=user_input)
                ]

            # Get assistant reply
            reply = model.invoke(messages_for_llm)

            # Save assistant reply
            st.session_state.full_chat_history.append(AIMessage(content=reply.content))
            st.rerun()
else:
    st.info("Conversation stopped. Refresh to start again.")
