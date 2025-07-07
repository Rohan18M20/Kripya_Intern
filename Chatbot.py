import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Initialize session state
if "full_chat_history" not in st.session_state:
    st.session_state.full_chat_history = []
if "summary" not in st.session_state:
    st.session_state.summary = "The conversation so far is: None."
if "chat_stopped" not in st.session_state:
    st.session_state.chat_stopped = False

st.title("💬 Summarizing Chatbot (Streamlit)")

# Display chat history
for msg in st.session_state.full_chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# User input
if not st.session_state.chat_stopped:
    user_input = st.chat_input("Type your message here (type 'exit' to stop)...")
    if user_input:
        if user_input.lower() == "exit":
            st.session_state.chat_stopped = True
            st.info("✅ Conversation stopped. Refresh the page to start again.")
        else:
            # Append new user message
            st.session_state.full_chat_history.append(HumanMessage(content=user_input))

            # Summarize full conversation
            summarize_messages = [
                SystemMessage(content="You are a summarization bot. Summarize the following conversation briefly."),
                *st.session_state.full_chat_history
            ]
            new_summary_result = model.invoke(summarize_messages)
            st.session_state.summary = new_summary_result.content

            # Prepare messages for actual reply
            messages_for_reply = [
                SystemMessage(content="You are a helpful AI assistant."),
                SystemMessage(content=f"Summary of previous conversation: {st.session_state.summary}"),
                HumanMessage(content=user_input)
            ]

            # Get AI reply
            reply = model.invoke(messages_for_reply)

            # Append AI reply
            st.session_state.full_chat_history.append(AIMessage(content=reply.content))

            # Rerun to show new messages
            st.rerun()
else:
    st.info("Conversation stopped. Refresh the page to start again.")

# Optional: Show final summary at the bottom
if st.session_state.summary != "The conversation so far is: None.":
    with st.expander("🔎 Final conversation summary so far"):
        st.write(st.session_state.summary)