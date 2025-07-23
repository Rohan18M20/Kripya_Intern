######## Playwright with MCP ########


import os
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv

load_dotenv()


# ===== 1. MCP Client =====
class MCPClient:
    """
    Smart context selector for chat history + webpage content.
    """
    def select_context(self, query, chat_history, webpage_content, max_tokens=1500):
        """
        - Summarize/reduce chat history to relevant turns.
        - Truncate/clean webpage content if needed.
        """
        # Step A: Compress chat history
        summarized_chat = self.summarize_chat(chat_history)

        # Step B: Trim webpage content if too long
        trimmed_webpage = self.trim_text(webpage_content, max_tokens // 2)

        # Merge both contexts
        combined_context = f"Chat Summary:\n{summarized_chat}\n\nWebpage Content:\n{trimmed_webpage}"
        return combined_context

    def summarize_chat(self, chat_history, max_chars=1000):
        """
        Naively concatenate chat turns; truncate if too big.
        (Can replace with LLM summarization later.)
        """
        history_text = ""
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                history_text += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_text += f"AI: {msg.content}\n"

        if len(history_text) > max_chars:
            return "...(previous chat truncated)...\n" + history_text[-max_chars:]
        return history_text

    def trim_text(self, text, max_tokens):
        """
        Truncate text to fit within token limits.
        """
        words = text.split()
        if len(words) > max_tokens:
            return " ".join(words[:max_tokens]) + "\n...(content truncated)..."
        return text

# ===== 2. Webpage Fetcher =====
def fetch_with_playwright(url):
    """
    Fetch dynamic pages using Playwright.
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0"
            })
            page.goto(url, timeout=15000)
            html_content = page.content()
            browser.close()

            soup = BeautifulSoup(html_content, "html.parser")
            clean_text = soup.get_text(separator=" ", strip=True)
            return clean_text
    except Exception as e:
        return f"Playwright failed: {str(e)}"

def fetch_with_requests(url):
    """
    Fetch static pages using Requests.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        clean_text = soup.get_text(separator=" ", strip=True)
        return clean_text
    except Exception as e:
        return f"Requests fallback failed: {str(e)}"

def fetch_and_clean_webpage(url):
    """
    Try Playwright first, fallback to Requests.
    """
    content = fetch_with_playwright(url)
    if content.startswith("Playwright failed"):
        st.warning("Playwright failed. Trying fallback method...")
        content = fetch_with_requests(url)
    return content

# ===== 3. Initialize LLM =====
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# ===== 4. Initialize MCP =====
mcp_client = MCPClient()

# ===== 5. Streamlit App =====
st.title("Chatbot with MCP: Chat History + Webpage Context")

# Session state
if "full_chat_history" not in st.session_state:
    st.session_state.full_chat_history = []
if "chat_stopped" not in st.session_state:
    st.session_state.chat_stopped = False

# Display chat history
for msg in st.session_state.full_chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# User input
if not st.session_state.chat_stopped:
    user_input = st.chat_input("Type your message here (type 'exit' to stop)...")
    if user_input:
        if user_input.lower() == "exit":
            st.session_state.chat_stopped = True
            st.info("Conversation stopped. Refresh to start again.")
        else:
            # Add user message
            st.session_state.full_chat_history.append(HumanMessage(content=user_input))

            # Check for URL in user input
            webpage_text = ""
            if "http" in user_input:
                url_start = user_input.find("http")
                url = user_input[url_start:].split()[0]
                st.info(f"Fetching webpage: {url}")
                webpage_text = fetch_and_clean_webpage(url)

            # Use MCP to merge chat history + webpage content
            combined_context = mcp_client.select_context(
                query=user_input,
                chat_history=st.session_state.full_chat_history,
                webpage_content=webpage_text,
                max_tokens=1500
            )

            # Prepare prompt for LLM
            rag_prompt = PromptTemplate(
                template="""
                You are a helpful assistant.
                Answer the user query using the following context.

                Context:
                {context}

                User Question: {question}
                """,
                input_variables=['context', 'question']
            )
            final_prompt_text = rag_prompt.invoke({
                "context": combined_context,
                "question": user_input
            }).to_string()

            # Compose messages for LLM
            messages_for_reply = [
                SystemMessage(content="You are a helpful AI assistant."),
                HumanMessage(content=final_prompt_text)
            ]

            # Get reply
            reply = model.invoke(messages_for_reply)

            # Add AI reply to chat history
            st.session_state.full_chat_history.append(AIMessage(content=reply.content))

            # Rerun Streamlit to show updated chat
            st.rerun()
else:
    st.info("Conversation stopped. Refresh to start again.")
