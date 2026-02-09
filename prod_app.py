import streamlit as st
import asyncio
from client_agent import NewsAgentClient
from dotenv import load_dotenv

load_dotenv()

st.title("🚀 News Search using Serper API")

# --- 1. Initialize the Agent Class in Session State ---
@st.cache_resource
def get_agent():
    return NewsAgentClient()

agent_instance = get_agent()
print(type(agent_instance))

# --- 2. Chat UI Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the news..."):
            # Bridge between Sync Streamlit and Async Agent
            try:
                # We use asyncio.run to call the async method of our class
                response = asyncio.run(agent_instance.get_news(prompt))
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")