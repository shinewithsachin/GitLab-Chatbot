import streamlit as st
import os

st.set_page_config(page_title="GitLab Handbook Assistant", page_icon="🦊", layout="wide")

# Custom CSS for a professional look (GitLab aesthetic)
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #faf9f6;
    }
    
    /* GitLab inspired highlight color for text */
    h1, h2, h3 {
        color: #e24329 !important; /* GitLab Orange/Red */
    }
    
    /* Stylish subtle container for chat messages */
    .stChatMessage {
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 8px;
    }
    
    div[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #ffffff; /* AI messages */
        border-left: 5px solid #292961; /* GitLab Purple */
    }
    
    div[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #fceceb; /* User messages (light orange) */
        border-left: 5px solid #e24329; /* GitLab Orange */
    }
</style>
""", unsafe_allow_html=True)

st.title("🦊 GitLab Handbook & Direction Assistant")
st.markdown("Ask anything about GitLab's open handbook, company culture, and product directions! Embracing the **build in public** philosophy.")

with st.sidebar:
    st.image("https://about.gitlab.com/images/press/logo/png/gitlab-logo-1-color-black-rgb.png", width=150)
    st.header("Configuration")
    
    api_key_input = st.text_input(
        "Gemini API Key",
        value=st.session_state.get("api_key", ""),
        placeholder="Paste your API key here",
        type="password",
        key="api_key_widget"
    )
    
    # Persist key in session state and set env var
    if api_key_input:
        st.session_state["api_key"] = api_key_input
        os.environ["GOOGLE_API_KEY"] = api_key_input
    
    st.markdown("Don't have one? Get an API key from [Google AI Studio](https://aistudio.google.com).")
    
    st.divider()
    st.markdown("### Resources")
    st.markdown("- [GitLab Handbook](https://handbook.gitlab.com)")
    st.markdown("- [GitLab Direction Pages](https://about.gitlab.com/direction)")
    
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = [{"text": "Hello! I am your AI assistant. I can help you find information about GitLab's Handbook and product directions. What would you like to know?", "is_user": False}]
        st.rerun()

if not st.session_state.get("api_key"):
    st.info("👋 Welcome! Please provide your Google API Key in the sidebar to begin.")
    st.stop()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append({"text": "Hello! I am your AI assistant. I can help you find information about GitLab's Handbook and product directions. What would you like to know?", "is_user": False})

# Display chat history
for message in st.session_state.chat_history:
    role = "human" if message["is_user"] else "ai"
    with st.chat_message(role, avatar="🧑‍💻" if message["is_user"] else "🦊"):
        st.markdown(message["text"])

def handle_query(user_input):
    import time
    from rag_chain import get_rag_chain
    from langchain_core.messages import HumanMessage, AIMessage
    
    try:
        rag_chain = get_rag_chain()
        
        # Convert internal dict history to LangChain message history
        chat_history_lc = []
        for msg in st.session_state.chat_history[:-1]:
            if msg["is_user"]:
                chat_history_lc.append(HumanMessage(content=msg["text"]))
            else:
                chat_history_lc.append(AIMessage(content=msg["text"]))
        
        # Auto-retry only on per-minute rate limits (not daily exhaustion)
        max_retries = 2
        for attempt in range(1, max_retries + 1):
            try:
                response = rag_chain.invoke({
                    "input": user_input,
                    "chat_history": chat_history_lc
                })
                return response["answer"]
            except Exception as e:
                err = str(e)
                # Only retry on per-minute limits, not daily quota exhaustion
                if "429" in err and "PerMinute" in err and attempt < max_retries:
                    time.sleep(15)
                    continue
                raise
        
    except FileNotFoundError:
        return "⚠️ **Vector database not found!**\n\nThe knowledge base hasn't been built yet. Please run `data_loader.py` to ingest the GitLab Handbook."
    except ValueError as ve:
        return f"⚠️ **Configuration Error**: {str(ve)}"
    except Exception as e:
        if "429" in str(e):
            return "⚠️ **API rate limit reached.** The Gemini API free tier has limited requests. Please wait a minute and try again, or enter a new API key from a different Google account in the sidebar."
        return f"⚠️ **Something went wrong**: {str(e)}"

# Handle user input
st.markdown("### Quick Questions")
col1, col2, col3 = st.columns(3)
if col1.button("🦊 What are GitLab's core values?"):
    st.session_state.preset_query = "What are GitLab's core values?"
if col2.button("📝 Tell me about Accounts Payable"):
    st.session_state.preset_query = "Tell me about Accounts Payable"
if col3.button("🚀 What's coming in the latest releases?"):
    st.session_state.preset_query = "What's coming in the latest releases?"

input_value = user_input if (user_input := st.chat_input("Ask a question about GitLab's culture or plans...")) else st.session_state.pop("preset_query", None)

if input_value:
    # Show user message
    with st.chat_message("human", avatar="🧑‍💻"):
        st.markdown(input_value)
    st.session_state.chat_history.append({"text": input_value, "is_user": True})
    
    # Get and show AI response
    with st.chat_message("ai", avatar="🦊"):
        with st.spinner("Searching the handbook..."):
            answer = handle_query(input_value)
            
            # Streaming / Typing effect
            def stream_data():
                import time
                for word in answer.split(" "):
                    yield word + " "
                    time.sleep(0.01)
                    
            st.write_stream(stream_data)
            
    st.session_state.chat_history.append({"text": answer, "is_user": False})
