"""
Streamlit UI — just imports from agent.py.
Run: streamlit run app.py
"""

import streamlit as st
from agentreal2 import chat, reset, df

st.set_page_config(page_title="PE Intelligence Agent", page_icon="📊", layout="wide")

# Sidebar
with st.sidebar:
    st.markdown("### 💡 Try these")
    examples = [
        "Top 10 most funded deals",
        "Map the FinTech landscape",
        "Which investors back multiple startups?",
        "Tell me about CRED",
        "Tell me about Stripe's latest funding",
        "Compare Bangalore vs Mumbai",
        "Who is Kunal Shah?",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["pending"] = ex

    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        reset()
        st.session_state.msgs = []
        st.rerun()

    st.markdown("---")
    st.caption(f"Dataset: {len(df)} records")
    st.caption("🟢 Dataset  🔴 External (web)")

# Header
st.markdown("## 📊 PE Intelligence Agent")
st.caption("Ask about startups, funding, investors, or market maps. Uses dataset first, web search when needed.")

# Chat history
if "msgs" not in st.session_state:
    st.session_state.msgs = []

for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input
user_input = st.chat_input("Ask about startups, funding, investors...")

if "pending" in st.session_state:
    user_input = st.session_state.pop("pending")

if user_input:
    st.session_state.msgs.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chat(user_input)
            except Exception as e:
                response = f"⚠️ Error: {e}"
        st.markdown(response)

    st.session_state.msgs.append({"role": "assistant", "content": response})
