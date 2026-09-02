"""Streamlit UI for the research agent -- single screen, single session."""

import os
import sys

import streamlit as st
from dotenv import load_dotenv

from agent import run_research, DEFAULT_MODELS

load_dotenv()

# Same fix as agent.py's CLI path: on Windows, the server process's stdout
# defaults to a codepage that can't encode every character LangChain's
# verbose=True logging tries to print (em dashes, emoji...). Without this,
# each research run spams "Error in StdOutCallbackHandler... callback"
# traces in the terminal even though the run itself succeeds.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

st.set_page_config(page_title="Research Agent", page_icon="🔎")

KEY_VAR = {"groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY"}

# --- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    provider = st.radio("Provider", options=["groq", "openrouter"], index=0)
    model = st.text_input("Model", value=DEFAULT_MODELS[provider])
    st.caption(f"Requires `{KEY_VAR[provider]}` in your `.env` file.")

# --- Main --------------------------------------------------------------
st.title("🔎 Research Agent")

query = st.text_input("Research question", placeholder="What is the current state of fusion energy?")
run_clicked = st.button("Research", type="primary")

if run_clicked:
    if not os.getenv(KEY_VAR[provider]):
        st.error(f"{KEY_VAR[provider]} is not set. Add it to your .env file and restart.")
    elif not query.strip():
        st.warning("Enter a research question first.")
    else:
        with st.spinner("Researching..."):
            try:
                parsed, steps, raw_text = run_research(query, provider, model)
            except Exception as e:
                st.error(f"Agent run failed: {e}")
                parsed, steps, raw_text = None, [], ""

        st.session_state["parsed"] = parsed
        st.session_state["steps"] = steps
        st.session_state["raw_text"] = raw_text

# --- Results -------------------------------------------------------------
parsed = st.session_state.get("parsed")
steps = st.session_state.get("steps")
raw_text = st.session_state.get("raw_text")

if parsed:
    st.header(parsed.topic)
    st.write(parsed.summary)

    if parsed.sources:
        st.subheader("Sources")
        for src in parsed.sources:
            st.markdown(f"- {src}")

    if parsed.tools_used:
        st.subheader("Tools used")
        # st.pills is a *selection* widget (built for picking an option, not
        # displaying tags) -- with nothing pre-selected it just renders a row
        # of empty, unhighlighted buttons. Plain chips are the right fit for
        # "these tools were used," a purely informational list.
        st.write(" ".join(f"`{t}`" for t in parsed.tools_used))
elif raw_text:
    st.warning("Couldn't parse structured output -- showing raw response instead.")
    st.write(raw_text)

if steps:
    with st.expander("What the agent did"):
        for i, (action, observation) in enumerate(steps, start=1):
            st.markdown(f"**Step {i}: `{action.tool}`**")
            st.code(str(action.tool_input), language="text")
            obs_text = str(observation)
            truncated = obs_text[:500] + ("..." if len(obs_text) > 500 else "")
            st.text(truncated)
            st.divider()

if os.path.exists("research_output.txt"):
    with open("research_output.txt", "rb") as f:
        st.download_button(
            "Download research_output.txt",
            data=f.read(),
            file_name="research_output.txt",
            mime="text/plain",
        )
