# S4.1 Agent Lab -- Research Agent

A small LangChain research agent (web search + Wikipedia + save-to-file) with a Streamlit UI.

## Setup

1. Create a virtualenv and install deps:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in the key for whichever provider you're using:
   ```bash
   copy .env.example .env
   ```
3. Get a free key:
   - Groq (default): https://console.groq.com/keys
   - OpenRouter: https://openrouter.ai/keys
4. Run the CLI once to sanity-check the tool loop:
   ```bash
   python agent.py
   ```
5. Launch the UI:
   ```bash
   streamlit run app.py
   ```

## Notes

- Switch providers via `LLM_PROVIDER=groq|openrouter` in `.env`, or the sidebar radio in the UI.
- OpenRouter's free tier is ~20 req/min and 50 req/day on an unfunded account -- if a whole
  class hits it at once expect 429s. Groq's free tier is more forgiving for a live demo.
- `research_output.txt` is created in this folder the first time the agent saves a summary.

## Known issue: `openai/gpt-oss-120b` + Groq tool calling

Groq's gpt-oss models have a [documented bug](https://community.groq.com/t/tool-calling-errors-on-both-gpt-oss-models/406):
when the model is done reasoning and ready to answer, it sometimes tries to call a
non-existent tool literally named `"json"` instead of just replying with text (an artifact of
its Harmony response format leaking through). Groq's API rejects that with a 400.

`agent.py`'s `_recover_groq_json_tool_bug()` catches this specific error and pulls the correct
answer back out of the error body (Groq's error response embeds the JSON the model was trying
to send), so a run doesn't crash when it happens -- but it's still worth knowing this model is
less consistent about tool calling than others. If you see it happening a lot during a live
session, `qwen/qwen3.8-27b` (also free on Groq) does not have this issue.
