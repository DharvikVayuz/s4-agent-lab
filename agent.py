"""LLM, prompt, output schema, and the agent itself.

Importable (`from agent import run_research`) and runnable directly:

    python agent.py
"""

import json
import os

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
# create_tool_calling_agent / AgentExecutor moved to langchain_classic as of
# langchain 1.0 (the new langchain.agents is LangGraph-based instead).
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

from tools import TOOLS

load_dotenv()

# --- Provider defaults -----------------------------------------------------
# Both ChatGroq and ChatOpenRouter implement the same LangChain chat-model
# interface, so nothing below this function needs to know which one is active.
DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "openrouter": "openrouter/free",
}


def get_llm(provider: str = "groq", model: str | None = None):
    """Return a chat model instance for the given provider.

    `provider` is read from LLM_PROVIDER if not passed explicitly. API keys
    are read from the environment (via python-dotenv) -- never hard-coded.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
    model = model or DEFAULT_MODELS.get(provider)

    if provider == "groq":
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
        # disable_streaming="tool_calling": Groq's gpt-oss models have a known bug
        # (groq.APIError: "attempted to call tool 'json' which was not in
        # request.tools") when AgentExecutor's default streaming call path is used
        # together with bound tools. Falling back to a plain (non-streamed) call
        # only when tools are involved avoids it without giving up streaming
        # elsewhere.
        return ChatGroq(model=model, api_key=api_key, disable_streaming="tool_calling")

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set. Add it to your .env file.")
        return ChatOpenRouter(model=model, api_key=api_key)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Use 'groq' or 'openrouter'.")


# --- Structured output -------------------------------------------------
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# --- Prompt --------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research assistant. Use the tools available to you to "
            "gather accurate, up-to-date information, then answer the user's "
            "query. One or two tool calls is usually enough -- as soon as you "
            "have enough information to answer, stop calling tools.\n\n"
            "When you're ready to answer, call save_research once with your "
            "findings, then respond with your final answer as your very next "
            "message. Do not call any more tools after that.\n\n"
            "Your final answer must be wrapped in exactly this JSON format, "
            "with no other text before or after it:\n{format_instructions}",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())


def build_executor(provider: str = "groq", model: str | None = None) -> AgentExecutor:
    llm = get_llm(provider, model)
    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=True,
    )


class StepRecorder(BaseCallbackHandler):
    """Records each (action, observation) pair as tool calls happen.

    AgentExecutor's own `return_intermediate_steps` only reflects the list if
    `invoke()` returns normally -- on the Groq/gpt-oss bug handled below,
    invoke() raises partway through, so that return value never comes back
    even though several tool calls already ran. Recording via callbacks as
    they happen means we still have the trail when that occurs.
    """

    def __init__(self):
        self.steps: list[list] = []

    def on_agent_action(self, action, **kwargs):
        self.steps.append([action, None])

    def on_tool_end(self, output, **kwargs):
        if self.steps and self.steps[-1][1] is None:
            self.steps[-1][1] = output


def _recover_groq_json_tool_bug(error: Exception) -> str | None:
    """Recover from a known Groq + gpt-oss quirk, or return None if this isn't it.

    gpt-oss models sometimes emit their final structured answer as a call to
    a pseudo-tool literally named "json" (an artifact of the model's Harmony
    response format) instead of plain text. Groq's API rejects that with a
    400 ("attempted to call tool 'json' which was not in request.tools")
    since "json" isn't one of our real tools -- but the correct answer is
    sitting right there in the error body's `failed_generation` field. If
    that's what happened, pull it back out instead of treating this as a
    hard failure. See: https://community.groq.com/t/tool-calling-errors-on-both-gpt-oss-models/406
    """
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return None
    failed_generation = (body.get("error") or {}).get("failed_generation")
    if not failed_generation:
        return None
    try:
        payload = json.loads(failed_generation)
        return json.dumps(payload.get("arguments", payload))
    except Exception:
        return None


def run_research(
    query: str, provider: str = "groq", model: str | None = None
) -> tuple[ResearchResponse | None, list, str]:
    """Run the agent on `query` and return (parsed_response, intermediate_steps, raw_text).

    parsed_response is None if the model's output couldn't be parsed into
    ResearchResponse -- callers should fall back to displaying raw_text
    rather than crashing, since students will hit malformed output often.
    """
    executor = build_executor(provider, model)
    recorder = StepRecorder()
    try:
        raw = executor.invoke({"query": query, "chat_history": []}, config={"callbacks": [recorder]})
        raw_text = raw.get("output", "")
        steps = raw.get("intermediate_steps") or recorder.steps
    except Exception as e:
        recovered = _recover_groq_json_tool_bug(e)
        if recovered is None:
            raise
        raw_text, steps = recovered, recorder.steps

    try:
        parsed = parser.parse(raw_text)
    except Exception:
        parsed = None

    return parsed, steps, raw_text


if __name__ == "__main__":
    import sys

    # Windows terminals default to a codepage (cp1252 etc.) that can't
    # encode every Unicode character a model might output (em dashes,
    # non-breaking hyphens...). Widen stdout to utf-8 so a stray character
    # doesn't crash the whole run after the agent already did the work.
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    provider = os.getenv("LLM_PROVIDER", "groq")
    query = input("Research query: ").strip() or "What is LangChain?"

    parsed, steps, raw_text = run_research(query, provider)

    print("\n" + "=" * 60)
    if parsed:
        print(f"Topic: {parsed.topic}")
        print(f"Summary: {parsed.summary}")
        print(f"Sources: {parsed.sources}")
        print(f"Tools used: {parsed.tools_used}")
    else:
        print("Could not parse structured output. Raw response:")
        print(raw_text)
    print("=" * 60)
    print(f"\n{len(steps)} intermediate step(s) taken.")
