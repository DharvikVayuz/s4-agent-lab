"""Tools available to the research agent.

Each tool is wrapped in LangChain's `Tool` class with a description written
for the *model*, not for a human reader -- the agent decides which tool to
call by reading this text, so being specific here actually changes behavior.
"""

import time
from datetime import datetime

import wikipedia
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool

# The `wikipedia` package's default User-Agent ("wikipedia
# (https://github.com/goldsmith/Wikipedia/)") gets soft-blocked by Wikimedia,
# which then returns an HTML/empty response instead of JSON -- surfacing as a
# confusing `requests.exceptions.JSONDecodeError` deep in the tool call, with
# nothing in the traceback pointing at the real cause. Setting a real
# User-Agent fixes it.
wikipedia.USER_AGENT = "s4-agent-lab-research-agent/1.0 (university lab; contact: set-your-email-here)"

# --- 1. Web search -----------------------------------------------------
# DuckDuckGoSearchRun needs no API key, which is why it's the default web
# search tool for a teaching lab (no signup friction for students).
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="web_search",
    func=search.run,
    description=(
        "Search the live web via DuckDuckGo. Use this for current events, "
        "recent facts, or anything that might have changed since a model's "
        "training data (prices, news, versions, dates). Input should be a "
        "concise search query, not a full question."
    ),
)

# --- 2. Wikipedia --------------------------------------------------------
# top_k_results=1 and a small char cap keep the tool's output short enough
# that it doesn't blow up the agent's context on every call.
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)


def wikipedia_lookup(query: str) -> str:
    """Look up `query` on Wikipedia, retrying a couple of times.

    Even with a real User-Agent set above, Wikimedia's API intermittently
    returns a non-JSON response to a normal-looking request (no clear
    pattern -- same query, same headers, succeeds most of the time and
    fails occasionally). A plain `WikipediaQueryRun` tool lets that
    exception propagate straight up through AgentExecutor and kill the
    whole run, which is a rough failure mode for a five-tool-call research
    loop. Retrying a couple of times, and falling back to a message the
    agent can react to (e.g. by trying web_search instead) rather than a
    crash, is worth the extra few lines.
    """
    last_error = None
    for attempt in range(3):
        try:
            return wiki_wrapper.run(query)
        except Exception as e:  # noqa: BLE001 -- deliberately broad, see docstring
            last_error = e
            time.sleep(0.5)
    return f"Wikipedia lookup failed after retries ({last_error}). Try web_search instead."


wiki_tool = Tool(
    name="wikipedia",
    func=wikipedia_lookup,
    description=(
        "Look up a stable, encyclopedic fact from Wikipedia -- definitions, "
        "history, biographies, well-established background info. Prefer this "
        "over web_search when the topic is unlikely to have changed recently. "
        "Input should be a search term (e.g. a person, place, or concept), "
        "not a full question."
    ),
)


# --- 3. Save to file ------------------------------------------------------
def save_to_txt(data: str, filename: str = "research_output.txt") -> str:
    """Append a timestamped research summary to a local text file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"--- Research Output ({timestamp}) ---\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted)

    return f"Data successfully saved to {filename}"


save_tool = Tool(
    name="save_research",
    func=save_to_txt,
    description=(
        "Save the final research findings to a local text file for the "
        "user to download later. Call this once, near the end, after you've "
        "gathered enough information to write a useful summary. Input should "
        "be the text you want saved (the filename defaults to "
        "research_output.txt, you don't need to specify it)."
    ),
)

TOOLS = [search_tool, wiki_tool, save_tool]
