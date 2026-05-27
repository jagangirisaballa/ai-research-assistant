"""Tool-using agent. Pattern: Tool-Calling Agent (ReAct-style) with retrieval + web search."""
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_core.vectorstores import VectorStore

from src.config import get_settings
from src.retrieval.rag_chain import build_llm


def _make_retrieval_tool(vectorstore: VectorStore, k: int = 4) -> Tool:
    """Pattern: Retriever-as-Tool. Wraps a vector store retriever as a callable tool."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def _run(query: str) -> str:
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant passages found in indexed documents."
        return "\n\n---\n\n".join(d.page_content for d in docs)

    return Tool(
        name="document_retrieval",
        description=(
            "Search the user's indexed documents for passages relevant to a query. "
            "Use this for questions about content the user has ingested."
        ),
        func=_run,
    )


def _make_web_search_tool() -> Tool:
    """Pattern: External API Tool. Tavily web search for current-events / out-of-corpus queries."""
    # Lazy import: avoids requiring tavily key at module-import time for tests.
    from langchain_tavily import TavilySearch

    settings = get_settings()
    tavily = TavilySearch(max_results=5, tavily_api_key=settings.tavily_api_key)

    def _run(query: str) -> str:
        result = tavily.invoke({"query": query})
        return str(result)

    return Tool(
        name="web_search",
        description=(
            "Search the public web for recent or general-knowledge information not in the "
            "indexed documents. Use for current events, definitions, or external facts."
        ),
        func=_run,
    )


_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a careful research assistant. Use tools to ground your answers. Cite which tool you used."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


def build_agent(vectorstore: VectorStore) -> AgentExecutor:
    """Compose the tool-calling agent with retrieval + web search tools."""
    tools = [_make_retrieval_tool(vectorstore), _make_web_search_tool()]
    agent = create_tool_calling_agent(build_llm(), tools, _AGENT_PROMPT)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
