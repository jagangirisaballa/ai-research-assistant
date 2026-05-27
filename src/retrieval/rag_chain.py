"""RAG chains. Patterns: RetrievalQA Chain, Conversational Retrieval Chain w/ Memory."""
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI

from src.config import get_settings


def build_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Pattern: LLM Factory."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=temperature,
        api_key=settings.openai_api_key,
    )


def build_retrieval_qa(vectorstore: VectorStore, k: int = 4) -> RetrievalQA:
    """Pattern: RetrievalQA Chain — stateless, single-shot Q&A over a vector store."""
    return RetrievalQA.from_chain_type(
        llm=build_llm(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        return_source_documents=True,
    )


def build_conversational_chain(vectorstore: VectorStore, k: int = 4) -> ConversationalRetrievalChain:
    """Pattern: Conversational Retrieval Chain — multi-turn with buffered chat memory."""
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    return ConversationalRetrievalChain.from_llm(
        llm=build_llm(),
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        memory=memory,
        return_source_documents=True,
    )
