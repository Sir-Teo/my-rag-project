"""
rag_pipeline.py

Combines everything into a single pipeline function:
1. Retrieve documents from the index
2. (Optionally) rerank them
3. Build a prompt
4. Pass prompt to the LLM to generate an answer
"""

from typing import Optional, Tuple, List
from transformers import Pipeline
from langchain.vectorstores import FAISS
from .reranking import rerank_docs
from .indexing import similarity_search


def build_rag_prompt_template():
    """
    Return a text template or a chat template for RAG.
    In this example, we keep it simple: user query plus context.

    For advanced usage with chat models, you'd adapt to the chat format (role-based).
    """
    template = (
        "You are a helpful assistant. Use only the context below to answer the user's question. \n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer concisely. If you cannot answer from the context, say 'Not enough information.'\n"
    )
    return template


def answer_with_rag(
    question: str,
    llm: Pipeline,
    knowledge_index: FAISS,
    use_reranker: bool = False,
    reranker=None,
    num_retrieved_docs: int = 10,
    num_docs_final: int = 5
) -> Tuple[str, List[str]]:
    """
    1. Retrieve top `num_retrieved_docs` from the FAISS index
    2. (Optional) rerank them down to `num_docs_final`
    3. Prompt the LLM with the final docs
    4. Return the answer and the final docs

    :param question: user query
    :param llm: Hugging Face Transformers pipeline for text-generation
    :param knowledge_index: FAISS vectorstore
    :param use_reranker: whether to use a cross-encoder reranker
    :param reranker: a ragatouille-based reranker or similar
    :param num_retrieved_docs: how many docs to fetch initially
    :param num_docs_final: how many docs to keep after reranking
    """

    # Step 1: retrieve
    docs = similarity_search(question, knowledge_index, k=num_retrieved_docs)
    doc_texts = [doc.page_content for doc in docs]

    # Step 2: rerank if requested
    if use_reranker and reranker is not None:
        doc_texts = rerank_docs(question, doc_texts, reranker=reranker, top_k=num_docs_final)
    else:
        doc_texts = doc_texts[:num_docs_final]

    # Step 3: build prompt
    # Merge final doc texts into a single context
    context_str = ""
    for i, text in enumerate(doc_texts):
        context_str += f"Document {i}:\n{text}\n\n"

    template = build_rag_prompt_template()
    final_prompt = template.format(context=context_str, question=question)

    # Step 4: get answer
    output = output = llm(final_prompt)

    return output, doc_texts
