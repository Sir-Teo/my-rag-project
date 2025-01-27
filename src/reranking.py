"""
reranking.py

Optional: uses a cross-encoder or any advanced model to rerank the top retrieved documents.
Example uses the ragatouille library with ColBERTv2.
"""

from ragatouille import RAGPretrainedModel

def get_reranker(model_name: str = "colbert-ir/colbertv2.0"):
    """
    Returns a cross-encoder-based reranker from 'ragatouille'.
    """
    reranker = RAGPretrainedModel.from_pretrained(model_name)
    return reranker


def rerank_docs(question: str, docs, reranker, top_k=5):
    """
    Rerank the retrieved docs with the cross-encoder and return top_k.
    Here, docs should be a list of (text) strings or something similar.
    """
    # ragatouille expects a list of strings; it returns a list of dicts with 'content' and 'score'
    results = reranker.rerank(question, docs, k=top_k)
    return [res["content"] for res in results]
