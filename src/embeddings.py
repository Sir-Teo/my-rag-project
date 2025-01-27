"""
embeddings.py

Creates an embedding model (for query/doc embedding) using HuggingFaceEmbeddings.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

def get_embedding_model(
    model_name: str = "thenlper/gte-small",
    device: str = "cuda",
    normalize_embeddings: bool = True,
    multi_process: bool = True
):
    """
    Creates and returns a HuggingFaceEmbeddings instance.

    :param model_name: The name/path of the sentence-transformers model
    :param device: "cuda" or "cpu"
    :param normalize_embeddings: Whether to normalize embeddings (needed for cosine similarity)
    :param multi_process: Whether to enable parallel encoding
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize_embeddings},
        multi_process=multi_process
    )
    return embedding_model
