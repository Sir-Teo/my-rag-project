import gradio as gr
import time
import logging
import sys
from typing import Optional, Callable
import torch

# Local imports from main.py and src/
from main import (
    load_documents_from_hf,
    split_documents,
    get_embedding_model,
    build_faiss_index,
    get_reranker,
    answer_with_rag,
    get_vllm_inference_fn,
    get_hf_pipeline_llm,
    DuckDuckGoSearchResults
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Global variables for caching models and index
_FAISS_INDEX = None
_EMBEDDING_MODEL = None
_RERANKER = None
_VLLM_LLM = None
_HF_LLM = None

# Constants (match those in main.py)
DATASET_PATH = "m-ric/huggingface_doc"
SPLIT = "train"
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
LLM_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
RERANKER_MODEL_NAME = "colbert-ir/colbertv2.0"

def preload_dataset_index():
    """Preload dataset and build FAISS index during startup"""
    global _FAISS_INDEX, _EMBEDDING_MODEL
    
    logging.info("Preloading dataset and building FAISS index...")
    try:
        # Load and process documents
        raw_docs = load_documents_from_hf(DATASET_PATH, SPLIT)
        docs_processed = split_documents(
            raw_docs,
            chunk_size=512,
            overlap_ratio=0.1,
            tokenizer_name=EMBEDDING_MODEL_NAME
        )
        
        # Load embedding model and build index
        _EMBEDDING_MODEL = get_embedding_model(
            EMBEDDING_MODEL_NAME,
            device="cuda" if torch.cuda.is_available() else "cpu",
            normalize_embeddings=True
        )
        _FAISS_INDEX = build_faiss_index(docs_processed, _EMBEDDING_MODEL)
        logging.info("Dataset preloaded and FAISS index built.")
        
    except Exception as e:
        logging.error(f"Failed to preload dataset: {e}")
        sys.exit(1)

def get_cached_reranker():
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = get_reranker(RERANKER_MODEL_NAME)
    return _RERANKER

def get_cached_vllm_llm():
    global _VLLM_LLM
    if _VLLM_LLM is None:
        _VLLM_LLM = get_vllm_inference_fn(LLM_MODEL_NAME)
    return _VLLM_LLM

def get_cached_hf_llm():
    global _HF_LLM
    if _HF_LLM is None:
        _HF_LLM = get_hf_pipeline_llm(LLM_MODEL_NAME)
    return _HF_LLM

def process_query(question: str, mode: str, use_reranker: bool, use_vllm: bool):
    """Process a query through the RAG pipeline"""
    start_total = time.time()
    stats = {
        'source': None,
        'model': LLM_MODEL_NAME,
        'retrieved_docs': 0,
        'used_docs': 0,
        'retrieval_time': 0.0,
        'generation_time': 0.0,
        'total_time': 0.0,
        'reranker_used': use_reranker,
        'vllm_used': use_vllm
    }

    try:
        if mode == "dataset":
            # Dataset-based RAG
            stats['source'] = f"dataset ({DATASET_PATH})"
            
            if not _FAISS_INDEX or not _EMBEDDING_MODEL:
                raise ValueError("Dataset index not preloaded")
                
            start_retrieval = time.time()
            
            # Get reranker if needed
            reranker = get_cached_reranker() if use_reranker else None
            
            # Get LLM
            llm_fn = get_cached_vllm_llm() if use_vllm else get_cached_hf_llm()
            
            # Run RAG pipeline
            start_generation = time.time()
            answer, used_docs = answer_with_rag(
                question=question,
                llm=llm_fn,
                knowledge_index=_FAISS_INDEX,
                use_reranker=use_reranker,
                reranker=reranker,
                num_retrieved_docs=10,
                num_docs_final=5
            )
            
            stats['retrieval_time'] = time.time() - start_retrieval
            stats['generation_time'] = time.time() - start_generation
            stats['retrieved_docs'] = 10
            stats['used_docs'] = len(used_docs)

        else:
            # Web-based RAG
            stats['source'] = "web search"
            start_retrieval = time.time()
            
            # Perform web search
            search = DuckDuckGoSearchResults(
                output_format="list",
                max_results=10
            )
            results = search.invoke(question)
            processed_docs = [{'content': f"{doc['title']} {doc['snippet']}".strip()} for doc in results]
            
            # Rerank if needed
            if use_reranker:
                reranker = get_cached_reranker()
                doc_contents = [d['content'] for d in processed_docs]
                reranked = reranker.rerank(question, doc_contents, k=5)
                used_docs = [processed_docs[r['rank']] for r in reranked]
            else:
                used_docs = processed_docs[:5]
                
            stats['retrieved_docs'] = len(processed_docs)
            stats['used_docs'] = len(used_docs)
            stats['retrieval_time'] = time.time() - start_retrieval
            
            # Generate answer
            llm_fn = get_cached_vllm_llm() if use_vllm else get_cached_hf_llm()
            context = "\n\n".join([d['content'] for d in used_docs])
            prompt = f"Question: {question}\nContext: {context}\nAnswer:"
            
            start_generation = time.time()
            answer = llm_fn(prompt)
            stats['generation_time'] = time.time() - start_generation

        stats['total_time'] = time.time() - start_total

    except Exception as e:
        answer = f"Error processing query: {str(e)}"
        stats = {}

    # Format statistics
    stats_text = (
        f"📊 Statistics:\n"
        f"• Data Source: {stats.get('source', 'N/A')}\n"
        f"• Retrieved Documents: {stats.get('retrieved_docs', 0)}\n"
        f"• Used Documents: {stats.get('used_docs', 0)}\n"
        f"• Reranker Used: {'Yes' if stats.get('reranker_used', False) else 'No'}\n"
        f"• vLLM Acceleration: {'Yes' if stats.get('vllm_used', False) else 'No'}\n"
        f"⏱️ Timing:\n"
        f"• Retrieval: {stats.get('retrieval_time', 0):.2f}s\n"
        f"• Generation: {stats.get('generation_time', 0):.2f}s\n"
        f"• Total: {stats.get('total_time', 0):.2f}s"
    )

    return answer, stats_text

# Preload dataset index on startup
preload_dataset_index()

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RAG Pipeline Web Interface")
    gr.Markdown("Ask questions using either our documentation dataset or live web search!")
    
    with gr.Row():
        with gr.Column():
            question = gr.Textbox(
                label="Your Question",
                placeholder="How to create a pipeline object?",
                lines=3
            )
            mode = gr.Radio(
                choices=["dataset", "web"],
                label="Knowledge Source",
                value="dataset"
            )
            with gr.Accordion("Advanced Options", open=False):
                use_reranker = gr.Checkbox(
                    label="Use Reranker",
                    value=True
                )
                use_vllm = gr.Checkbox(
                    label="Use vLLM Acceleration",
                    value=True
                )
            submit_btn = gr.Button("Ask", variant="primary")
        
        with gr.Column():
            answer = gr.Textbox(
                label="Answer",
                interactive=False,
                lines=10
            )
            stats = gr.Textbox(
                label="Statistics",
                interactive=False,
                lines=7
            )
    
    submit_btn.click(
        fn=process_query,
        inputs=[question, mode, use_reranker, use_vllm],
        outputs=[answer, stats]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)