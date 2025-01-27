"""
main.py

Command-line entry point to run the entire RAG pipeline end-to-end, with optional vLLM inference.

Example usage:
  python main.py \
      --question "How to create a pipeline object?" \
      --use_reranker \
      --use_vllm
"""
import time
import os
import argparse
import torch
import logging
import sys
from typing import Optional, List, Tuple, Callable

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# vLLM imports (only used if --use_vllm is True)
try:
    from vllm import LLM, SamplingParams
except ImportError:
    # If vllm is not installed, handle gracefully
    LLM = None
    SamplingParams = None

# Ensure TRANSFORMERS_CACHE is set
os.environ.setdefault("TRANSFORMERS_CACHE", "/gpfs/scratch/wz1492/models")

# Local imports from src/
from src.data_loading import load_documents_from_hf, split_documents
from src.embeddings import get_embedding_model
from src.indexing import build_faiss_index
from src.reranking import get_reranker
from src.rag_pipeline import answer_with_rag  # This function must handle a flexible "llm" callable.
from langchain_community.tools import DuckDuckGoSearchResults
from typing import Any

def load_models_for_web(
    llm_model_name: str,
    use_vllm: bool,
    reranker_model_name: Optional[str] = None,
    use_reranker: bool = False
) -> Tuple[Optional[Any], Optional[Callable]]:
    """
    Load models needed for web search: reranker (optional) and LLM.
    Returns:
        (reranker, llm_inference_fn)
    """
    reranker = None
    if use_reranker:
        logging.info("Loading reranker model...")
        if not reranker_model_name:
            logging.warning("use_reranker=True but no model name provided. Proceeding without reranker.")
        else:
            from src.reranking import get_reranker
            reranker = get_reranker(reranker_model_name)

    # Load LLM
    if use_vllm:
        if LLM is None or SamplingParams is None:
            raise ImportError("vLLM is not installed. Please install with 'pip install vllm'.")
        llm_inference_fn = get_vllm_inference_fn(llm_model_name)
    else:
        llm_inference_fn = get_hf_pipeline_llm(llm_model_name)

    return reranker, llm_inference_fn

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run an end-to-end RAG pipeline.")
    parser.add_argument(
        "--question",
        type=str,
        default="How to create a pipeline object?",
        help="The user query or question to pass to the RAG pipeline."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="m-ric/huggingface_doc",
        help="The path to the dataset on Hugging Face."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Which dataset split to load (train, test, validation, etc.)."
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="thenlper/gte-small",
        help="Hugging Face model name for the embedding model."
    )
    parser.add_argument(
        "--llm_model_name",
        type=str,
        default="HuggingFaceH4/zephyr-7b-beta",
        help="Hugging Face model name for the LLM reader."
    )
    parser.add_argument(
        "--use_reranker",
        action="store_true",
        help="If set, use a re-ranker to rerank retrieved documents."
    )
    parser.add_argument(
        "--reranker_model_name",
        type=str,
        default="colbert-ir/colbertv2.0",
        help="Hugging Face model name for the re-ranker."
    )
    parser.add_argument(
        "--num_retrieved_docs",
        type=int,
        default=10,
        help="Number of documents to retrieve initially."
    )
    parser.add_argument(
        "--num_docs_final",
        type=int,
        default=10,
        help="Number of documents to use after optional reranking."
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="If set, use vLLM for faster inference instead of HF pipeline."
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Use web search instead of the dataset for retrieval."
    )
    return parser.parse_args()


def load_and_process_documents(
    dataset_path: str,
    split: str,
    embedding_model_name: str,
    chunk_size: int = 512,
    overlap_ratio: float = 0.1,
) -> List[dict]:
    """
    Loads the dataset from Hugging Face, splits documents into chunks,
    and returns the processed documents.
    """
    logging.info("Loading dataset from Hugging Face...")
    try:
        raw_docs = load_documents_from_hf(dataset_path=dataset_path, split=split)
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    logging.info("Splitting documents into chunks...")
    docs_processed = split_documents(
        raw_docs,
        chunk_size=chunk_size,
        overlap_ratio=overlap_ratio,
        tokenizer_name=embedding_model_name
    )
    return docs_processed


def get_vllm_inference_fn(
    model_name: str,
    temperature: float = 0.2,
    max_new_tokens: int = 500,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> Callable[[str], str]:
    """
    Returns a callable that uses vLLM to generate text from a prompt.
    This callable is compatible with the 'answer_with_rag' function
    if it calls the function with a single string prompt.

    NOTE: You must have installed vllm separately to use this.

    Args:
        model_name (str): Name of the model to load with vLLM.
        temperature (float): Sampling temperature.
        max_new_tokens (int): Maximum tokens to generate.
        top_p (float): Nucleus sampling.
        repetition_penalty (float): Repetition penalty for decoding.

    Returns:
        Callable[[str], str]: A function that takes a prompt and returns the generation.
    """
    if LLM is None or SamplingParams is None:
        raise ImportError("vllm is not installed or failed to import.")

    logging.info("Initializing vLLM engine. This may take a moment...")
    llm_engine = LLM(
        model=model_name,
        tensor_parallel_size=1,  # Adjust based on how many GPUs you'd like to use
        dtype="float16"          # Or bfloat16, depending on hardware
    )

    # Prepare typical decoding parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )

    def _vllm_inference(prompt: str) -> str:
        outputs = llm_engine.generate([prompt], sampling_params)
        # vLLM returns a list of RequestOutput, each containing multiple outputs for that prompt
        return outputs[0].outputs[0].text

    return _vllm_inference


def get_hf_pipeline_llm(
    llm_model_name: str,
    temperature: float = 0.2,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.1
) -> Callable[[str], str]:
    """
    Returns a callable that uses the Hugging Face Transformers pipeline
    for text-generation.

    Args:
        llm_model_name (str): Name of the model to load from HF.
        temperature (float): Sampling temperature.
        max_new_tokens (int): Maximum tokens to generate.
        repetition_penalty (float): Repetition penalty for decoding.

    Returns:
        Callable[[str], str]: A function that takes a prompt and returns the generation.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(llm_model_name, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_pipe = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        return_full_text=False,
        max_new_tokens=max_new_tokens,
    )

    def _hf_inference(prompt: str) -> str:
        outputs = hf_pipe(prompt)
        # HF pipeline returns a list of dicts with "generated_text" field
        return outputs[0]["generated_text"]

    return _hf_inference


def load_models(
    embedding_model_name: str,
    llm_model_name: str,
    use_vllm: bool,
    reranker_model_name: Optional[str] = None,
    use_reranker: bool = False
) -> Tuple:
    """
    Loads the embedding model, the optional reranker, and returns a text-generation
    callable for either vLLM or HF pipeline.

    Returns:
        (embedding_model, reranker, llm_inference_fn)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Embedding model
    logging.info("Loading embedding model...")
    embedding_model = get_embedding_model(
        model_name=embedding_model_name,
        device=device,
        normalize_embeddings=True
    )

    # Optional reranker
    reranker = None
    if use_reranker:
        logging.info("Loading reranker model...")
        if not reranker_model_name:
            logging.warning(
                "use_reranker=True but no reranker_model_name provided. Proceeding without reranker."
            )
        else:
            reranker = get_reranker(reranker_model_name)

    # Choose between vLLM and HF pipeline
    if use_vllm:
        if LLM is None or SamplingParams is None:
            raise ImportError(
                "You must install vllm (pip install vllm) to use --use_vllm."
            )
        logging.info("Loading LLM with vLLM...")
        llm_inference_fn = get_vllm_inference_fn(llm_model_name)
    else:
        logging.info("Loading LLM with Hugging Face pipeline...")
        llm_inference_fn = get_hf_pipeline_llm(llm_model_name)

    return embedding_model, reranker, llm_inference_fn


def main() -> None:
    """
    Main execution function for the RAG pipeline with statistics reporting.
    """
    # Basic logger configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    args = parse_args()
    start_total = time.time()
    stats = {
        'source': None,
        'model': args.llm_model_name,
        'retrieved_docs': 0,
        'used_docs': 0,
        'retrieval_time': 0.0,
        'generation_time': 0.0,
        'total_time': 0.0,
        'reranker_used': args.use_reranker,
        'vllm_used': args.use_vllm
    }

    logging.info("Starting RAG pipeline...")

    if args.web:
        logging.info("Using web search for retrieval.")
        stats['source'] = "web"
        
        try:
            # Web search implementation
            start_retrieval = time.time()
            
            # Perform web search
            search = DuckDuckGoSearchResults(
                output_format="list",
                max_results=args.num_retrieved_docs,
            )

            search_results = search.invoke(args.question)
            
            processed_docs = [
                {'content': content}
                for doc in search_results
                if (content := f"{doc['title']} {doc['snippet']}".strip())
            ]
            stats['retrieved_docs'] = len(processed_docs)

            # Reranking
            if args.use_reranker:
                from src.reranking import get_reranker
                reranker = get_reranker(args.reranker_model_name)
                doc_contents = [doc['content'] for doc in processed_docs if doc['content'].strip()]
                
                if not doc_contents:
                    logging.warning("No document content available for reranking.")
                    used_docs = processed_docs[:args.num_docs_final]
                else:
                    reranked_results = reranker.rerank(args.question, doc_contents, k=args.num_docs_final)
                    
                    if not reranked_results:
                        logging.warning("Reranker returned no results. Using original retrieved docs.")
                        used_docs = processed_docs[:args.num_docs_final]
                    else:
                        reranked_indices = [result['rank'] for result in reranked_results]
                        used_docs = [processed_docs[i] for i in reranked_indices]
            else:  # Handle case when reranker is not used
                used_docs = processed_docs[:args.num_docs_final]

            stats['used_docs'] = len(used_docs)
            stats['retrieval_time'] = time.time() - start_retrieval

            # Load LLM
            if args.use_vllm:
                llm_inference_fn = get_vllm_inference_fn(args.llm_model_name)
            else:
                llm_inference_fn = get_hf_pipeline_llm(args.llm_model_name)

            # Generate answer
            start_generation = time.time()
            context = "\n\n".join([doc['content'] for doc in used_docs])
            prompt = f"Question: {args.question}\nContext: {context}\nAnswer:"
            answer = llm_inference_fn(prompt)
            stats['generation_time'] = time.time() - start_generation

        except ImportError:
            logging.error("Web search requires additional packages:")
            logging.error("pip install langchain-community duckduckgo-search")
            sys.exit(1)

    else:
        # Dataset-based RAG implementation
        stats['source'] = f"dataset ({args.dataset_path})"
        start_retrieval = time.time()
        
        # Load and process documents
        docs_processed = load_and_process_documents(
            args.dataset_path,
            args.split,
            args.embedding_model_name
        )
        
        # Load models
        embedding_model, reranker, llm_inference_fn = load_models(
            args.embedding_model_name,
            args.llm_model_name,
            args.use_vllm,
            args.reranker_model_name,
            args.use_reranker
        )
        
        # Build index
        knowledge_db = build_faiss_index(docs_processed, embedding_model)
        stats['retrieval_time'] = time.time() - start_retrieval

        # Generate answer
        start_generation = time.time()
        answer, used_docs = answer_with_rag(
            question=args.question,
            llm=llm_inference_fn,
            knowledge_index=knowledge_db,
            use_reranker=args.use_reranker,
            reranker=reranker,
            num_retrieved_docs=args.num_retrieved_docs,
            num_docs_final=args.num_docs_final
        )
        stats['generation_time'] = time.time() - start_generation
        stats['retrieved_docs'] = args.num_retrieved_docs
        stats['used_docs'] = len(used_docs)

    # Final statistics calculations
    stats['total_time'] = time.time() - start_total
    
    # Print results
    logging.info("\n" + "=" * 40 + " ANSWER " + "=" * 40)
    logging.info(answer)
    
    logging.info("\n" + "=" * 40 + " STATISTICS " + "=" * 38)
    logging.info(f"| {'Metric':<25} | {'Value':<50} |")
    logging.info("|---------------------------|----------------------------------------------------|")
    logging.info(f"| {'Data Source':<25} | {stats['source']:<50} |")
    logging.info(f"| {'LLM Model':<25} | {stats['model']:<50} |")
    logging.info(f"| {'Documents Retrieved':<25} | {stats['retrieved_docs']:<50} |")
    logging.info(f"| {'Documents Used':<25} | {stats['used_docs']:<50} |")
    logging.info(f"| {'Reranker Used':<25} | {'Yes' if stats['reranker_used'] else 'No':<50} |")
    logging.info(f"| {'vLLM Acceleration':<25} | {'Yes' if stats['vllm_used'] else 'No':<50} |")
    logging.info(f"| {'Retrieval Time (s)':<25} | {stats['retrieval_time']:.2f}{'*' if args.web else '':<49} |")
    logging.info(f"| {'Generation Time (s)':<25} | {stats['generation_time']:.2f}{'*' if args.web else '':<49} |")
    logging.info(f"| {'Total Time (s)':<25} | {stats['total_time']:.2f}{'*' if args.web else '':<49} |")
    if args.web:
        logging.info("| " + "-"*84 + " |")
        logging.info("| * Web search timing breakdown includes different components than dataset RAG |")
    logging.info("=" * 88 + "\n")

if __name__ == "__main__":
    main()