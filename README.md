# My RAG Project

This repository demonstrates an advanced **Retrieval-Augmented Generation (RAG)** system. It allows you to query either:

1. **A local documentation dataset** stored on [Hugging Face Datasets](https://huggingface.co/docs/datasets), or
2. **Live web search** results (via DuckDuckGo)

Then, it leverages Large Language Models (LLMs) to synthesize final answers with relevant context. You can also optionally enable **re-ranking** of retrieved documents (using [ColBERTv2](https://github.com/stanford-futuredata/ColBERT)) and **vLLM acceleration** for faster inference.

Below you'll find instructions on installation, usage, and how everything works under the hood.

---

## Key Features

- **Document Retrieval**: 
  - Retrieve documents either from a local Hugging Face dataset or from DuckDuckGo web search.
  - Uses [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search.
  
- **Embeddings**:
  - Leverages a Sentence Transformer model (e.g., [`thenlper/gte-small`](https://huggingface.co/thenlper/gte-small)) to embed text for retrieval.
  
- **Re-ranking (Optional)**:
  - Incorporates [ColBERTv2](https://huggingface.co/colbert-ir/colbertv2.0) to refine the ranking of top documents for higher accuracy.

- **Language Model Generation**:
  - Provides two backends:
    - **Hugging Face Transformers Pipeline** (default)
    - **vLLM** for optimized inference (if installed)

- **Gradio Web Interface**:
  - A user-friendly web UI (`app.py`) to interact with the RAG pipeline live.

- **Command-Line Interface**:
  - A CLI tool (`main.py`) for programmatic RAG queries.

---

## Repository Structure

```
.
├── app.py              # Gradio web application for RAG
├── main.py             # CLI interface for RAG
├── src/
│   ├── data_loading.py # Functions for loading datasets
│   ├── embeddings.py   # Functions to create embedding models
│   ├── indexing.py     # Functions to build FAISS index
│   ├── reranking.py    # Functions for re-ranking retrieved docs
│   └── rag_pipeline.py # High-level pipeline logic for QA
├── requirements.txt    # (Optional) Python dependencies
└── README.md           # This file
```

---

## Sample Output

Simply input

```bash
python main.py --question "Who is Barack Obama?" --web --use_reranker --use_vllm
```

then you will get

```
2025-01-27 15:44:23 [INFO] root: 
======================================== ANSWER ========================================
2025-01-27 15:44:23 [INFO] root:  Barack Obama was the 44th president of the United States, serving from 2009 to 2017. He represented Illinois in the U.S. Senate from 2005 to 2008 before winning the presidency. Obama was the first African American to hold the office of president.
2025-01-27 15:44:23 [INFO] root: 
======================================== STATISTICS ======================================
2025-01-27 15:44:23 [INFO] root: | Metric                    | Value                                              |
2025-01-27 15:44:23 [INFO] root: |---------------------------|----------------------------------------------------|
2025-01-27 15:44:23 [INFO] root: | Data Source               | web                                                |
2025-01-27 15:44:23 [INFO] root: | LLM Model                 | HuggingFaceH4/zephyr-7b-beta                       |
2025-01-27 15:44:23 [INFO] root: | Documents Retrieved       | 4                                                  |
2025-01-27 15:44:23 [INFO] root: | Documents Used            | 4                                                  |
2025-01-27 15:44:23 [INFO] root: | Reranker Used             | Yes                                                |
2025-01-27 15:44:23 [INFO] root: | vLLM Acceleration         | Yes                                                |
2025-01-27 15:44:23 [INFO] root: | Retrieval Time (s)        | 3.53*                                                 |
2025-01-27 15:44:23 [INFO] root: | Generation Time (s)       | 0.86*                                                 |
2025-01-27 15:44:23 [INFO] root: | Total Time (s)            | 64.99*                                                 |
2025-01-27 15:44:23 [INFO] root: | ------------------------------------------------------------------------------------ |
2025-01-27 15:44:23 [INFO] root: | * Web search timing breakdown includes different components than dataset RAG |
2025-01-27 15:44:23 [INFO] root: ========================================================================================
```

---

## Installation

1. **Clone the repository** (or download the code):

   ```bash
   git clone https://github.com/YourUsername/rag-project.git
   cd rag-project
   ```

2. **Install dependencies**:

   - Install [Faiss](https://github.com/facebookresearch/faiss). On most systems:

     ```bash
     pip install faiss-gpu   # if you have a GPU
     # or
     pip install faiss-cpu   # if you do not have a GPU
     ```

   - Install other dependencies:

     ```bash
     pip install -r requirements.txt
     ```

   - *(If you want to use vLLM acceleration)*
     ```bash
     pip install vllm
     ```

   - *(If you want to use web search)*
     ```bash
     pip install langchain-community duckduckgo-search
     ```

3. **(Optional) Set model cache location**:

   By default, the code sets:
   ```
   os.environ["TRANSFORMERS_CACHE"] = "/gpfs/scratch/wz1492/models"
   ```
   Adjust this to your desired cache path, or remove it if you’re okay with the default Hugging Face cache location.

4. **Check GPU availability**:

   - The code will automatically check for a CUDA device (`torch.cuda.is_available()`). If you do not have a GPU, it will fall back to CPU, but performance may be slower.

---

## Quickstart: Gradio Web Interface

The easiest way to get started is with the **Gradio** web app:

```bash
python app.py
```

- This will spin up a local web server on [http://0.0.0.0:7860](http://0.0.0.0:7860) (by default).
---

## Command-Line Interface (CLI)

For a more programmatic approach or to run the pipeline end-to-end without the web interface, use `main.py`.

### Example Commands

- **Basic usage (dataset-based RAG)**:
  ```bash
  python main.py --question "How to create a pipeline object?"
  ```

- **Use the re-ranker**:
  ```bash
  python main.py \
      --question "How to create a pipeline object?" \
      --use_reranker
  ```

- **Use vLLM acceleration**:
  ```bash
  python main.py \
      --question "How to create a pipeline object?" \
      --use_vllm
  ```

- **Use web search** (instead of the dataset):
  ```bash
  python main.py \
      --question "Explain the difference between GPT-3 and GPT-4" \
      --web
  ```

- **Use both re-ranker and vLLM**:
  ```bash
  python main.py \
      --question "How to create a pipeline object?" \
      --use_reranker \
      --use_vllm
  ```

### CLI Arguments

| Argument               | Default                        | Description                                                                                                  |
|------------------------|--------------------------------|--------------------------------------------------------------------------------------------------------------|
| `--question`           | "How to create a pipeline..."  | The user query.                                                                                              |
| `--dataset_path`       | "m-ric/huggingface_doc"        | Path to the dataset on Hugging Face (when not using `--web`).                                               |
| `--split`             | "train"                        | Which dataset split to load.                                                                                |
| `--embedding_model_name` | "thenlper/gte-small"        | HF model name for embeddings.                                                                               |
| `--llm_model_name`     | "HuggingFaceH4/zephyr-7b-beta" | HF model name for the language model.                                                                       |
| `--use_reranker`       | False                          | If set, use ColBERTv2 re-ranker.                                                                            |
| `--reranker_model_name`| "colbert-ir/colbertv2.0"       | Re-ranker model name to use.                                                                                |
| `--num_retrieved_docs` | 10                             | Number of documents to retrieve initially from the index.                                                   |
| `--num_docs_final`     | 10                             | Number of documents to use in final step.                                                                    |
| `--use_vllm`           | False                          | If set, use vLLM for the final generation step.                                                              |
| `--web`                | False                          | If set, use DuckDuckGo web search for retrieval instead of local dataset.                                   |

**Note**: Some arguments only matter for dataset-based retrieval (`dataset_path`, `split`), while others matter for web-based retrieval (`--web`, etc.).

---

## How It Works

1. **Data Loading**  
   - When using the **dataset** mode, `main.py` or `app.py` loads documents from a Hugging Face dataset (default: `m-ric/huggingface_doc`).  
   - Documents are split into chunks of ~512 tokens with overlap for better retrieval granularity.

2. **Embedding & Indexing**  
   - We use a SentenceTransformer or compatible model (e.g., `thenlper/gte-small`) to embed each chunk into a vector.
   - A **FAISS** index is built (or queried if pre-loaded) to efficiently find the top-k chunks relevant to the user query.

3. **(Optional) Re-ranking**  
   - If **re-ranker** is enabled, the top-k retrieved docs are re-scored using a cross-encoder (e.g. [ColBERTv2](https://huggingface.co/colbert-ir/colbertv2.0)) to reorder them for maximum relevance. 
   - The final top-n docs are then selected.

4. **Language Model Synthesis**  
   - The top-n docs are concatenated into a “context” that is fed to an LLM:
     - **HF Pipeline** (`transformers.pipeline`) using a model like [`HuggingFaceH4/zephyr-7b-beta`](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta), **or**
     - **vLLM** for faster inference, if installed.
   - The LLM generates a final answer, referencing the retrieved and (optionally) re-ranked docs.

5. **Answer + Statistics**  
   - The final answer is returned, along with various stats (retrieval time, generation time, total time, number of docs retrieved/used, etc.).

---

## Customization

- **Use your own dataset**:
  - Change the `--dataset_path` argument to point to your Hugging Face dataset or local dataset directory.
- **Change embedding model**:
  - Modify `--embedding_model_name` to use a different Sentence Transformer or embedding model.
- **Change LLM model**:
  - Set `--llm_model_name` to any Hugging Face CausalLM. 
  - Make sure the model is compatible with 4-bit/bfloat16 if you are using the pipeline code that loads it in 4-bit mode.
- **Adjust chunk size & overlap**:
  - See `split_documents(...)` in `src/data_loading.py` for chunking parameters.

---

## Potential Issues and Troubleshooting

- **CUDA errors**:
  - Make sure your environment can see your GPU: `torch.cuda.is_available()` should return `True`.
  - Check that you have the correct CUDA version installed.
- **Out of memory**:
  - Large LLMs can be memory-intensive. You may need to use smaller models, or reduce `max_new_tokens`, or switch to CPU (much slower).
- **vLLM ImportError**:
  - If you want to use `--use_vllm`, you must install `vllm` with `pip install vllm`. If not installed, you’ll get an ImportError.
- **Web search not working**:
  - Ensure you have installed `langchain-community` and `duckduckgo-search`. 
  - Some regions may block DuckDuckGo; consider a VPN or alternative search plugin.

---

## Acknowledgements

- [**FAISS**](https://github.com/facebookresearch/faiss) for vector similarity search.
- [**Hugging Face Transformers**](https://github.com/huggingface/transformers) for the LLM and embeddings pipeline.
- [**Sentence Transformers**](https://github.com/UKPLab/sentence-transformers) for the embedding models.
- [**ColBERTv2**](https://github.com/stanford-futuredata/ColBERT) for re-ranking.
- [**vLLM**](https://github.com/vllm-project/vllm) for faster inference.
- [**LangChain Community Tools**](https://github.com/hwchase17/langchain/tree/master/libs/community) for the DuckDuckGo search integration.

---

## License

This project is licensed under the [MIT License](./LICENSE) (or the license of your choice). See the [LICENSE](./LICENSE) file for details.

