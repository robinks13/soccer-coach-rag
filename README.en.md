# Soccer Coach Assistant — V1

[Lire cette documentation en français](README.md).

This local RAG project answers soccer coaching questions using PDF documents. It combines semantic and keyword search, then asks Qwen through Ollama to write the answer. The documents used are listed at the end of each response.

I used GPT-5.6-sol to help develop and debug this V1.

The PDF collection can contain both English and French documents. You can ask questions in either language; the assistant responds in the detected language of your question. You can also ask explicitly to “answer in English” or “réponds en français”.

## Setup

You need Python (developed with 3.14), [Ollama](https://ollama.com/), and soccer PDFs in the `data/` directory.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
ollama pull qwen2.5:32b
python ingest.py
```

On its first run, `ingest.py` also downloads the `paraphrase-multilingual-MiniLM-L12-v2` embedding model from Hugging Face. It creates a local index in `chroma_db_v1/`.

## Usage

```bash
python chat.py
```

Example questions: `Give me a finishing exercise for U17`, `Plan a 90-minute U17 finishing session`, or `Propose un exercice de finition pour U17`.

To inspect the retrieved documents before generation:

```bash
python search.py "finishing U17"
```

## Project structure

- `ingest.py` extracts PDF text, adds metadata, and builds the index.
- `retrieval.py` searches and ranks passages.
- `chat.py` chooses a response style and queries the local model.
- `search.py` displays retrieved passages for debugging.

PDFs, local models, and Chroma databases are excluded from Git because they are large and redistribution rights must be checked for each document. On a new machine, place the PDFs in `data/` and run `python ingest.py`.

## Current limitations

Age groups and topics are inferred automatically, so broad documents may be misclassified. The language detector uses simple keywords; for short or mixed-language questions, specify the desired language explicitly. The model may still suggest plausible details that are absent from the documents; those should be marked as adaptations. Check cited PDF pages and adjust training loads before using a session on the field.
