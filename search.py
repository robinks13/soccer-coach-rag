"""Inspect the passages selected for a question without running Ollama."""

import sys

from retrieval import FootballRetriever, source_label


# Show retrieval results directly, without waiting for text generation.
def main() -> None:
    query = " ".join(sys.argv[1:]).strip() or "Propose-moi un exercice de finition pour U17"
    print(f"Recherche : {query}\n")
    for index, doc in enumerate(FootballRetriever().retrieve(query, limit=10), 1):
        print(f"{index}. {source_label(doc)}")
        print(
            f"   Public : {doc.metadata.get('age_group')} | "
            f"Type : {doc.metadata.get('document_type')} | "
            f"Thèmes : {doc.metadata.get('themes')}"
        )
        preview = doc.page_content[:260].replace("\n", " ")
        print(f"   {preview}...\n")


if __name__ == "__main__":
    main()
