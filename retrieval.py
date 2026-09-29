"""Find relevant passages in the local football PDF collection."""

import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


ROOT_DIR = Path(__file__).resolve().parent
DB_DIR = ROOT_DIR / "chroma_db_v1"
COLLECTION_NAME = "football_v1"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

STOPWORDS = {
    "avec", "aux", "ces", "dans", "des", "donne", "donner", "pour", "propose",
    "the", "training", "une", "give", "moi", "that", "from", "this", "coach", "exercice",
}
SYNONYMS = {
    "finition": ("finishing", "shooting", "tir", "frappe", "scoring", "goals"),
    "finishing": ("finition", "shooting", "tir", "frappe", "scoring", "buts"),
    "transition": ("contre attaque", "counter attack", "counterattack", "turnover"),
    "echauffement": ("warm up", "activation", "mobilite"),
    "gardien": ("goalkeeper", "goalkeeping", "keeper"),
    "vitesse": ("speed", "sprint", "acceleration"),
    "agilite": ("agility", "coordination", "motricite"),
    "passe": ("passing", "pass", "transmission"),
    "dribble": ("dribbling", "conduite", "elimination"),
}
ORGANIZATIONS = (
    "Football Australia", "US Youth Soccer", "US Soccer", "Belgian FA",
    "Canada Soccer", "Scottish FA", "Irish FA", "MA Youth Soccer", "The FA",
    "CONCACAF", "UEFA", "DFB", "FIFA",
)


# Remove accents and normalize whitespace for matching across languages.
def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", value.lower()).strip()


# Keep meaningful words for the lightweight keyword search.
def tokens(value: str) -> list[str]:
    return [
        word for word in re.findall(r"[a-z0-9]+", normalize(value))
        if len(word) > 2 and word not in STOPWORDS
    ]


# Extract an age category such as U17 from the coach's question.
def requested_age(query: str) -> int | None:
    match = re.search(r"\bu\s*[-_]?\s*(\d{1,2})\b", normalize(query))
    return int(match.group(1)) if match else None


# Add common French and English football synonyms before searching.
def expand_query(query: str) -> str:
    normalized = normalize(query)
    additions = []
    for key, values in SYNONYMS.items():
        if key in normalized:
            additions.extend(values)
    age = requested_age(query)
    if age is not None:
        if age <= 12:
            additions.extend(("grassroots", "children"))
        elif age <= 19:
            additions.extend(("youth", "formation", "development phase"))
        else:
            additions.extend(("senior", "adult"))
    return " ".join((query, *additions))


# Identify the coaching topics that should influence ranking.
def requested_themes(query: str) -> set[str]:
    value = normalize(query)
    themes = set()
    if any(term in value for term in ("finition", "finishing", "shooting", "tir", "frappe", "but")):
        themes.add("finition")
    if any(term in value for term in ("transition", "contre attaque", "counter attack")):
        themes.add("transition")
    if any(term in value for term in ("gardien", "goalkeeper", "goalkeeping")):
        themes.add("gardien")
    if any(term in value for term in ("vitesse", "speed", "sprint", "agilite", "agility", "physique")):
        themes.add("athletique")
    return themes


# Convert a technical filename into a readable publication title.
def source_title(doc: Document) -> str:
    raw = doc.metadata.get("source_name", Path(doc.metadata.get("source", "Document inconnu")).name)
    title = re.sub(r"^\d+_", "", Path(raw).stem)
    title = re.sub(r"[_-]+", " ", title).strip()
    title = re.sub(r"\s+Methodology$", "", title, flags=re.IGNORECASE)
    for organization in ORGANIZATIONS:
        if title.lower().startswith(organization.lower()):
            subject = title[len(organization):].strip()
            return f"{organization} — {subject}" if subject else organization
    return title


# Attach the one-based PDF page number to a readable source title.
def source_label(doc: Document) -> str:
    page = doc.metadata.get("page")
    page_number = page + 1 if isinstance(page, int) else "?"
    return f"{source_title(doc)}, p. {page_number}"


class FootballRetriever:
    # Load the index and the text needed for keyword ranking once.
    def __init__(self) -> None:
        if not DB_DIR.exists():
            raise FileNotFoundError("Base absente. Lancez d'abord : python ingest.py")

        embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_db = Chroma(
            persist_directory=str(DB_DIR),
            embedding_function=embedding,
            collection_name=COLLECTION_NAME,
        )
        if self.vector_db._collection.count() == 0:
            raise RuntimeError("La base est vide. Lancez : python ingest.py")

        # Keep text in memory so the small collection can be searched by keywords too.
        stored = self.vector_db._collection.get(include=["documents", "metadatas"])
        self.documents = [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(stored["documents"], stored["metadatas"])
        ]
        self.term_counts = [
            Counter(tokens(doc.page_content + " " + str(doc.metadata.get("title", ""))))
            for doc in self.documents
        ]
        self.document_frequency = Counter()
        for counts in self.term_counts:
            self.document_frequency.update(counts.keys())

    # Rank passages by informative words shared with the expanded query.
    def keyword_results(self, query: str, limit: int = 40) -> list[Document]:
        query_terms = tokens(expand_query(query))
        ranked = []
        total = len(self.documents)
        for index, counts in enumerate(self.term_counts):
            length_norm = 1.0 + math.log(1 + sum(counts.values()))
            score = sum(
                (1 + math.log(counts[term]))
                * (math.log((total + 1) / (self.document_frequency[term] + 1)) + 1)
                / length_norm
                for term in query_terms if counts[term]
            )
            if score:
                ranked.append((score, self.documents[index]))
        ranked.sort(key=lambda result: result[0], reverse=True)
        return [doc for _, doc in ranked[:limit]]

    # Combine both searches, then filter by age, topic, and source diversity.
    def retrieve(self, query: str, limit: int = 6) -> list[Document]:
        expanded = expand_query(query)
        vector_results = self.vector_db.similarity_search(expanded, k=40)
        keyword_results = self.keyword_results(query)
        scores = defaultdict(float)
        candidates = {}

        # Reciprocal rank fusion lets semantic and keyword search contribute equally.
        for results in (vector_results, keyword_results):
            for rank, doc in enumerate(results, 1):
                key = self._key(doc)
                candidates[key] = doc
                scores[key] += 1 / (40 + rank)

        age = requested_age(query)
        themes = requested_themes(query)
        query_terms = set(tokens(expanded))
        for key, doc in candidates.items():
            metadata = doc.metadata
            if age is not None:
                low = int(metadata.get("age_min", 0))
                high = int(metadata.get("age_max", 99))
                if low <= age <= high:
                    scores[key] += 0.025
                elif low != 0 or high != 99:
                    scores[key] -= 0.055
            overlap = len(query_terms & set(tokens(doc.page_content)))
            scores[key] += min(overlap, 8) * 0.003
            if metadata.get("document_type") in {"seance", "guide_pratique"}:
                scores[key] += 0.008
            doc_themes = set(str(metadata.get("themes", "general")).split(","))
            if themes:
                scores[key] += 0.035 if themes & doc_themes else -0.025

        ranked = sorted(candidates.values(), key=lambda doc: scores[self._key(doc)], reverse=True)
        selected = []
        per_source = Counter()
        seen_pages = set()
        for doc in ranked:
            source = doc.metadata.get("source_name", doc.metadata.get("source", "inconnu"))
            page_key = (source, doc.metadata.get("page"))
            if page_key in seen_pages or per_source[source] >= 3:
                continue
            if age is not None:
                low = int(doc.metadata.get("age_min", 0))
                high = int(doc.metadata.get("age_max", 99))
                if (low != 0 or high != 99) and not (low <= age <= high):
                    continue
            doc_themes = set(str(doc.metadata.get("themes", "general")).split(","))
            if themes and not (themes & doc_themes):
                continue
            selected.append(doc)
            per_source[source] += 1
            seen_pages.add(page_key)
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _key(doc: Document) -> tuple:
        return (doc.metadata.get("source"), doc.metadata.get("page"), doc.metadata.get("chunk_index"))
