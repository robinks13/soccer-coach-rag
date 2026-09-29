import re
import unicodedata
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DB_DIR = ROOT_DIR / "chroma_db_v1"
COLLECTION_NAME = "football_v1"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

THEME_ALIASES = {
    "finition": ("finition", "finishing", "shooting", "tir au but", "score goals", "scoring goals"),
    "transition": ("transition", "contre attaque", "counter attack", "counterattack"),
    "construction": ("construction", "build up", "building up", "possession"),
    "defense": ("defense", "defending", "pressing", "recuperation", "preventing"),
    "technique": ("technique", "dribbling", "dribble", "passing", "passe", "control", "controle"),
    "athletique": ("agility", "agilite", "running", "course", "speed", "vitesse", "strength", "force"),
    "gardien": ("goalkeeper", "goalkeeping", "gardien"),
    "pedagogie": ("coaching", "coach education", "pedagog", "methodology", "methodologie"),
}


# Normalize accents so French and English terms are easier to match.
def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", value.lower()).strip()


# Turn a PDF filename into a readable title for search metadata.
def clean_title(source: str) -> str:
    stem = re.sub(r"^\d+_", "", Path(source).stem)
    return re.sub(r"[_-]+", " ", stem).strip()


# Infer an approximate age range from the filename, then from the page text.
def infer_age_range(source: str, text: str) -> tuple[int, int, str]:
    source_name = normalize(Path(source).stem).replace("_", " ").replace("-", " ")
    source_ages = [int(value) for value in re.findall(r"\bu\s*(\d{1,2})\b", source_name)]
    ages_match = re.search(r"\bages?\s+(\d{1,2})\s+(\d{1,2})\b", source_name)
    if ages_match:
        source_ages.extend((int(ages_match.group(1)), int(ages_match.group(2))))
    source_ages = [age for age in source_ages if 4 <= age <= 23]
    if source_ages:
        low, high = min(source_ages), max(source_ages)
        return low, high, f"U{low}" if low == high else f"U{low}-U{high}"

    haystack = normalize(text[:2500])
    ages = [int(value) for value in re.findall(r"\bu\s*[-_]?\s*(\d{1,2})\b", haystack)]
    for low, high in re.findall(r"\b(\d{1,2})\s*(?:-|to|a)\s*(\d{1,2})\s*years?\b", haystack):
        ages.extend((int(low), int(high)))
    ages = [age for age in ages if 4 <= age <= 23]
    if ages:
        low, high = min(ages), max(ages)
        return low, high, f"U{low}" if low == high else f"U{low}-U{high}"
    if any(word in haystack for word in ("active start", "ages 5 6")):
        return 4, 6, "U5-U6"
    if "fundamentals" in haystack:
        return 6, 9, "U6-U9"
    if any(word in haystack for word in ("grassroots", "community guide", "children young people")):
        return 5, 13, "U5-U13"
    if any(word in haystack for word in ("junioren", "youth development", "youth football")):
        return 13, 19, "U13-U19"
    return 0, 99, "tous_ages"


# Classify each PDF by its likely coaching use.
def infer_doc_type(source: str) -> str:
    name = normalize(Path(source).name)
    if any(term in name for term in ("session", "lesson plan", "training session", "c te")):
        return "seance"
    if "toolkit" in name or "starter pack" in name or "booklet" in name:
        return "guide_pratique"
    if "syllabus" in name or "diploma" in name or "course" in name:
        return "formation_entraineur"
    if "charter" in name or "policy" in name or "review" in name:
        return "politique_rapport"
    if "manual" in name or "curriculum" in name or "methodology" in name:
        return "manuel_methodologique"
    return "ressource_football"


# Use a small word-frequency heuristic to label the page language.
def infer_language(text: str) -> str:
    sample = f" {normalize(text[:4000])} "
    french = sum(sample.count(f" {word} ") for word in ("le", "les", "des", "joueur", "entrainement"))
    english = sum(sample.count(f" {word} ") for word in ("the", "and", "player", "coach", "training"))
    return "fr" if french > english else "en"


# Tag pages with football themes found in their text or filename.
def infer_themes(text: str, source: str) -> str:
    haystack = normalize(f"{Path(source).name} {text}")
    themes = [theme for theme, aliases in THEME_ALIASES.items() if any(alias in haystack for alias in aliases)]
    return ",".join(themes[:6]) or "general"


# Preserve the original page number so answers can point back to the PDF.
def load_pdf_pages() -> list[Document]:
    pages = []
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages):
            pages.append(Document(
                page_content=page.extract_text() or "",
                metadata={"source": str(pdf_path), "page": page_number},
            ))
    return pages


# Build the local vector index from the PDFs in data/.
def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dossier PDF introuvable : {DATA_DIR}")
    if DB_DIR.exists():
        raise FileExistsError(f"La base V1 existe déjà : {DB_DIR}")

    print("1. Chargement des PDF...")
    documents = load_pdf_pages()
    print(f"   {len(documents)} pages extraites depuis {len(list(DATA_DIR.glob('*.pdf')))} PDF.")

    print("2. Nettoyage et ajout des métadonnées football...")
    prepared_pages = []
    for document in documents:
        text = document.page_content.replace("\x00", "").strip()
        if len(text) < 40:
            continue
        source = document.metadata.get("source", "document_inconnu.pdf")
        age_min, age_max, age_group = infer_age_range(source, text)
        title = clean_title(source)
        metadata = {
            **document.metadata,
            "title": title,
            "source_name": Path(source).name,
            "age_min": age_min,
            "age_max": age_max,
            "age_group": age_group,
            "document_type": infer_doc_type(source),
            "language": infer_language(text),
            "themes": infer_themes(text, source),
        }
        # Repeating the title and tags in each chunk makes short passages searchable.
        header = (
            f"DOCUMENT: {title}\n"
            f"PUBLIC: {age_group} | TYPE: {metadata['document_type']} | "
            f"THEMES: {metadata['themes']}\n"
        )
        document.page_content = header + text
        document.metadata = metadata
        prepared_pages.append(document)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1400,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "],
    )
    chunks = splitter.split_documents(prepared_pages)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    print(f"3. Création de {len(chunks)} passages dans {DB_DIR}...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(DB_DIR),
        collection_name=COLLECTION_NAME,
    )


if __name__ == "__main__":
    main()
