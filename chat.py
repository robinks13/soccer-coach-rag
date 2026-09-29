"""Command-line assistant for questions about football training."""

import re
import sys
import threading
import time
from contextlib import contextmanager

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from retrieval import FootballRetriever, normalize, source_label


MODEL_NAME = "qwen2.5:32b"

PROMPT = PromptTemplate.from_template("""Tu es un assistant pour les entraîneurs de football. Réponds naturellement, avec des explications directement utiles sur le terrain.

Langue de la réponse : {language_instruction}

Fiabilité :
- Utilise les extraits comme documentation. Ne présente pas une proposition personnelle comme une consigne tirée d'un PDF.
- Ne mélange pas les dimensions, le nombre de joueurs ou les règles de plusieurs exercices.
- Ne devine jamais un effectif, une dimension, une durée, un ballon par joueur ou une règle. Si l'information manque, omets-la ou écris « non précisé dans le document ».
- Si tu complètes un exercice ou construis une séance avec des choix absents des extraits, présente-les comme des adaptations proposées.
- Si les extraits ne permettent pas de répondre précisément, dis-le et propose une piste utile sans inventer de source.
- Vérifie que les durées des blocs d'une séance totalisent la durée demandée.

Sources :
- Mets toutes les références uniquement dans une courte section « {sources_heading} » à la fin.
- Chaque référence doit reprendre exactement le nom lisible et la page donnés avec l'extrait.
- Ne mets aucune citation dans le corps de la réponse, ni après une phrase, ni après un exercice.
- N'utilise ni [SOURCE n], ni « extrait n », ni les noms de fichiers avec underscores.
- Mentionne seulement les sources réellement utilisées. Si tu n'en utilises aucune, écris « Aucune source du corpus directement applicable ».

Style :
- Adapte la structure à la demande. Ne suis pas de formulaire fixe.
- Utilise des titres courts, des paragraphes brefs et des listes quand elles aident à comprendre l'action.
- Évite les champs vides et les répétitions.

{mode_instructions}

EXTRAITS DISPONIBLES
{context}

DEMANDE DE L'ENTRAÎNEUR
{question}
""")


# Choose the amount of structure needed for the coach's request.
def response_mode(question: str) -> str:
    value = normalize(question)
    if re.search(r"\b(seance|session|entrainement)\b", value):
        return "session"
    if re.search(r"\b(exercice|exercise|situation|atelier|jeu reduit)\b", value):
        return "exercise"
    return "general"


# Give the model guidance without forcing every answer into one template.
def mode_instructions(mode: str) -> str:
    if mode == "session":
        return """La demande porte sur une séance complète. Propose une progression cohérente avec l'objectif, le public, la durée totale, puis des blocs successifs. Donne l'organisation, le déroulement et les points à observer pour chaque bloc. Termine par un récapitulatif du minutage. Signale brièvement les durées ou règles que tu as choisies toi-même."""
    if mode == "exercise":
        return """La demande porte sur un exercice. Explique l'objectif, la mise en place, le déroulement et les points à observer. Pars d'un exercice documentaire principal. Signale les adaptations que tu proposes pour la catégorie demandée."""
    return """Réponds directement à la question, avec la structure la plus simple qui convient. N'impose pas une fiche d'exercice."""


# Attach readable source labels to passages before they reach the model.
def format_context(documents) -> str:
    if not documents:
        return "Aucun passage pertinent trouvé dans le corpus."
    parts = []
    for index, doc in enumerate(documents, 1):
        label = source_label(doc)
        parts.append(
            f"EXTRAIT {index}\n"
            f"Référence exacte : {label}\n"
            f"Public : {doc.metadata.get('age_group', 'inconnu')}\n"
            f"Type : {doc.metadata.get('document_type', 'inconnu')}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts)


def answer_language(question: str) -> str:
    """Use English for English questions or an explicit request for English."""
    value = normalize(question)
    if "en anglais" in value or "in english" in value:
        return "en"
    if "en francais" in value or "in french" in value:
        return "fr"
    english_words = re.findall(r"\b(give|create|make|how|what|which|training|session|finishing|shooting|drill|exercise)\b", value)
    french_words = re.findall(r"\b(propose|pour|seance|entrainement|exercice|comment|quel|quelle|donne|cree)\b", value)
    return "en" if len(english_words) > len(french_words) else "fr"


# Retrieve evidence and pass the requested response style to Ollama.
def answer(question: str, retriever: FootballRetriever, llm: OllamaLLM) -> str:
    mode = response_mode(question)
    language = answer_language(question)
    documents = retriever.retrieve(question, limit=12 if mode == "session" else 6)
    prompt = PROMPT.format(
        question=question,
        context=format_context(documents),
        mode_instructions=mode_instructions(mode),
        language_instruction=("Answer entirely in English. Keep source titles unchanged." if language == "en" else "Réponds entièrement en français. Conserve les titres des sources."),
        sources_heading=("Sources used" if language == "en" else "Sources utilisées"),
    )
    return llm.invoke(prompt).strip()


# Update the terminal once per second while the local model is working.
@contextmanager
def thinking_timer():
    """Show elapsed time while retrieval and local generation are running."""
    finished = threading.Event()
    started = time.perf_counter()

    def refresh() -> None:
        while not finished.is_set():
            elapsed = time.perf_counter() - started
            print(f"\rRecherche et génération en cours… {elapsed:5.1f} s", end="", flush=True)
            finished.wait(1.0)

    worker = threading.Thread(target=refresh, daemon=True)
    worker.start()
    try:
        yield
    finally:
        finished.set()
        worker.join()
        elapsed = time.perf_counter() - started
        print(f"\rRéponse prête en {elapsed:.1f} s.{' ' * 24}")


# Keep one retriever and one model instance for the interactive session.
def main() -> None:
    print("Chargement de la base football…")
    retriever = FootballRetriever()
    llm = OllamaLLM(model=MODEL_NAME, temperature=0.1)
    print("Assistant coach V1 prêt. Tapez 'exit' pour quitter.")

    while True:
        try:
            question = input("\nCoach > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nÀ bientôt sur les terrains !")
            return
        if question.lower() in {"exit", "quit"}:
            print("À bientôt sur les terrains !")
            return
        if not question:
            continue
        try:
            with thinking_timer():
                response = answer(question, retriever, llm)
        except Exception as error:
            print(f"Impossible de générer la réponse : {error}", file=sys.stderr)
            continue
        print(f"\n{response}")


if __name__ == "__main__":
    main()
