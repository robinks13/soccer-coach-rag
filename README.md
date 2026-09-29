# Assistant coach de soccer — V1

[Read this documentation in English](README.en.md).

Petit projet RAG local qui répond aux questions d'entraînement de soccer à partir de PDF. Il combine une recherche sémantique et une recherche par mots-clés, puis confie la rédaction à Qwen via Ollama. Les sources consultées apparaissent à la fin de chaque réponse.

J'ai utilisé GPT-5.6-sol pour m'aider à développer et à déboguer cette V1.

Les PDF du corpus peuvent être en français ou en anglais. Tu peux poser une question dans les deux langues ; l'assistant répond dans la langue détectée de la question. Tu peux aussi demander explicitement « réponds en anglais » ou « answer in French ».

## Préparer le projet

Pré-requis : Python (développé avec 3.14), [Ollama](https://ollama.com/) et des PDF de soccer placés dans `data/`.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
ollama pull qwen2.5:32b
python ingest.py
```

Le premier lancement de `ingest.py` télécharge aussi le modèle d'embedding `paraphrase-multilingual-MiniLM-L12-v2` depuis Hugging Face. La base locale est créée dans `chroma_db_v1/`.

## Utilisation

```bash
python chat.py
```

Exemples : `Propose un exercice de finition pour U17`, `Prépare une séance de 90 minutes pour U17 sur la finition` ou `Give me a finishing exercise for U17`.

Pour voir les documents retrouvés avant la génération :

```bash
python search.py "finition U17"
```

## Organisation

- `ingest.py` : extrait les PDF, ajoute des métadonnées et construit l'index.
- `retrieval.py` : recherche et classe les passages.
- `chat.py` : choisit la forme de réponse et interroge le modèle local.
- `search.py` : affiche les passages retrouvés pour aider au diagnostic.

Les PDF, les modèles locaux et les bases Chroma sont exclus de Git : ils sont volumineux et les conditions de réutilisation des documents doivent être vérifiées avant toute redistribution. Sur une nouvelle machine, il faut placer les PDF dans `data/`, puis lancer `python ingest.py`.

## Limites actuelles

Les catégories d'âge et les thèmes sont déduits automatiquement : certains PDF généralistes peuvent être mal classés. Le modèle peut encore proposer une organisation plausible qui ne figure pas dans les documents ; la réponse doit alors la présenter comme une adaptation. Avant d'utiliser une séance sur le terrain, vérifier la page citée et ajuster les charges au groupe.

La détection de langue utilise des mots-clés simples. Une question très courte ou mélangée peut être interprétée comme du français ; précise alors la langue souhaitée dans ta demande.
