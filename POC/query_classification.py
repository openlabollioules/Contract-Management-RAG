import re
from ollama import Client
from sklearn.linear_model import LogisticRegression

# 1. Initialisation Ollama
client = Client()

# 2. Motifs pour détection rapide LLM (traduction, rédaction, modèle…)
LLM_PATTERNS = [
    r"tradui", r"rédige", r"modèle", r"courier", r"avenant",
    r"résumer", r"liste des actions", r"problématique"
]  # :contentReference[oaicite:7]{index=7}

# 3. Chargement du classifieur embeddings (pré-entraîné séparément)
# questions_train, labels_train = [...]
# embs_train = [client.embeddings(q) for q in questions_train]
# clf = LogisticRegression().fit(embs_train, labels_train)
# Pour ce POC, on simule un classifieur déjà entraîné
clf = LogisticRegression()  # placeholder

def classify_zero_shot(query: str) -> str:
    """Zero-shot via prompt ; renvoie 'RAG' ou 'LLM'."""
    prompt = (
        "Vous êtes un assistant expert en contrats. Répondez EXACTEMENT par "
        "'RAG' si la réponse nécessite d'extraire des informations précises "
        "depuis le contrat A, sinon 'LLM' pour un appel LLM basique.\n\n"
        f"Question : {query}\nRéponse :"
    )  # :contentReference[oaicite:8]{index=8}
    resp = client.chat(
        model="mistral-small3.1:latest",
        messages=[{"role": "user", "content": prompt}]
    )
    ans = resp.message.content.strip().upper()
    return "RAG" if "RAG" in ans else "LLM"

def classify_with_embeddings(query: str) -> str:
    """Classification via embeddings + logistic regression."""
    emb = client.embeddings(query)  # :contentReference[oaicite:9]{index=9}
    pred = clf.predict([emb])[0]
    return "RAG" if pred == 0 else "LLM"

def call_rag(query: str) -> str:
    """Pipeline RAG : retrieve + generate."""
    # ex. docs = retrieve(query); ctx = format(docs)
    # return client.chat(model="llama3.2", messages=[{"role":"user","content":ctx+query}])
    return f"[RAG] Réponse pour : {query}"

def call_llm(query: str) -> str:
    """Appel LLM classique sans contexte externe."""
    return "LLM"
    resp = client.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": query}]
    )
    return resp.message.content

def route_and_execute(query: str) -> str:
    """Router hybride intégrant règles, zero-shot et embeddings."""
    # 1. Règles métiers
    if any(re.search(p, query, re.I) for p in LLM_PATTERNS):
        return call_llm(query)
    # 2. Classification zero-shot
    mode = classify_zero_shot(query)
    if mode in ("RAG", "LLM"):
        return call_rag(query) if mode == "RAG" else call_llm(query)
    # 3. Fallback embeddings
    return call_rag(query) if classify_with_embeddings(query) == "RAG" else call_llm(query)

# --- Test du POC ---
if __name__ == "__main__":
    TEST_QUESTIONS = [
        "Peux-tu m'indiquer les dates clés du Contrat A ?",
        "Peux-tu me lister les éléments du contrat A qui impliquent le paiement potentiel d'indemnités ou de pénalités de la part du fournisseur ?",
        # "Indique-moi quelles clauses du contrat A diffèrent de manière notable du modèle de contrat B et explique-moi de manière synthétique en quoi consistent ces différences.",
        # "Indique-moi s'il existe des contradictions entre le contrat A et son annexe C.",
        "Peux-tu résumer les obligations de garantie prévues dans le contrat A ?",
        # "Pour le contrat A, merci de répondre aux questions posées dans la Checklist D",
        "Dans le contrat A, quelle est la clause qui est la plus problématique du point de vue du fournisseur et pourquoi ? Comment suggèrerais-tu de corriger cette clause pour la rendre moins problématique du point de vue du fournisseur ?",
        "Dans le contrat A, quel est le risque de change introduit par le fait qu'une partie des prix soient établis en roubles ?",
        # "Je voudrais évaluer les surcoûts liés aux retards client sur ce contrat A. En te servant du mode opératoire F, quels types de préjudices me conseilles-tu de prendre en compte et peux-tu me préciser comment les valoriser ?",
        # "J'aimerais aller plus loin sur le sujet de la Propriété Intellectuelle, sur la base de l'annuaire G, quel est le spécialiste que je peux contacter ?",
        "Quelle est la puissance délivrée attendue telle que spécifiée dans le contrat A ?",
        "Quelles sont les lois applicables mentionnées dans le contrat A ?",
        "Je suis le représentant du fournisseur. J'aimerais envoyer un Courier de notification de retard au client du contrat A concernant des retards subis de sa part. Peux-tu me proposer un modèle ? ",
        "Rédige un avenant simplifié prolongeant la date de fin du contrat A de 6 mois.",
        # "Rédige un contrat d'achat de Alstom vers un fournisseur couvrant toutes les clauses du contrat A qui ne sont pas déjà couvertes par les conditions générales d'achats E",
        "A partir du contrat A, peux-tu dresser la liste des actions à mener par le fournisseur en termes de documents à fournir au client ?",
        "Peux-tu évaluer les dates liées à la liste d'actions précédentes ?",
        "Quelles obligations du contrat A doivent être impérativement intégrées aux contrats qu'ALSTOM signera avec ses fournisseurs ou sous-traitants ?",
        "Comment traduire la clause de garantie du contrat A vis-à-vis des fournisseurs et sous-traitants de Alstom ?",
        "Comment traduire la clause de responsabilité du contrat A vis-à-vis des fournisseurs et sous-traitants de Alstom ?",
    ]
    for q in TEST_QUESTIONS:
        print(f"Q: {q}\n→ {route_and_execute(q)}\n{'-'*40}")
