import re
import logging
import os
from typing import List

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from sklearn.linear_model import LogisticRegression

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Initialize LLM and Embeddings
llm = ChatOllama(model="llama3.2")
embeddings_model = OllamaEmbeddings(model="llama3.2")

# 2. Define LLM-pattern rules for immediate routing
LLM_PATTERNS = [
    r"tradui", r"rédige", r"modèle", r"courier", r"avenant",
    r"résumer", r"liste des actions", r"problématique",
    r"écri[rstvz]", r"formule[rz]", r"synthétise[rz]", r"explique[rz]",
    r"conseil", r"opinion", r"avis sur", r"comment faire"
]

# 3. Create mock contract data for demonstration
CONTRAT_A_MOCK = """
CONTRAT DE PRESTATION DE SERVICES A

Entre les soussignés :
La société XYZ, immatriculée au RCS sous le numéro 123456789, dont le siège social est situé au 123 rue de la République, 75001 Paris, représentée par Monsieur Jean Dupont, en sa qualité de Directeur Général,
Ci-après dénommée "le Prestataire",
D'une part,

Et
La société ABC, immatriculée au RCS sous le numéro 987654321, dont le siège social est situé au 456 avenue des Champs-Élysées, 75008 Paris, représentée par Madame Marie Martin, en sa qualité de Présidente,
Ci-après dénommée "le Client",
D'autre part,

Article 1 - Objet du contrat
Le présent contrat a pour objet de définir les conditions dans lesquelles le Prestataire s'engage à fournir au Client les prestations définies à l'article 2 ci-après.

Article 2 - Description des prestations
Le Prestataire s'engage à fournir au Client les prestations suivantes : développement d'une application mobile.

Article 3 - Durée du contrat
Le présent contrat est conclu pour une durée de 12 mois à compter de sa date de signature, soit du 1er janvier 2023 au 31 décembre 2023.

Article 4 - Conditions financières
En contrepartie des prestations fournies, le Client s'engage à verser au Prestataire la somme de 50 000 euros HT, payable selon l'échéancier suivant :
- 30% à la signature du contrat
- 40% à la livraison de la version beta
- 30% à la recette finale

Article 5 - Pénalités
En cas de retard de livraison imputable au Prestataire, des pénalités d'un montant de 500 euros par jour de retard seront appliquées, dans la limite de 10% du montant total du contrat.

Article 9 - Loi applicable et juridiction compétente
Le présent contrat est soumis au droit français. Tout litige sera soumis aux tribunaux compétents de Paris.
"""

# 4. Setup Chroma vector store
def setup_chroma_db():
    """Initialize or load ChromaDB for the mock contract"""
    persist_directory = "./chroma_db"
    
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        logger.info("Loading existing Chroma database")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings_model
        )
    
    logger.info("Creating new Chroma database")
    # Split text into chunks, in a real scenario you would use a proper text splitter
    chunks = [CONTRAT_A_MOCK[i:i+500] for i in range(0, len(CONTRAT_A_MOCK), 400)]
    texts = [{"id": str(i), "text": chunk} for i, chunk in enumerate(chunks)]
    
    # Create documents
    documents = [
        {"id": item["id"], "text": item["text"], "metadata": {"source": "Contrat A"}}
        for item in texts
    ]
    
    # Initialize Chroma
    db = Chroma.from_texts(
        texts=[doc["text"] for doc in documents],
        metadatas=[doc["metadata"] for doc in documents],
        embedding=embeddings_model,
        persist_directory=persist_directory
    )
    return db

# Initialize vector store
try:
    vector_store = setup_chroma_db()
except Exception as e:
    logger.error(f"Error setting up ChromaDB: {e}")
    # Fallback to a mock vector store for demonstration
    vector_store = None

# 5. Load or train embeddings classifier
# For POC, assume clf is pre-loaded
clf = LogisticRegression()  # placeholder

# 6. Zero-shot classification
zero_shot_template = PromptTemplate(
    input_variables=["query"],
    template=(
        "Vous êtes un assistant expert en contrats. Répondez EXACTEMENT par 'RAG' "
        "si la réponse nécessite d'extraire des informations précises depuis la base documentaire, "
        "sinon 'LLM' pour un appel LLM basique.\n\nQuestion : {query}\nRéponse :"
    )
)

def classify_query_zero_shot(query: str) -> str:
    """Classify query using zero-shot prompting"""
    message = HumanMessage(content=zero_shot_template.format(query=query))
    resp = llm.invoke([message]).content.strip().upper()
    logger.info(f"Zero-shot classification for '{query}': {resp}")
    return "RAG" if "RAG" in resp else "LLM"

def classify_embeddings(query: str) -> str:
    """Classify query using embeddings (simulated)"""
    # In a real scenario, this would use the pre-trained classifier
    emb = embeddings_model.embed_query(query)
    # Simulate classification - in production use clf.predict([emb])[0]
    is_rag = "contrat" in query.lower() or "date" in query.lower()
    return "RAG" if is_rag else "LLM"

def retrieve_context(query: str, k: int = 3) -> str:
    """Retrieve relevant context from vector store"""
    if vector_store is None:
        return f"[Simulation: Voici les informations pertinentes du contrat pour: {query}]"
    
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

def call_rag(query: str) -> str:
    """Execute RAG pipeline"""
    return "RAG"
    context = retrieve_context(query)
    prompt = f"""Vous êtes un assistant juridique expert.
Voici les informations pertinentes extraites d'un contrat:

{context}

En vous basant uniquement sur ces informations, répondez à la question suivante.
Si la réponse n'est pas dans les informations fournies, indiquez-le clairement.

Question: {query}

Réponse:"""
    
    message = HumanMessage(content=prompt)
    return llm.invoke([message]).content

def call_llm(query: str) -> str:
    """Execute direct LLM call"""
    message = HumanMessage(content=query)
    #return llm.invoke([message]).content
    return "Réponse de LLM"

def determine_inference_mode(query: str) -> str:
    """Main router function implementing the hybrid approach"""
    # 1) Pattern-based rules
    if any(re.search(p, query, re.IGNORECASE) for p in LLM_PATTERNS):
        logger.info(f"Query '{query}' matched LLM pattern")
        return call_llm(query)
    
    # 2) Zero-shot classification
    choice = classify_query_zero_shot(query)
    if choice == "RAG":
        logger.info(f"Query '{query}' routed to RAG via zero-shot")
        return call_rag(query)
    elif choice == "LLM":
        logger.info(f"Query '{query}' routed to LLM via zero-shot")
        return call_llm(query)
    
    # 3) Embeddings classification fallback
    choice_emb = classify_embeddings(query)
    logger.info(f"Query '{query}' routed to {choice_emb} via embeddings fallback")
    return call_rag(query) if choice_emb == "RAG" else call_llm(query)

# 7. Example usage
if __name__ == "__main__":
    test_queries = [
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
    
    for q in test_queries:
        print(f"\nQ: {q}")
        try:
            resp = determine_inference_mode(q)
            print(f"→ {resp}")
        except Exception as e:
            logger.error(f"Error processing query '{q}': {e}")
            print(f"→ Une erreur est survenue: {e}")
        print("-" * 40)