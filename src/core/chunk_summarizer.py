from typing import Dict, List
from document_processing.llm_chat import LLMChat
from utils.logger import setup_logger
import re

# Configurer le logger pour ce module
logger = setup_logger(__file__)

class ChunkSummarizer:
    """Classe pour résumer les chunks de texte en utilisant Ollama"""

    def __init__(self):
        """Initialise le summarizer avec une instance de LLMChat"""
        self.llm = LLMChat(model="mistral-small3.1:latest",
            system_prompt="""Tu es un assistant spécialisé dans le résumé de textes juridiques.
            Ta tâche est de produire des résumés concis mais informatifs des extraits de texte qui te sont fournis.
            Concentre-toi sur les points clés, les obligations, les conditions et les informations importantes.
            
            Pour les tableaux :
            - Conserve la structure tabulaire si elle est importante pour la compréhension
            - Résume le contenu en gardant les relations entre les colonnes
            - Mentionne explicitement qu'il s'agit d'un tableau dans ton résumé
            - Indique les titres des colonnes si présents
            
            Pour le texte normal :
            - Le résumé doit être court (max 2-3 phrases)
            - Capture l'essence du texte de manière concise"""
        )

    def _is_likely_table(self, text: str) -> bool:
        """
        Détecte si le texte contient probablement un tableau

        Args:
            text: Le texte à analyser

        Returns:
            True si le texte semble contenir un tableau
        """
        # Patterns qui suggèrent la présence d'un tableau
        patterns = [
            r"[\|\+][-\+]+[\|\+]",  # Lignes de séparation ASCII
            r"\|.*\|.*\|",          # Plusieurs colonnes séparées par |
            r"\t.*\t",              # Plusieurs tabulations
            r"(?:\s{2,}[^\s]+){3,}",  # Colonnes alignées avec espaces
            r"^[^:]+:\s+[^:]+(?:\s{2,}[^:]+:\s+[^:]+)+$"  # Format clé-valeur tabulaire
        ]

        # Vérifier chaque pattern
        for pattern in patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True

        # Vérifier l'alignement vertical des caractères
        lines = text.split('\n')
        if len(lines) > 2:
            # Compter les positions des espaces dans chaque ligne
            space_positions = [
                [i for i, char in enumerate(line) if char.isspace()]
                for line in lines if line.strip()
            ]
            
            # Si plusieurs lignes ont des espaces aux mêmes positions, c'est probablement un tableau
            if len(space_positions) > 2:
                common_spaces = set.intersection(*[set(pos) for pos in space_positions])
                if len(common_spaces) >= 2:
                    return True

        return False

    def summarize_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Résume le contenu de chaque chunk en utilisant Ollama

        Args:
            chunks: Liste de dictionnaires contenant les chunks avec leur contenu et métadonnées

        Returns:
            Liste de chunks avec le contenu résumé
        """
        logger.info("\n📝 Résumé des chunks avec Ollama...")
        summarized_chunks = []

        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Résumé du chunk {i}/{len(chunks)}...")
            
            # Détecter si le chunk contient un tableau
            is_table = self._is_likely_table(chunk['content'])
            
            # Préparer le prompt approprié
            print("Ce n'est pas une table: ")
            print(chunk['content'])
            prompt = f"""Voici un extrait de texte juridique à résumer :

{chunk['content']}

Résumé concis :

Instructions : 
 - Si l'extrait de texte contient un tableau, veille à fournir un résumé des informations contenues dans ce tableau toujours en restant dans la limite de 800 tokens.
 - Tu ne dois jamais dépasser la barre symbolique des 800 tokens dans le résumé que tu fournis.
 - Le but étant de fournir un résumé mais tout en gardant réinsérant dans le résumé les informations importantes (les numéro d'articles par exemple).
"""

            # Générer le résumé
            summary = self.llm.generate(prompt)

            print("voila le résultat: ")
            print(summary)

            # Créer un nouveau chunk avec le résumé
            summarized_chunk = {
                "content": summary,
                "metadata": {
                    **chunk["metadata"],
                    "original_content": chunk["content"],  # Garder le contenu original dans les métadonnées
                    "is_summary": "true",
                    "contains_table": str(is_table).lower()
                }
            }
            summarized_chunks.append(summarized_chunk)

        logger.info(f"✅ {len(summarized_chunks)} chunks résumés")
        return summarized_chunks 