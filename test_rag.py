"""
contract_rag_test.py - Functional tests to compare GraphRAG vs basic RAG performance on entire database
"""

import subprocess
import time
import json
import os
import sys
import argparse
from typing import List, Dict, Tuple

# Different types of contract-related questions for multi-document testing
TEST_QUESTIONS = {
    "questions T": [
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
    ],
    "questions chatGPT": [
        #"Could you indicate the key dates specified in Contract A?",
        #"Can you list the provisions in Contract A that may give rise to potential indemnities or penalties payable by the supplier?",
        #"Could you summarize the warranty obligations set out in Contract A?",
        #"In Contract A, which clause is the most problematic from the supplier's perspective and why? How would you suggest amending that clause to make it less onerous for the supplier?",
        #"In Contract A, what is the foreign-exchange risk introduced by the fact that part of the pricing is denominated in rubles?",
        #"What is the guaranteed delivered power as specified in Contract A?",
        #"Which governing laws are mentioned in Contract A?",
        #"I represent the supplier. I would like to send a notice of delay letter to the client under Contract A regarding delays caused by them. Could you propose a template?",
        #"Draft a simplified amendment extending the end date of Contract A by six months.",
        #"Based on Contract A, can you list the actions the supplier must take in terms of documents to be provided to the client?",
        #"Could you estimate the deadlines associated with the list of actions mentioned above?",
        #"Which obligations in Contract A must be mandatorily flowed down into the contracts that ALSTOM will sign with its suppliers or subcontractors?",
        #"How would you translate the warranty clause of Contract A for ALSTOM's suppliers and subcontractors?",
        #"How would you translate the liability clause of Contract A for ALSTOM's suppliers and subcontractors?"
    ]
}


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run RAG comparison tests")
    parser.add_argument(
        "--output-dir", 
        default="test_results_full_db", 
        help="Directory where test results will be stored"
    )
    parser.add_argument(
        "--description", 
        default="", 
        help="Description of the test configuration"
    )
    parser.add_argument(
        "--mode",
        choices=["classic", "graph", "advanced", "alternatives", "all"],
        default="all",
        help="Which RAG mode to test: classic, graph, advanced, alternatives, or all"
    )
    parser.add_argument(
        "--classic-classification",
        choices=["on", "off", "both"],
        default="off",
        help="Test classic mode with classification (on), without (off), or both"
    )
    parser.add_argument(
        "--hybrid",
        choices=["on", "off", "both"],
        default="off",
        help="Test with hybrid search (BM25 + semantic) (on), without (off), or both"
    )
    return parser.parse_args()

def extract_subqueries(output: str) -> List[str]:
    """Extract sub-queries from the output"""
    subqueries = []
    subquery_marker = "--- Sources pour: '"
    
    # Find all subquery sections in the output
    parts = output.split(subquery_marker)
    
    # First part is before any subquery, skip it
    for part in parts[1:]:
        if "'" in part:
            # Extract the subquery text between the quotes
            subquery = part.split("'", 1)[0].strip()
            # Exclude the "Question principale"
            if subquery and subquery != "Question principale":
                subqueries.append(subquery)
    
    return subqueries

def run_test(question: str, use_graph: bool = False, use_advanced: bool = False, use_alternatives: bool = False, with_classification: bool = False, use_hybrid: bool = False) -> Tuple[str, float, List[Dict], str]:
    """
    Run test with a single question using either graph, advanced, alternatives or basic RAG on the entire database
    
    Args:
        question: The question to ask
        use_graph: Whether to use graph-based RAG
        use_advanced: Whether to use advanced (decomposition) RAG
        use_alternatives: Whether to use alternatives (multiple queries) RAG
        with_classification: Whether to add --classification for classic mode
        use_hybrid: Whether to use hybrid search (BM25 + semantic)
    
    Returns:
        Tuple of (answer, response_time, sources, error_logs)
    """
    start_time = time.time()
    
    # Prepare command - use standalone mode to query entire database
    if use_graph:
        mode = "--graph-chat"
    elif use_advanced:
        mode = "--advanced-chat"
    elif use_alternatives:
        mode = "--alternatives-chat"
    else:
        mode = "--chat"
    
    # Add flags
    extra_flags = []
    
    # Add --classification if requested and in classic mode
    if mode == "--chat" and with_classification:
        extra_flags.append("--classification")

    # Add hybrid search flag (BM25)
    if use_hybrid:
        extra_flags.append("--hybrid")
    elif use_hybrid is False:  # Explicitly disabled
        extra_flags.append("--no-hybrid")
    
    extra_flags_str = " " + " ".join(extra_flags) if extra_flags else ""

    # Print a clear, human-friendly test mode
    mode_name = []
    if mode == "--chat":
        mode_name.append("Classic")
        if with_classification:
            mode_name.append("Classification")
        if use_hybrid:
            mode_name.append("BM25")
    else:
        mode_name.append(mode.replace("--", "").replace("-chat", "").replace("_", " ").title())
        if use_hybrid:
            mode_name.append("BM25")
    
    mode_display = " + ".join(mode_name)
    print(f"Testing with {mode_display} Chat...\n")
    
    # Run the command and capture output
    cmd = f'echo "{question}" | python src/main.py {mode}{extra_flags_str}'
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    
    response_time = time.time() - start_time
    
    # Parse output to extract the answer and sources
    output_text = output.decode('utf-8', errors='replace')
    error_text = error.decode('utf-8', errors='replace')
    
    # Extract the answer from stdout
    answer = extract_answer(output_text)
    
    # Extract subqueries if in advanced mode
    subqueries = []
    if use_advanced:
        subqueries = extract_subqueries(output_text)
        print(f"  Extracted subqueries: {len(subqueries)}")
        for i, subq in enumerate(subqueries, 1):
            print(f"    {i}. {subq}")
    
    # Extract alternative queries if in alternatives mode
    alternative_queries = []
    if use_alternatives:
        # Look for alternative query information in the output
        if "Génération de requêtes alternatives" in error_text:
            alt_lines = [line for line in error_text.split('\n') if "Recherche avec requête alternative" in line]
            for line in alt_lines:
                if ":" in line:
                    alt_query = line.split(":", 2)[-1].strip()
                    alternative_queries.append(alt_query)
        print(f"  Extracted alternative queries: {len(alternative_queries)}")
        for i, alt_q in enumerate(alternative_queries, 1):
            print(f"    {i}. {alt_q[:60]}...")
    
    # Extract sources primarily from logs (stderr)
    sources = extract_sources_from_logs(error_text)
    
    # If no sources found in logs, try to extract from stdout as fallback
    if not sources:
        sources = extract_sources(output_text)
    
    # Process sources based on interaction.py output format
    for source in sources:
        # Set default values
        source["is_graph"] = False
        source["is_advanced"] = use_advanced
        source["is_alternatives"] = use_alternatives
        source["is_hybrid"] = use_hybrid
        source["type"] = "vector"
        source["relation_type"] = None
        source["sub_query"] = source.get("sub_query", None)  # Preserve sub-query info if present
        source["source_query"] = source.get("source_query", None)  # Preserve source query info if present
        
        # Look for specific graph indicators in content
        content = source.get("content", "")
        
        # Look for "Source obtenue via le graphe de connaissances" marker
        if "📊 Source obtenue via le graphe de connaissances" in content:
            source["is_graph"] = True
            source["type"] = "graph"
            
            # Try to extract relation type
            relation_lines = [line for line in content.split('\n') if "Relation:" in line]
            if relation_lines:
                try:
                    relation = relation_lines[0].split("Relation:", 1)[1].strip()
                    source["relation_type"] = relation
                except:
                    pass
        
        # Also check if source_type is explicitly mentioned
        if "source_type" in source and source["source_type"] == "graph":
            source["is_graph"] = True
            source["type"] = "graph"
            
        # For hybrid search, check for BM25 score indicators
        if use_hybrid:
            if "bm25_score" in source:
                source["is_hybrid"] = True
            elif "Score BM25:" in content:
                source["is_hybrid"] = True
                # Try to extract BM25 score
                bm25_lines = [line for line in content.split('\n') if "Score BM25:" in line]
                if bm25_lines:
                    try:
                        bm25_score = bm25_lines[0].split("Score BM25:", 1)[1].strip()
                        source["bm25_score"] = float(bm25_score)
                    except:
                        pass
        
        # For advanced mode, try to find subquery info in source content if not already set
        if use_advanced and not source.get("sub_query"):
            for subq in subqueries:
                if subq in content:
                    source["sub_query"] = subq
                    break
        
        # For alternatives mode, try to find source query info in source content if not already set
        if use_alternatives and not source.get("source_query"):
            for alt_q in alternative_queries:
                if alt_q[:30] in content:
                    source["source_query"] = f"Alternative: {alt_q[:50]}..."
                    break
    
    # Check stderr for stats section to get accurate graph source count
    graph_source_count = sum(1 for s in sources if s.get("is_graph", False))
    
    if "📊 Statistiques des sources:" in error_text and use_graph:
        try:
            stats_section = error_text.split("📊 Statistiques des sources:", 1)[1].strip()
            stats_lines = stats_section.split('\n')
            
            # Extract declared graph sources count
            graph_count_line = next((line for line in stats_lines if "Sources du graphe:" in line), None)
            if graph_count_line:
                try:
                    declared_count = int(graph_count_line.split(":", 1)[1].strip())
                    
                    # If the count doesn't match what we've found, mark additional sources
                    if declared_count > graph_source_count:
                        # Sort remaining sources by distance
                        remaining_sources = [s for s in sources if not s.get("is_graph", False)]
                        sorted_sources = sorted(
                            remaining_sources,
                            key=lambda x: float(x.get("distance", 999)) if isinstance(x.get("distance"), (int, float)) else 999
                        )
                        
                        # Mark additional sources as graph sources
                        for i in range(min(declared_count - graph_source_count, len(sorted_sources))):
                            sorted_sources[i]["is_graph"] = True
                            sorted_sources[i]["type"] = "graph"
                            sorted_sources[i]["relation_type"] = "unknown_relation"
                except:
                    pass
        except:
            pass
    
    # Also check for hybrid source count if using hybrid search
    if "📊 Statistiques des sources:" in error_text and use_hybrid:
        try:
            stats_section = error_text.split("📊 Statistiques des sources:", 1)[1].strip()
            stats_lines = stats_section.split('\n')
            
            # Extract declared hybrid sources count
            hybrid_count_line = next((line for line in stats_lines if "Sources via recherche hybride:" in line), None)
            if hybrid_count_line:
                try:
                    declared_count = int(hybrid_count_line.split(":", 1)[1].strip())
                    
                    # Mark appropriate number of sources as hybrid
                    hybrid_source_count = sum(1 for s in sources if s.get("is_hybrid", False))
                    
                    # If the count doesn't match what we've found, mark additional sources
                    if declared_count > hybrid_source_count:
                        # Sort remaining sources by distance
                        remaining_sources = [s for s in sources if not s.get("is_hybrid", False)]
                        sorted_sources = sorted(
                            remaining_sources,
                            key=lambda x: float(x.get("distance", 999)) if isinstance(x.get("distance"), (int, float)) else 999
                        )
                        
                        # Mark additional sources as hybrid sources
                        for i in range(min(declared_count - hybrid_source_count, len(sorted_sources))):
                            sorted_sources[i]["is_hybrid"] = True
                except:
                    pass
        except:
            pass
    
    # Extract error logs (lines containing ERROR)
    error_logs = "\n".join([line for line in error_text.split('\n') if 'ERROR' in line])
    
    return answer, response_time, sources, error_logs

def extract_answer(output: str) -> str:
    """Extraire la réponse depuis la sortie standard"""
    answer = ""
    answer_markers = ["🤖 Réponse :", "🤖 Réponse synthétisée:"]
    sources_marker = "📚 Sources :"
    
    # Chercher la réponse en utilisant différents marqueurs possibles
    for marker in answer_markers:
        if marker in output:
            answer_section = output.split(marker, 1)[1]
            if sources_marker in answer_section:
                answer = answer_section.split(sources_marker, 1)[0].strip()
            else:
                answer = answer_section.strip()
            break
    
    # Nettoyage supplémentaire pour éviter les artefacts
    if answer:
        # Supprimer les lignes qui contiennent des séparateurs
        cleaned_lines = []
        for line in answer.split('\n'):
            if not line.strip().startswith('=') and not line.strip().startswith('-') and not "-----" in line:
                cleaned_lines.append(line)
        
        answer = '\n'.join(cleaned_lines).strip()
    
    return answer

def extract_sources_from_logs(logs: str) -> List[Dict]:
    """Extract sources from logs (stderr) based on format in interaction.py"""
    sources = []
    current_source = None
    collecting_content = False
    current_content = []
    section_separator = "----------------------------------------"
    source_header_marker = "📚 Sources :"
    stats_marker = "📊 Statistiques des sources:"
    graph_source_marker = "📊 Source obtenue via le graphe de connaissances"
    relation_marker = "Relation:"
    sub_query_marker = "--- Sources pour:"
    current_sub_query = None
    
    # Simplify source extraction by focusing on the core patterns
    lines = logs.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Clean up the log prefix if present
        if " - interaction - INFO - " in line and " - chat_with_contract" in line:
            line = line.split(" - chat_with_contract", 1)[1].strip()
            if line.startswith(" - "):
                line = line[3:]
        
        # Detect sub-query sections
        if sub_query_marker in line:
            try:
                # Format: "--- Sources pour: 'sous-question' (N sources) ---"
                sub_query_part = line.split(sub_query_marker, 1)[1].strip()
                if "'" in sub_query_part:
                    current_sub_query = sub_query_part.split("'", 2)[1].strip()
                    logger.debug(f"Detected sub-query: {current_sub_query}")
            except:
                current_sub_query = None
        
        # Start of source section - skip to actual sources
        if source_header_marker in line or "===============" in line:
            i += 1
            continue
            
        # End of sources section
        if stats_marker in line:
            break
            
        # Source separator - marks start/end of source
        if section_separator in line:
            # If we have a complete source, add it
            if current_source and "document" in current_source:
                if current_content:
                    current_source["content"] = "\n".join(current_content)
                # Add sub-query information if available
                if current_sub_query:
                    current_source["sub_query"] = current_sub_query
                sources.append(current_source)
                current_content = []
                current_source = None
            i += 1
            continue
            
        # New source
        if line.startswith("Source ") and "/" in line:
            # Save previous source if needed
            if current_source and "document" in current_source:
                if current_content:
                    current_source["content"] = "\n".join(current_content)
                # Add sub-query information if available
                if current_sub_query:
                    current_source["sub_query"] = current_sub_query
                sources.append(current_source)
            
            # Start new source
            current_source = {
                "is_graph": False,
                "type": "vector",
                "relation_type": None
            }
            if current_sub_query:
                current_source["sub_query"] = current_sub_query
                
            current_content = []
            collecting_content = False
            
            # Extract position/total
            try:
                pos_total = line.replace("Source ", "").strip()
                pos, total = pos_total.split("/", 1)
                current_source["position"] = pos.strip()
                current_source["total"] = total.strip()
            except:
                current_source["position"] = "unknown"
                current_source["total"] = "unknown"
        
        # Check for graph source marker
        elif current_source is not None and graph_source_marker in line:
            current_source["is_graph"] = True
            current_source["type"] = "graph"
            current_content.append(line)  # Keep this in content too
        
        # Check for relation information
        elif current_source is not None and relation_marker in line:
            try:
                relation = line.split(relation_marker, 1)[1].strip()
                current_source["relation_type"] = relation
            except:
                current_source["relation_type"] = "unknown_relation"
            current_content.append(line)  # Keep this in content too
        
        # Extract hierarchy
        elif current_source is not None and "Hierarchie:" in line:
            hierarchy = line.split("Hierarchie:", 1)[1].strip()
            current_source["hierarchy"] = hierarchy
        
        # Extract document
        elif current_source is not None and "Document:" in line:
            document = line.split("Document:", 1)[1].strip()
            current_source["document"] = document
        
        # Extract distance
        elif current_source is not None and "Distance:" in line:
            try:
                distance = float(line.split("Distance:", 1)[1].strip())
                current_source["distance"] = distance
            except:
                current_source["distance"] = "N/A"
        
        # Extract dates if present
        elif current_source is not None and "Dates:" in line:
            dates = line.split("Dates:", 1)[1].strip()
            if "metadata" not in current_source:
                current_source["metadata"] = {}
            current_source["metadata"]["dates"] = dates
        
        # Start of content section
        elif current_source is not None and "Contenu:" in line:
            collecting_content = True
            # Extract content on same line if present
            if len(line.split("Contenu:", 1)) > 1:
                content_start = line.split("Contenu:", 1)[1].strip()
                if content_start:
                    current_content.append(content_start)
        
        # Start of extract section (common in advanced mode)
        elif current_source is not None and "Extrait:" in line:
            collecting_content = True
            # Extract content on same line if present
            if len(line.split("Extrait:", 1)) > 1:
                content_start = line.split("Extrait:", 1)[1].strip()
                if content_start:
                    current_content.append(content_start)
        
        # Collect content if in content section
        elif collecting_content and current_source is not None:
            # Stop if we hit a new section marker
            if any(marker in line for marker in ["Source ", "Hierarchie:", "Document:", "Distance:", 
                                              section_separator, stats_marker, sub_query_marker]):
                collecting_content = False
            else:
                # Clean up section/hierarchy info that might be mixed in content
                if not (line.startswith("Section:") or line.startswith("Hiérarchie complète:") or 
                        line.startswith("Chapitre:") or line.startswith("Position:")):
                    current_content.append(line)
        
        # Collect other lines that might be part of a source but not yet identified as content
        elif current_source is not None and not collecting_content:
            # Skip empty lines
            if line and not line.isspace():
                # Check for potential content markers
                if "Résumé utilisé:" in line or "Contenu original:" in line or "📊" in line:
                    current_content.append(line)
        
        i += 1
    
    # Add the last source if it exists
    if current_source and "document" in current_source:
        if current_content:
            current_source["content"] = "\n".join(current_content)
        # Add sub-query information if available
        if current_sub_query:
            current_source["sub_query"] = current_sub_query
        sources.append(current_source)
    
    # Ensure all sources have required fields
    for source in sources:
        if "document" not in source:
            source["document"] = "Unknown document"
        if "distance" not in source:
            source["distance"] = "N/A"
        if "content" not in source:
            source["content"] = "No content available"
        if "hierarchy" not in source:
            source["hierarchy"] = "unknown"
        if "metadata" not in source:
            source["metadata"] = {}
        
    return sources

def extract_sources(output: str) -> List[Dict]:
    """Extraire les sources depuis la sortie standard (fallback)"""
    sources = []
    
    # Méthode 1: extraction par blocs de sources
    if "📚 Sources :" in output:
        # Section des sources complète
        sources_section = output.split("📚 Sources :", 1)[1]
        
        # Diviser en blocs de sources individuels
        source_blocks = sources_section.split("----------------------------------------")
        
        for block in source_blocks:
            if not block.strip():
                continue
                
            source = {}
            lines = block.strip().split("\n")
            content_lines = []
            in_content = False
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Extraire les infos de base de la source
                if line.startswith("Source ") and "/" in line:
                    parts = line.split("/")
                    if len(parts) >= 2:
                        source["id"] = line
                        source["position"] = parts[0].replace("Source ", "").strip()
                        source["total"] = parts[1].strip()
                elif "Hierarchie:" in line:
                    source["hierarchy"] = line.split(":", 1)[1].strip() if ":" in line else line
                    in_content = False
                elif "Document:" in line:
                    source["document"] = line.split(":", 1)[1].strip() if ":" in line else line
                    in_content = False
                elif "Distance:" in line:
                    try:
                        dist_value = line.split(":", 1)[1].strip() if ":" in line else line
                        source["distance"] = float(dist_value)
                    except:
                        source["distance"] = dist_value
                    in_content = False
                elif "Section:" in line:
                    source["section"] = line.split(":", 1)[1].strip() if ":" in line else line
                    in_content = False
                elif "Contenu:" in line:
                    in_content = True
                    # Capturer le contenu initial s'il est sur la même ligne
                    content_start = line.split(":", 1)[1].strip() if ":" in line else line
                    if content_start:
                        content_lines.append(content_start)
                else:
                    # Si nous sommes dans une section de contenu, ajouter tout le texte
                    if in_content or not any(key in line for key in ["Source ", "Hierarchie:", "Document:", "Distance:", "Section:"]):
                        content_lines.append(line)
            
            # Ajouter le contenu à la source
            if content_lines:
                source["content"] = "\n".join(content_lines)
            
            # Si la source a au moins un attribut utile, l'ajouter à la liste
            if source and ("document" in source or "content" in source):
                # S'assurer que les champs essentiels sont présents
                if "document" not in source:
                    source["document"] = "Unknown document"
                if "distance" not in source:
                    source["distance"] = "N/A"
                if "content" not in source:
                    source["content"] = "No content available"
                sources.append(source)
    
    # Méthode 2: extraction directe (si aucune source trouvée avec la méthode 1)
    if not sources:
        lines = output.strip().split("\n")
        temp_source = None
        content_lines = []
        
        for line in lines:
            if "Document:" in line and "Distance:" in line:
                # Ligne contenant à la fois Document et Distance
                if temp_source and "document" in temp_source:
                    if content_lines:
                        temp_source["content"] = "\n".join(content_lines)
                    sources.append(temp_source)
                    content_lines = []
                
                temp_source = {}
                
                doc_part = line.split("Document:", 1)[1]
                if "Distance:" in doc_part:
                    doc, dist = doc_part.split("Distance:", 1)
                    temp_source["document"] = doc.strip()
                    try:
                        temp_source["distance"] = float(dist.strip())
                    except:
                        temp_source["distance"] = dist.strip()
            elif "Document:" in line:
                if temp_source and "document" in temp_source:
                    if content_lines:
                        temp_source["content"] = "\n".join(content_lines)
                    sources.append(temp_source)
                    content_lines = []
                
                temp_source = {}
                temp_source["document"] = line.split("Document:", 1)[1].strip()
            elif "Distance:" in line and temp_source:
                try:
                    temp_source["distance"] = float(line.split("Distance:", 1)[1].strip())
                except:
                    temp_source["distance"] = line.split("Distance:", 1)[1].strip()
            elif temp_source:
                # Tout le reste est considéré comme du contenu
                content_lines.append(line.strip())
        
        # Ajouter la dernière source si elle existe
        if temp_source and "document" in temp_source:
            if content_lines:
                temp_source["content"] = "\n".join(content_lines)
            if "content" not in temp_source:
                temp_source["content"] = "No content available"
            sources.append(temp_source)
    
    return sources

def write_results_in_real_time(result, output_dir, mode):
    """Write results to file in real time"""
    # Create file if it doesn't exist
    file_path = f"{output_dir}/real_time_results_{mode}.json"
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump([], f)
    
    # Read existing results
    with open(file_path, 'r') as f:
        existing_results = json.load(f)
    
    # Ensure sources are properly prepared for JSON serialization
    # Some source fields might have non-serializable data
    if "chat" in result and "sources" in result["chat"]:
        for source in result["chat"]["sources"]:
            # Convert any non-serializable values to strings
            for key, value in list(source.items()):
                if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    source[key] = str(value)
            
            # Convert numeric strings to numbers where appropriate
            for key in ["distance", "position", "total"]:
                if key in source and isinstance(source[key], str):
                    try:
                        if "." in source[key]:
                            source[key] = float(source[key])
                        else:
                            source[key] = int(source[key])
                    except (ValueError, TypeError):
                        # Keep as string if conversion fails
                        pass
            
            # Ensure metadata is a dictionary
            if "metadata" not in source:
                source["metadata"] = {}
            elif not isinstance(source["metadata"], dict):
                source["metadata"] = {"original_metadata": str(source["metadata"])}
            
            # Ensure sub_query field is preserved
            if "sub_query" not in source and mode == "advanced":
                source["sub_query"] = None
    
    # Check if the answer is empty and try to fix
    if "chat" in result and "answer" in result["chat"] and not result["chat"]["answer"].strip():
        print(f"  ⚠️ Empty answer detected for {mode} mode. Checking if this can be fixed...")
        # Let's see if there's information in the errors that might help
        if result["chat"].get("errors"):
            error_text = result["chat"]["errors"]
            # Sometimes the answer might be logged in the error output
            answer_markers = ["🤖 Réponse :", "🤖 Réponse synthétisée:"]
            for marker in answer_markers:
                if marker in error_text:
                    answer_section = error_text.split(marker, 1)[1]
                    potential_answer = answer_section.split("\n", 1)[0].strip()
                    if potential_answer:
                        print(f"  📝 Found potential answer in error logs: {potential_answer[:50]}...")
                        result["chat"]["answer"] = potential_answer
                        break
    
    # Append new result
    existing_results.append(result)
    
    # Write back to file
    with open(file_path, 'w') as f:
        json.dump(existing_results, f, indent=2)

def run_test_suite(args):
    """Run all tests and save results"""
    # Setup
    OUTPUT_DIR = args.output_dir
    DESCRIPTION = args.description
    TEST_MODE = args.mode
    CLASSIC_CLASSIFICATION = args.classic_classification
    HYBRID_MODE = args.hybrid
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize results based on test mode
    results = {}
    metrics = {}
    
    # Update mode names for backwards compatibility
    if TEST_MODE == "all":
        modes_to_test = ["classic", "graph", "advanced", "alternatives"]
    else:
        modes_to_test = [TEST_MODE]
    
    # For classic mode, handle classification and hybrid variants
    expanded_modes = []
    for mode in modes_to_test:
        if mode == "classic":
            # Classic has different variants for classification and hybrid
            class_variants = []
            if CLASSIC_CLASSIFICATION == "both":
                class_variants.extend([False, True])
            elif CLASSIC_CLASSIFICATION == "on":
                class_variants.append(True)
            else:  # off
                class_variants.append(False)
                
            hybrid_variants = []
            if HYBRID_MODE == "both":
                hybrid_variants.extend([False, True])
            elif HYBRID_MODE == "on":
                hybrid_variants.append(True)
            else:  # off
                hybrid_variants.append(False)
                
            # Expand classic mode into all combinations
            for class_var in class_variants:
                for hybrid_var in hybrid_variants:
                    mode_name = "classic"
                    if class_var:
                        mode_name += "_classification"
                    if hybrid_var:
                        mode_name += "_bm25"
                    expanded_modes.append(mode_name)
        elif mode == "graph" or mode == "advanced" or mode == "alternatives":
            # Non-classic modes can also have hybrid option
            if HYBRID_MODE == "both":
                expanded_modes.append(mode)
                expanded_modes.append(f"{mode}_bm25")
            elif HYBRID_MODE == "on":
                expanded_modes.append(f"{mode}_bm25")
            else:  # off
                expanded_modes.append(mode)
    
    # Initialize metrics for each mode
    for mode in expanded_modes:
        results[mode] = []
        metrics[mode] = {"total_time": 0, "question_count": 0}
    
    # Add configuration information
    config = {
        "description": DESCRIPTION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_mode": TEST_MODE,
        "output_dir": OUTPUT_DIR,
        "classic_classification": CLASSIC_CLASSIFICATION,
        "hybrid_mode": HYBRID_MODE
    }
    
    # Use both question sets
    for category in ["questions T", "questions chatGPT"]:
        questions = TEST_QUESTIONS[category]
        print(f"\n== Testing {category.upper()} questions ==")
        
        for i, question in enumerate(questions, 1):
            print(f"\nQ{i}: {question}")
            
            # Test with each selected mode
            for mode in expanded_modes:
                print(f"  Testing with {mode.replace('_', ' ').title()} Chat...")
                
                # Set appropriate flags
                use_graph = mode.startswith("graph")
                use_advanced = mode.startswith("advanced")
                use_alternatives = mode.startswith("alternatives")
                with_classification = "_classification" in mode
                use_hybrid = "_bm25" in mode
                
                # Run the test
                answer, time_taken, sources, errors = run_test(
                    question, 
                    use_graph=use_graph, 
                    use_advanced=use_advanced,
                    use_alternatives=use_alternatives,
                    with_classification=with_classification,
                    use_hybrid=use_hybrid
                )
                
                # Update metrics
                metrics[mode]["total_time"] += time_taken
                metrics[mode]["question_count"] += 1
                
                # Display error logs if any
                if errors:
                    print(f"  [!] Errors detected in {mode.replace('_', ' ').capitalize()} Chat:")
                    print(f"  {errors.replace(chr(10), chr(10)+'  ')}")
                
                # Display sources summary
                print(f"  Sources - {mode.replace('_', ' ').capitalize()} ({len(sources)}):")
                
                # Special source info based on mode
                if use_graph:
                    graph_count = sum(1 for s in sources if s.get("type") == "graph" or s.get("is_graph", False))
                    special_info = f"{graph_count} from graph"
                elif use_advanced:
                    sub_queries_count = len(set(s.get("sub_query") for s in sources if s.get("sub_query") is not None))
                    special_info = f"from {sub_queries_count} sub-queries"
                elif use_alternatives:
                    alt_queries_count = len(set(s.get("source_query") for s in sources if s.get("source_query") is not None and s.get("source_query") != "Original"))
                    special_info = f"from {alt_queries_count + 1} query variations"
                else:
                    special_info = ""
                
                # Add hybrid info if using BM25
                if use_hybrid:
                    hybrid_count = sum(1 for s in sources if s.get("is_hybrid", False))
                    hybrid_info = f"{hybrid_count} from hybrid search"
                    special_info = f"{hybrid_info}, {special_info}" if special_info else hybrid_info
                
                # Display source preview
                for idx, source in enumerate(sources[:5], 1):  # Display first 5 sources
                    doc = source.get("document", "Unknown")
                    distance = source.get("distance", "N/A")
                    hierarchy = source.get("hierarchy", "unknown")
                    
                    source_info = f"   {idx}. {doc[:30]}... [Distance: {distance}"
                    
                    # Add mode-specific info
                    if use_graph and (source.get("type") == "graph" or source.get("is_graph", False)):
                        relation = source.get("relation_type", "N/A")
                        source_info += f", Graph: ✓, Relation: {relation}"
                    elif use_advanced and source.get("sub_query"):
                        source_info += f", Sub-query: {source.get('sub_query')[:20]}..."
                    elif use_alternatives and source.get("source_query"):
                        source_info += f", Query: {source.get('source_query')[:20]}..."
                    
                    # Add hybrid-specific info
                    if use_hybrid and source.get("is_hybrid", False):
                        bm25_score = source.get("bm25_score", "N/A")
                        source_info += f", BM25: ✓, Score: {bm25_score}"
                    
                    source_info += f", Hierarchy: {hierarchy[:20]}...]"
                    print(source_info)
                    
                if len(sources) > 5:
                    print(f"   ... and {len(sources) - 5} more sources" + (f" ({special_info})" if special_info else ""))
                
                # Save result
                result = {
                    "question": question,
                    "category": category,
                    "chat": {
                        "answer": answer, 
                        "time": time_taken,
                        "sources": sources,
                        "errors": errors
                    }
                }
                
                results[mode].append(result)
                
                # Write results in real-time
                write_results_in_real_time(result, OUTPUT_DIR, mode)
                
                print(f"  Response time - {mode.replace('_', ' ').capitalize()}: {time_taken:.2f}s")
    
    # Calculate averages
    for mode in metrics:
        if metrics[mode]["question_count"] > 0:
            metrics[mode]["avg_time"] = metrics[mode]["total_time"] / metrics[mode]["question_count"]
    
    # Save all results
    with open(f"{OUTPUT_DIR}/test_results.json", "w") as f:
        json.dump({
            "config": config,
            "results": results,
            "metrics": metrics
        }, f, indent=2)
    
    # Save sources separately to keep file sizes manageable
    with open(f"{OUTPUT_DIR}/sources.json", "w") as f:
        sources_data = []
        
        # Add sources for each mode
        for mode in results:
            for result in results[mode]:
                sources_data.append({
                    "question": result["question"],
                    "category": result.get("category", "Unknown"),
                    "mode": mode,
                    "sources": result["chat"]["sources"],
                    "errors": result["chat"].get("errors", "")
                })
            
        json.dump(sources_data, f, indent=2)
    
    # Generate summary
    generate_summary(results, metrics, config)

def generate_summary(results, metrics, config):
    """Generate a human-readable summary of test results"""
    OUTPUT_DIR = config.get("output_dir", "test_results_full_db")
    TEST_MODE = config.get("test_mode", "all")
    HYBRID_MODE = config.get("hybrid_mode", "off")
    
    with open(f"{OUTPUT_DIR}/summary.md", "w") as f:
        if TEST_MODE == "all":
            f.write("# RAG Performance Comparison Summary\n\n")
        elif TEST_MODE == "classic":
            f.write("# Classic RAG Performance Summary\n\n")
        elif TEST_MODE == "graph":
            f.write("# GraphRAG Performance Summary\n\n")
        elif TEST_MODE == "advanced":
            f.write("# Advanced RAG Performance Summary\n\n")
        elif TEST_MODE == "alternatives":
            f.write("# Alternatives RAG Performance Summary\n\n")
        else:
            f.write("# RAG Performance Summary\n\n")
        
        # Test configuration
        f.write("## Test Configuration\n\n")
        f.write(f"- **Date:** {config['timestamp']}\n")
        f.write(f"- **Test Mode:** {TEST_MODE}\n")
        f.write(f"- **Classification Mode:** {config.get('classic_classification', 'off')}\n")
        f.write(f"- **Hybrid Search (BM25):** {HYBRID_MODE}\n")
        if config.get('description', ''):
            f.write(f"- **Description:** {config['description']}\n")
        f.write("\n")
        
        # Overall metrics
        f.write("## Overall Metrics\n\n")
        
        # Dynamic table based on available modes
        header = "| Metric |"
        separator = "|--------|"
        time_row = "| Average Response Time |"
        count_row = "| Total Questions |"
        
        # Collect available modes and sort them in a sensible order
        available_modes = [mode for mode in results.keys() if metrics[mode]["question_count"] > 0]
        
        # Sort modes in a logical order: classic variants first, then other modes
        def mode_sort_key(mode):
            # Return a tuple for sorting (primary sort key, secondary sort key)
            if mode.startswith("classic"):
                return (0, mode)  # Classic modes first
            elif mode.startswith("graph"):
                return (1, mode)  # Graph modes second
            elif mode.startswith("advanced"):
                return (2, mode)  # Advanced modes third
            elif mode.startswith("alternatives"):
                return (3, mode)  # Alternatives modes fourth
            else:
                return (4, mode)  # Others last
                
        available_modes.sort(key=mode_sort_key)
        
        # Build table header and rows
        for mode in available_modes:
            # Display mode name nicely
            if "_bm25" in mode and "_classification" in mode:
                mode_name = mode.replace("_bm25", " + BM25").replace("_classification", " + Classification")
            elif "_bm25" in mode:
                mode_name = mode.replace("_bm25", " + BM25")
            elif "_classification" in mode:
                mode_name = mode.replace("_classification", " + Classification")
            else:
                mode_name = mode.replace("_", " ").title()
                
            # Add " Chat" suffix if it's not already there
            if not mode_name.endswith(" Chat"):
                mode_name += " Chat"
                
            header += f" {mode_name} |"
            separator += "---------------|"
            
            # Ensure avg_time exists, calculate if needed
            if 'avg_time' not in metrics[mode] and metrics[mode]["question_count"] > 0:
                metrics[mode]["avg_time"] = metrics[mode]["total_time"] / metrics[mode]["question_count"]
            avg_time = metrics[mode].get('avg_time', 0.0)
            time_row += f" {avg_time:.2f}s |"
            count_row += f" {metrics[mode]['question_count']} |"
        
        f.write(header + "\n")
        f.write(separator + "\n")
        f.write(time_row + "\n")
        f.write(count_row + "\n\n")
        
        # Group results by category
        categorized_results = {}
        for category in ["questions T", "questions chatGPT"]:
            categorized_results[category] = {}
            for mode in available_modes:
                categorized_results[category][mode] = []
            
        # Sort results by category and mode
        for mode in results:
            for result in results[mode]:
                category = result.get("category", "Uncategorized")
                if category in categorized_results and mode in categorized_results[category]:
                    categorized_results[category][mode].append(result)
        
        # Per-category summaries
        for category, cat_results in categorized_results.items():
            if not any(cat_results.values()):  # Skip empty categories
                continue
                
            f.write(f"## {category}\n\n")
            
            # Calculate metrics per category
            f.write("### Performance Metrics\n\n")
            
            # Create a dynamic table for this category's metrics
            cat_header = "| Metric |"
            cat_separator = "|--------|"
            cat_time_row = "| Average Response Time |"
            cat_sources_row = "| Average Sources Used |"
            cat_graph_row = "| Average Graph Sources |"
            cat_hybrid_row = "| Average Hybrid Sources |"
            
            for mode in available_modes:
                # Skip modes with no data in this category
                if not cat_results[mode]:
                    continue
                    
                # Display mode name nicely as before
                if "_bm25" in mode and "_classification" in mode:
                    mode_name = mode.replace("_bm25", " + BM25").replace("_classification", " + Classification")
                elif "_bm25" in mode:
                    mode_name = mode.replace("_bm25", " + BM25")
                elif "_classification" in mode:
                    mode_name = mode.replace("_classification", " + Classification")
                else:
                    mode_name = mode.replace("_", " ").title()
                    
                # Add " Chat" suffix if it's not already there
                if not mode_name.endswith(" Chat"):
                    mode_name += " Chat"
                
                cat_header += f" {mode_name} |"
                cat_separator += "---------------|"
                
                # Safely calculate metrics for this mode in this category
                results_len = max(1, len(cat_results[mode]))
                
                # Time metrics
                time_sum = sum(q["chat"]["time"] for q in cat_results[mode])
                avg_time = time_sum / results_len
                cat_time_row += f" {avg_time:.2f}s |"
                
                # Sources metrics
                sources_sum = sum(len(q["chat"]["sources"]) for q in cat_results[mode])
                avg_sources = sources_sum / results_len
                cat_sources_row += f" {avg_sources:.1f} |"
                
                # Graph sources (if applicable)
                if mode.startswith("graph") or "_bm25" in mode:  # Graph mode or any mode with BM25
                    graph_sum = sum(
                        sum(1 for s in q["chat"]["sources"] if s.get("type") == "graph" or s.get("is_graph", False))
                        for q in cat_results[mode]
                    )
                    avg_graph = graph_sum / results_len
                    cat_graph_row += f" {avg_graph:.1f} |"
                else:
                    cat_graph_row += " N/A |"
                
                # Hybrid sources (if applicable)
                if "_bm25" in mode:  # Any mode with BM25
                    hybrid_sum = sum(
                        sum(1 for s in q["chat"]["sources"] if s.get("is_hybrid", False))
                        for q in cat_results[mode]
                    )
                    avg_hybrid = hybrid_sum / results_len
                    cat_hybrid_row += f" {avg_hybrid:.1f} |"
                else:
                    cat_hybrid_row += " N/A |"
            
            # Write the table
            f.write(cat_header + "\n")
            f.write(cat_separator + "\n")
            f.write(cat_time_row + "\n")
            f.write(cat_sources_row + "\n")
            f.write(cat_graph_row + "\n")
            f.write(cat_hybrid_row + "\n\n")
            
            # Write individual question results
            for i, question_index in enumerate(range(len(cat_results[available_modes[0]])), 1):
                # Check if all modes have this question (they should, but check anyway)
                if not all(question_index < len(cat_results[mode]) for mode in available_modes):
                    continue
                
                # Get the question from the first mode (should be the same across all)
                sample_mode = available_modes[0]
                if question_index >= len(cat_results[sample_mode]):
                    continue
                    
                question_text = cat_results[sample_mode][question_index]["question"]
                f.write(f"### Question {i}: {question_text}\n\n")
                
                # Write results for each mode
                for mode in available_modes:
                    if question_index >= len(cat_results[mode]):
                        continue
                        
                    result = cat_results[mode][question_index]
                    
                    # Format mode name nicely as before
                    if "_bm25" in mode and "_classification" in mode:
                        mode_display = mode.replace("_bm25", " + BM25").replace("_classification", " + Classification")
                    elif "_bm25" in mode:
                        mode_display = mode.replace("_bm25", " + BM25")
                    elif "_classification" in mode:
                        mode_display = mode.replace("_classification", " + Classification")
                    else:
                        mode_display = mode.replace("_", " ").title()
                    
                    # Add source counts
                    source_count = len(result['chat']['sources'])
                    graph_count = sum(1 for s in result['chat']['sources'] if s.get("type") == "graph" or s.get("is_graph", False))
                    hybrid_count = sum(1 for s in result['chat']['sources'] if s.get("is_hybrid", False))
                    
                    source_info = f"{source_count} sources"
                    if "_bm25" in mode and hybrid_count > 0:
                        source_info += f", {hybrid_count} hybrid"
                    if mode.startswith("graph") and graph_count > 0:
                        source_info += f", {graph_count} graph"
                    
                    # Write the result
                    f.write(f"**{mode_display} Chat** ({result['chat']['time']:.2f}s, {source_info}):\n")
                    f.write(f"```\n{result['chat']['answer']}\n```\n\n")
                    
                    # Add errors if any
                    if 'errors' in result['chat'] and result['chat']['errors']:
                        f.write(f"**{mode_display} Chat Errors:**\n")
                        f.write(f"```\n{result['chat']['errors']}\n```\n\n")
                    
                    # Add sources preview
                    f.write(f"**{mode_display} Chat Sources (top 5 of {source_count}):**\n\n")
                    for idx, source in enumerate(result['chat']['sources'][:5], 1):
                        doc = source.get("document", "Unknown")
                        distance = source.get("distance", "N/A")
                        
                        source_text = f"{idx}. **Document:** {doc} **Distance:** {distance}"
                        
                        # Add source-specific information
                        if source.get("is_graph", False) or source.get("type") == "graph":
                            relation = source.get("relation_type", "N/A")
                            source_text += f" **Graph:** Yes, **Relation:** {relation}"
                        
                        if source.get("is_hybrid", False):
                            bm25_score = source.get("bm25_score", "N/A")
                            source_text += f" **BM25:** Yes, **Score:** {bm25_score}"
                            
                        if source.get("sub_query"):
                            subquery = source.get("sub_query", "")[:30]
                            source_text += f" **Sub-query:** {subquery}..."
                            
                        if source.get("source_query"):
                            source_query = source.get("source_query", "")[:30]
                            source_text += f" **Source Query:** {source_query}..."
                        
                        f.write(f"{source_text}\n")
                    
                    f.write("\n")
                
        # Provide instructions for manual evaluation
        f.write("## Manual Evaluation Instructions\n\n")
        f.write("To evaluate the quality of responses, please score each answer on the following criteria:\n\n")
        f.write("1. **Accuracy (1-5)**: How factually correct is the answer based on the contract?\n")
        f.write("2. **Completeness (1-5)**: How comprehensive is the answer? Does it address all aspects of the question?\n")
        f.write("3. **Relevance (1-5)**: How relevant is the information provided to the question asked?\n")
        f.write("4. **Coherence (1-5)**: How well-structured and easy to understand is the answer?\n\n")
        
        f.write("\n**Note:** Detailed source information is available in the `sources.json` file.\n")
        f.write("**Real-time results** are available in the `real_time_results_*.json` files.\n")
    
    print(f"\nTest completed! Results saved to {OUTPUT_DIR}/")
    print(f"Summary available at {OUTPUT_DIR}/summary.md")
    print(f"Raw data available at {OUTPUT_DIR}/test_results.json")
    print(f"Source details available at {OUTPUT_DIR}/sources.json")
    print(f"Real-time results available at {OUTPUT_DIR}/real_time_results_*.json")

if __name__ == "__main__":
    args = parse_arguments()
    print(f"Starting RAG performance tests...")
    print(f"Configuration:")
    print(f"- Mode: {args.mode}")
    print(f"- Classification: {args.classic_classification}")
    print(f"- Hybrid Search (BM25): {args.hybrid}")
    print(f"- Output Directory: {args.output_dir}")
    if args.description:
        print(f"- Description: {args.description}")
    run_test_suite(args)
