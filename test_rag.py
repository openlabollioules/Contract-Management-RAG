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
        "Could you indicate the key dates specified in Contract A?",
        "Can you list the provisions in Contract A that may give rise to potential indemnities or penalties payable by the supplier?",
        "Could you summarize the warranty obligations set out in Contract A?",
        "In Contract A, which clause is the most problematic from the supplier's perspective and why? How would you suggest amending that clause to make it less onerous for the supplier?",
        "In Contract A, what is the foreign-exchange risk introduced by the fact that part of the pricing is denominated in rubles?",
        "What is the guaranteed delivered power as specified in Contract A?",
        "Which governing laws are mentioned in Contract A?",
        "I represent the supplier. I would like to send a notice of delay letter to the client under Contract A regarding delays caused by them. Could you propose a template?",
        "Draft a simplified amendment extending the end date of Contract A by six months.",
        "Based on Contract A, can you list the actions the supplier must take in terms of documents to be provided to the client?",
        "Could you estimate the deadlines associated with the list of actions mentioned above?",
        "Which obligations in Contract A must be mandatorily flowed down into the contracts that ALSTOM will sign with its suppliers or subcontractors?",
        "How would you translate the warranty clause of Contract A for ALSTOM's suppliers and subcontractors?",
        "How would you translate the liability clause of Contract A for ALSTOM's suppliers and subcontractors?"
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
        choices=["classic", "graph", "both"],
        default="both",
        help="Which RAG mode to test: classic, graph, or both"
    )
    return parser.parse_args()

def run_test(question: str, use_graph: bool) -> Tuple[str, float, List[Dict], str]:
    """
    Run test with a single question using either graph or basic RAG on the entire database
    
    Returns:
        Tuple of (answer, response_time, sources, error_logs)
    """
    start_time = time.time()
    
    # Prepare command - use standalone mode to query entire database
    mode = "--graph-chat" if use_graph else "--chat"
    
    # Run the command and capture output
    cmd = f'echo "{question}" | python src/main.py {mode}'
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    
    response_time = time.time() - start_time
    
    # Parse output to extract the answer and sources
    output_text = output.decode('utf-8', errors='replace')
    error_text = error.decode('utf-8', errors='replace')
    
    # Extract the answer from stdout
    answer = extract_answer(output_text)
    
    # Extract sources primarily from logs (stderr)
    sources = extract_sources_from_logs(error_text)
    
    # If no sources found in logs, try to extract from stdout as fallback
    if not sources:
        sources = extract_sources(output_text)
    
    # Process sources based on interaction.py output format
    for source in sources:
        # Set default values
        source["is_graph"] = False
        source["type"] = "vector"
        source["relation_type"] = None
        
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
    
    # Extract error logs (lines containing ERROR)
    error_logs = "\n".join([line for line in error_text.split('\n') if 'ERROR' in line])
    
    return answer, response_time, sources, error_logs

def extract_answer(output: str) -> str:
    """Extraire la réponse depuis la sortie standard"""
    answer = ""
    answer_marker = "🤖 Réponse :"
    sources_marker = "📚 Sources :"
    
    if answer_marker in output:
        answer_section = output.split(answer_marker, 1)[1]
        if sources_marker in answer_section:
            answer = answer_section.split(sources_marker, 1)[0].strip()
        else:
            answer = answer_section.strip()
    
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
    
    # Simplify source extraction by focusing on the core patterns
    lines = logs.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Clean up the log prefix if present
        if " - interaction - INFO - " in line and " - chat_with_contract - " in line:
            line = line.split(" - chat_with_contract - ", 1)[1].strip()
        
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
                sources.append(current_source)
            
            # Start new source
            current_source = {
                "is_graph": False,
                "type": "vector",
                "relation_type": None
            }
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
        
        # Start of content section
        elif current_source is not None and "Contenu:" in line:
            collecting_content = True
            # Extract content on same line if present
            if len(line.split("Contenu:", 1)) > 1:
                content_start = line.split("Contenu:", 1)[1].strip()
                if content_start:
                    current_content.append(content_start)
        
        # Collect content if in content section
        elif collecting_content and current_source is not None:
            # Stop if we hit a new section marker
            if any(marker in line for marker in ["Source ", "Hierarchie:", "Document:", "Distance:", 
                                              section_separator, stats_marker]):
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
            for key, value in source.items():
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
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize results based on test mode
    results = {}
    metrics = {}
    
    if TEST_MODE in ["classic", "both"]:
        results["chat"] = []
        metrics["chat"] = {"total_time": 0, "question_count": 0}
    
    if TEST_MODE in ["graph", "both"]:
        results["graph_chat"] = []
        metrics["graph_chat"] = {"total_time": 0, "question_count": 0}
    
    # Add configuration information
    config = {
        "description": DESCRIPTION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_mode": TEST_MODE,
        "output_dir": OUTPUT_DIR
    }
    
    # Use both question sets
    for category in ["questions T", "questions chatGPT"]:
        questions = TEST_QUESTIONS[category]
        print(f"\n== Testing {category.upper()} questions ==")
        
        for i, question in enumerate(questions, 1):
            print(f"\nQ{i}: {question}")
            
            # Test with classic chat if selected
            if TEST_MODE in ["classic", "both"]:
                print("  Testing with Classic Chat...")
                chat_answer, chat_time, chat_sources, chat_errors = run_test(question, use_graph=False)
                metrics["chat"]["total_time"] += chat_time
                metrics["chat"]["question_count"] += 1
                
                # Display error logs if any
                if chat_errors:
                    print(f"  [!] Errors detected in Classic Chat:")
                    print(f"  {chat_errors.replace(chr(10), chr(10)+'  ')}")
                
                # Display sources summary
                print(f"  Sources - Classic ({len(chat_sources)}):")
                for idx, source in enumerate(chat_sources[:5], 1):  # Display first 5 sources
                    doc = source.get("document", "Unknown")
                    distance = source.get("distance", "N/A")
                    hierarchy = source.get("hierarchy", "unknown")
                    print(f"   {idx}. {doc[:30]}... [Distance: {distance}, Hierarchy: {hierarchy[:20]}...]")
                if len(chat_sources) > 5:
                    print(f"   ... and {len(chat_sources) - 5} more sources")
                
                # Save classic chat results
                chat_result = {
                    "question": question,
                    "category": category,
                    "chat": {
                        "answer": chat_answer, 
                        "time": chat_time,
                        "sources": chat_sources,
                        "errors": chat_errors
                    }
                }
                
                results["chat"].append(chat_result)
                
                # Write results in real-time
                write_results_in_real_time(chat_result, OUTPUT_DIR, "classic")
                
                print(f"  Response time - Classic: {chat_time:.2f}s")
            
            # Test with graph-enhanced chat if selected
            if TEST_MODE in ["graph", "both"]:
                print("  Testing with GraphRAG Chat...")
                graph_answer, graph_time, graph_sources, graph_errors = run_test(question, use_graph=True)
                metrics["graph_chat"]["total_time"] += graph_time
                metrics["graph_chat"]["question_count"] += 1
                
                # Display error logs if any
                if graph_errors:
                    print(f"  [!] Errors detected in GraphRAG Chat:")
                    print(f"  {graph_errors.replace(chr(10), chr(10)+'  ')}")
                
                # Display sources summary
                print(f"  Sources - GraphRAG ({len(graph_sources)}):")
                graph_count = sum(1 for s in graph_sources if s.get("type") == "graph" or s.get("is_graph", False))
                for idx, source in enumerate(graph_sources[:5], 1):  # Display first 5 sources
                    doc = source.get("document", "Unknown")
                    distance = source.get("distance", "N/A")
                    is_graph = "✓" if source.get("type") == "graph" or source.get("is_graph", False) else "✗"
                    hierarchy = source.get("hierarchy", "unknown")
                    relation = source.get("relation_type", "N/A") if is_graph == "✓" else "N/A"
                    
                    source_info = f"   {idx}. {doc[:30]}... [Distance: {distance}, Graph: {is_graph}"
                    if is_graph == "✓":
                        source_info += f", Relation: {relation}"
                    source_info += f", Hierarchy: {hierarchy[:20]}...]"
                    print(source_info)
                    
                if len(graph_sources) > 5:
                    print(f"   ... and {len(graph_sources) - 5} more sources ({graph_count} from graph)")
                
                # Save graph chat results
                graph_result = {
                    "question": question,
                    "category": category,
                    "chat": {
                        "answer": graph_answer, 
                        "time": graph_time,
                        "sources": graph_sources,
                        "errors": graph_errors
                    }
                }
                
                results["graph_chat"].append(graph_result)
                
                # Write results in real-time
                write_results_in_real_time(graph_result, OUTPUT_DIR, "graph")
                
                print(f"  Response time - GraphRAG: {graph_time:.2f}s")
    
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
        
        # Classic chat sources
        if "chat" in results:
            for result in results["chat"]:
                sources_data.append({
                    "question": result["question"],
                    "category": result.get("category", "Unknown"),
                    "mode": "classic",
                    "sources": result["chat"]["sources"],
                    "errors": result["chat"].get("errors", "")
                })
            
        # Graph chat sources
        if "graph_chat" in results:
            for result in results["graph_chat"]:
                sources_data.append({
                    "question": result["question"],
                    "category": result.get("category", "Unknown"),
                    "mode": "graph",
                    "sources": result["chat"]["sources"],
                    "errors": result["chat"].get("errors", "")
                })
            
        json.dump(sources_data, f, indent=2)
    
    # Generate summary
    generate_summary(results, metrics, config)

def generate_summary(results, metrics, config):
    """Generate a human-readable summary of test results"""
    OUTPUT_DIR = config.get("output_dir", "test_results_full_db")
    TEST_MODE = config.get("test_mode", "both")
    
    with open(f"{OUTPUT_DIR}/summary.md", "w") as f:
        if TEST_MODE == "both":
            f.write("# RAG Performance Comparison Summary\n\n")
        elif TEST_MODE == "classic":
            f.write("# Classic RAG Performance Summary\n\n")
        else:
            f.write("# GraphRAG Performance Summary\n\n")
        
        # Test configuration
        f.write("## Test Configuration\n\n")
        f.write(f"- **Date:** {config['timestamp']}\n")
        f.write(f"- **Test Mode:** {TEST_MODE}\n")
        if config['description']:
            f.write(f"- **Description:** {config['description']}\n")
        f.write("\n")
        
        # Overall metrics
        f.write("## Overall Metrics\n\n")
        
        if TEST_MODE == "both":
            f.write("| Metric | Classic Chat | GraphRAG Chat |\n")
            f.write("|--------|-------------|---------------|\n")
            f.write(f"| Average Response Time | {metrics['chat']['avg_time']:.2f}s | {metrics['graph_chat']['avg_time']:.2f}s |\n")
            f.write(f"| Total Questions | {metrics['chat']['question_count']} | {metrics['graph_chat']['question_count']} |\n\n")
        elif TEST_MODE == "classic":
            f.write("| Metric | Classic Chat |\n")
            f.write("|--------|-------------|\n")
            f.write(f"| Average Response Time | {metrics['chat']['avg_time']:.2f}s |\n")
            f.write(f"| Total Questions | {metrics['chat']['question_count']} |\n\n")
        else:
            f.write("| Metric | GraphRAG Chat |\n")
            f.write("|--------|---------------|\n")
            f.write(f"| Average Response Time | {metrics['graph_chat']['avg_time']:.2f}s |\n")
            f.write(f"| Total Questions | {metrics['graph_chat']['question_count']} |\n\n")
        
        # Group results by category
        categorized_results = {}
        for category in ["questions T", "questions chatGPT"]:
            categorized_results[category] = {}
            if "chat" in results:
                categorized_results[category]["chat"] = []
            if "graph_chat" in results:
                categorized_results[category]["graph_chat"] = []
            
        # Sort results by category and mode
        for mode in results:
            for result in results[mode]:
                category = result.get("category", "Uncategorized")
                if category in categorized_results and mode in categorized_results[category]:
                    categorized_results[category][mode].append(result)
        
        # Per-category summaries
        for category, cat_results in categorized_results.items():
            f.write(f"## {category}\n\n")
            
            # Calculate metrics based on test mode
            if TEST_MODE == "both" and "chat" in cat_results and cat_results["chat"]:
                classic_time = sum(q["chat"]["time"] for q in cat_results["chat"])
                classic_sources = sum(len(q["chat"]["sources"]) for q in cat_results["chat"])
                classic_avg_time = classic_time / len(cat_results["chat"])
                classic_avg_sources = classic_sources / len(cat_results["chat"])
                
                graph_time = sum(q["chat"]["time"] for q in cat_results["graph_chat"])
                graph_sources = sum(len(q["chat"]["sources"]) for q in cat_results["graph_chat"])
                graph_avg_time = graph_time / len(cat_results["graph_chat"])
                graph_avg_sources = graph_sources / len(cat_results["graph_chat"])
                
                # Count graph sources
                graph_from_graph = sum(
                    sum(1 for s in q["chat"]["sources"] if s.get("type") == "graph") 
                    for q in cat_results["graph_chat"]
                )
                graph_from_graph_avg = graph_from_graph / len(cat_results["graph_chat"])
                
                f.write("### Performance Metrics\n\n")
                f.write("| Metric | Classic Chat | GraphRAG Chat |\n")
                f.write("|--------|-------------|---------------|\n")
                f.write(f"| Average Response Time | {classic_avg_time:.2f}s | {graph_avg_time:.2f}s |\n")
                f.write(f"| Average Sources Used | {classic_avg_sources:.1f} | {graph_avg_sources:.1f} |\n")
                f.write(f"| Average Graph Sources | N/A | {graph_from_graph_avg:.1f} |\n\n")
            
            elif TEST_MODE == "classic" and "chat" in cat_results and cat_results["chat"]:
                classic_time = sum(q["chat"]["time"] for q in cat_results["chat"])
                classic_sources = sum(len(q["chat"]["sources"]) for q in cat_results["chat"])
                classic_avg_time = classic_time / len(cat_results["chat"])
                classic_avg_sources = classic_sources / len(cat_results["chat"])
                
                f.write("### Performance Metrics\n\n")
                f.write("| Metric | Classic Chat |\n")
                f.write("|--------|-------------|\n")
                f.write(f"| Average Response Time | {classic_avg_time:.2f}s |\n")
                f.write(f"| Average Sources Used | {classic_avg_sources:.1f} |\n\n")
                
            elif TEST_MODE == "graph" and "graph_chat" in cat_results and cat_results["graph_chat"]:
                graph_time = sum(q["chat"]["time"] for q in cat_results["graph_chat"])
                graph_sources = sum(len(q["chat"]["sources"]) for q in cat_results["graph_chat"])
                graph_avg_time = graph_time / len(cat_results["graph_chat"])
                graph_avg_sources = graph_sources / len(cat_results["graph_chat"])
                
                # Count graph sources
                graph_from_graph = sum(
                    sum(1 for s in q["chat"]["sources"] if s.get("type") == "graph") 
                    for q in cat_results["graph_chat"]
                )
                graph_from_graph_avg = graph_from_graph / len(cat_results["graph_chat"])
                
                f.write("### Performance Metrics\n\n")
                f.write("| Metric | GraphRAG Chat |\n")
                f.write("|--------|---------------|\n")
                f.write(f"| Average Response Time | {graph_avg_time:.2f}s |\n")
                f.write(f"| Average Sources Used | {graph_avg_sources:.1f} |\n")
                f.write(f"| Average Graph Sources | {graph_from_graph_avg:.1f} |\n\n")
            
            # Write individual question results
            if TEST_MODE == "both":
                for i, result in enumerate(cat_results["chat"], 1):
                    graph_result = cat_results["graph_chat"][i-1]  # corresponding graph result
                    
                    f.write(f"### Question {i}: {result['question']}\n\n")
                    
                    # Classic chat result
                    f.write(f"**Classic Chat** ({result['chat']['time']:.2f}s, {len(result['chat']['sources'])} sources):\n")
                    f.write(f"```\n{result['chat']['answer']}\n```\n\n")
                    
                    # Classic chat errors (if any)
                    if 'errors' in result['chat'] and result['chat']['errors']:
                        f.write(f"**Classic Chat Errors:**\n")
                        f.write(f"```\n{result['chat']['errors']}\n```\n\n")
                    
                    # Classic chat sources (top 5)
                    f.write(f"**Classic Chat Sources (top 5 of {len(result['chat']['sources'])}):**\n\n")
                    for idx, source in enumerate(result['chat']['sources'][:5], 1):
                        doc = source.get("document", "Unknown")
                        distance = source.get("distance", "N/A")
                        is_summary = "Yes" if source.get("is_summary", False) else "No"
                        f.write(f"{idx}. **Document:** {doc} **Distance:** {distance} **Summary:** {is_summary}\n")
                    f.write("\n")
                    
                    # Graph chat result
                    f.write(f"**GraphRAG Chat** ({graph_result['chat']['time']:.2f}s, {len(graph_result['chat']['sources'])} sources):\n")
                    f.write(f"```\n{graph_result['chat']['answer']}\n```\n\n")
                    
                    # Graph chat errors (if any)
                    if 'errors' in graph_result['chat'] and graph_result['chat']['errors']:
                        f.write(f"**GraphRAG Chat Errors:**\n")
                        f.write(f"```\n{graph_result['chat']['errors']}\n```\n\n")
                    
                    # Graph chat sources (top 5)
                    graph_source_count = sum(1 for s in graph_result['chat']['sources'] if s.get("type") == "graph" or s.get("is_graph", False))
                    f.write(f"**GraphRAG Chat Sources (top 5 of {len(graph_result['chat']['sources'])}, {graph_source_count} from graph):**\n\n")
                    for idx, source in enumerate(graph_result['chat']['sources'][:5], 1):
                        doc = source.get("document", "Unknown")
                        distance = source.get("distance", "N/A")
                        is_summary = "Yes" if source.get("is_summary", False) else "No"
                        is_graph = "Yes" if source.get("type") == "graph" or source.get("is_graph", False) else "No"
                        relation = source.get("relation_type", "N/A") if is_graph == "Yes" else "N/A"
                        hierarchy = source.get("hierarchy", "unknown")
                        
                        f.write(f"{idx}. **Document:** {doc} **Distance:** {distance} **Graph:** {is_graph}")
                        if is_graph == "Yes":
                            f.write(f" **Relation:** {relation}")
                        f.write(f" **Hierarchy:** {hierarchy}\n")
            
            elif TEST_MODE == "classic":
                for i, result in enumerate(cat_results["chat"], 1):
                    f.write(f"### Question {i}: {result['question']}\n\n")
                    f.write(f"**Classic Chat** ({result['chat']['time']:.2f}s, {len(result['chat']['sources'])} sources):\n")
                    f.write(f"```\n{result['chat']['answer']}\n```\n\n")
                    
                    # Classic chat errors (if any)
                    if 'errors' in result['chat'] and result['chat']['errors']:
                        f.write(f"**Classic Chat Errors:**\n")
                        f.write(f"```\n{result['chat']['errors']}\n```\n\n")
                    
                    # Classic chat sources (top 5)
                    f.write(f"**Classic Chat Sources (top 5 of {len(result['chat']['sources'])}):**\n\n")
                    for idx, source in enumerate(result['chat']['sources'][:5], 1):
                        doc = source.get("document", "Unknown")
                        distance = source.get("distance", "N/A")
                        is_summary = "Yes" if source.get("is_summary", False) else "No"
                        f.write(f"{idx}. **Document:** {doc} **Distance:** {distance} **Summary:** {is_summary}\n")
                    f.write("\n")
            
            elif TEST_MODE == "graph":
                for i, result in enumerate(cat_results["graph_chat"], 1):
                    f.write(f"### Question {i}: {result['question']}\n\n")
                    f.write(f"**GraphRAG Chat** ({result['chat']['time']:.2f}s, {len(result['chat']['sources'])} sources):\n")
                    f.write(f"```\n{result['chat']['answer']}\n```\n\n")
                    
                    # Graph chat errors (if any)
                    if 'errors' in result['chat'] and result['chat']['errors']:
                        f.write(f"**GraphRAG Chat Errors:**\n")
                        f.write(f"```\n{result['chat']['errors']}\n```\n\n")
                    
                    # Graph chat sources (top 5)
                    graph_source_count = sum(1 for s in result['chat']['sources'] if s.get("type") == "graph" or s.get("is_graph", False))
                    f.write(f"**GraphRAG Chat Sources (top 5 of {len(result['chat']['sources'])}, {graph_source_count} from graph):**\n\n")
                    for idx, source in enumerate(result['chat']['sources'][:5], 1):
                        doc = source.get("document", "Unknown")
                        distance = source.get("distance", "N/A")
                        is_summary = "Yes" if source.get("is_summary", False) else "No"
                        is_graph = "Yes" if source.get("type") == "graph" or source.get("is_graph", False) else "No"
                        relation = source.get("relation_type", "N/A") if is_graph == "Yes" else "N/A"
                        hierarchy = source.get("hierarchy", "unknown")
                        
                        f.write(f"{idx}. **Document:** {doc} **Distance:** {distance} **Graph:** {is_graph}")
                        if is_graph == "Yes":
                            f.write(f" **Relation:** {relation}")
                        f.write(f" **Hierarchy:** {hierarchy}\n")
                    f.write("\n")
        
        # Provide instructions for manual evaluation
        f.write("## Manual Evaluation Instructions\n\n")
        f.write("To evaluate the quality of responses, please score each answer on the following criteria:\n\n")
        f.write("1. **Accuracy (1-5)**: How factually correct is the answer based on the contract?\n")
        f.write("2. **Completeness (1-5)**: How comprehensive is the answer? Does it address all aspects of the question?\n")
        f.write("3. **Relevance (1-5)**: How relevant is the information provided to the question asked?\n")
        f.write("4. **Coherence (1-5)**: How well-structured and easy to understand is the answer?\n\n")
        
        if TEST_MODE == "both":
            f.write("Compare the scores between Classic Chat and GraphRAG Chat to determine which approach performs better.\n")
        
        f.write("\n**Note:** Detailed source information is available in the `sources.json` file.\n")
        f.write("**Real-time results** are available in the `real_time_results_*.json` files.\n")
    
    print(f"\nTest completed! Results saved to {OUTPUT_DIR}/")
    print(f"Summary available at {OUTPUT_DIR}/summary.md")
    print(f"Raw data available at {OUTPUT_DIR}/test_results.json")
    print(f"Source details available at {OUTPUT_DIR}/sources.json")
    print(f"Real-time results available at {OUTPUT_DIR}/real_time_results_*.json")

if __name__ == "__main__":
    args = parse_arguments()
    print(f"Starting GraphRAG vs Basic RAG performance tests...")
    print(f"Configuration: Output={args.output_dir}")
    if args.description:
        print(f"Description: {args.description}")
    run_test_suite(args)
