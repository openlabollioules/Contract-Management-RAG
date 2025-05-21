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
        "In Contract A, which clause is the most problematic from the supplier’s perspective and why? How would you suggest amending that clause to make it less onerous for the supplier?",
        "In Contract A, what is the foreign-exchange risk introduced by the fact that part of the pricing is denominated in rubles?",
        "What is the guaranteed delivered power as specified in Contract A?",
        "Which governing laws are mentioned in Contract A?",
        "I represent the supplier. I would like to send a notice of delay letter to the client under Contract A regarding delays caused by them. Could you propose a template?",
        "Draft a simplified amendment extending the end date of Contract A by six months.",
        "Based on Contract A, can you list the actions the supplier must take in terms of documents to be provided to the client?",
        "Could you estimate the deadlines associated with the list of actions mentioned above?",
        "Which obligations in Contract A must be mandatorily flowed down into the contracts that ALSTOM will sign with its suppliers or subcontractors?",
        "How would you translate the warranty clause of Contract A for ALSTOM’s suppliers and subcontractors?",
        "How would you translate the liability clause of Contract A for ALSTOM’s suppliers and subcontractors?"
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

def run_test(question: str, use_graph: bool) -> Tuple[str, float, List[Dict]]:
    """
    Run test with a single question using either graph or basic RAG on the entire database
    
    Returns:
        Tuple of (answer, response_time, sources)
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
    output_text = output.decode('utf-8')
    answer, sources = parse_answer_and_sources(output_text)
    
    return answer, response_time, sources

def parse_answer_and_sources(output: str) -> Tuple[str, List[Dict]]:
    """Extract the model's answer and sources from the output"""
    lines = output.split('\n')
    answer_start = False
    sources_start = False
    answer_lines = []
    sources = []
    current_source = {}
    
    for line in lines:
        # Detect answer section
        if '🤖 Réponse :' in line:
            answer_start = True
            sources_start = False
            continue
        # Detect sources section
        elif '📚 Sources :' in line:
            answer_start = False
            sources_start = True
            continue
        
        # Collect answer lines
        if answer_start and not sources_start:
            answer_lines.append(line)
        
        # Parse sources
        if sources_start:
            if "=" * 40 in line or "=" * 80 in line:
                continue
                
            # New source starts
            if "-" * 40 in line and current_source:
                if any(current_source.values()):  # Only add if it has data
                    sources.append(current_source)
                current_source = {}
                continue
                
            # Parse source information
            if "Source " in line and "/" in line:
                current_source["id"] = line.strip()
            elif "Source obtenue via le graphe" in line:
                current_source["type"] = "graph"
            elif "Relation:" in line:
                current_source["relation"] = line.replace("Relation:", "").strip()
            elif "Hierarchie:" in line:
                current_source["hierarchy"] = line.replace("Hierarchie:", "").strip()
            elif "Document:" in line:
                current_source["document"] = line.replace("Document:", "").strip()
            elif "Distance:" in line:
                try:
                    current_source["distance"] = float(line.replace("Distance:", "").strip())
                except:
                    current_source["distance"] = line.replace("Distance:", "").strip()
            elif "Résumé utilisé:" in line:
                current_source["is_summary"] = True
            elif "Contenu original:" in line:
                # Next line will be the content
                continue
            elif "Contenu:" in line:
                # Next line will be the content
                continue
            elif len(line.strip()) > 0 and "content" not in current_source:
                # This is likely content text
                current_source["content"] = line.strip()[:200] + "..." if len(line.strip()) > 200 else line.strip()
    
    # Add the last source if it exists
    if current_source and any(current_source.values()):
        sources.append(current_source)
            
    return '\n'.join(answer_lines).strip(), sources

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
        "test_mode": TEST_MODE
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
                chat_answer, chat_time, chat_sources = run_test(question, use_graph=False)
                metrics["chat"]["total_time"] += chat_time
                metrics["chat"]["question_count"] += 1
                
                # Save classic chat results
                results["chat"].append({
                    "question": question,
                    "category": category,
                    "chat": {
                        "answer": chat_answer, 
                        "time": chat_time,
                        "sources": chat_sources
                    }
                })
                
                print(f"  Response time - Classic: {chat_time:.2f}s")
                print(f"  Sources - Classic: {len(chat_sources)}")
            
            # Test with graph-enhanced chat if selected
            if TEST_MODE in ["graph", "both"]:
                print("  Testing with GraphRAG Chat...")
                graph_answer, graph_time, graph_sources = run_test(question, use_graph=True)
                metrics["graph_chat"]["total_time"] += graph_time
                metrics["graph_chat"]["question_count"] += 1
                
                # Save graph chat results
                results["graph_chat"].append({
                    "question": question,
                    "category": category,
                    "chat": {
                        "answer": graph_answer, 
                        "time": graph_time,
                        "sources": graph_sources
                    }
                })
                
                print(f"  Response time - GraphRAG: {graph_time:.2f}s")
                print(f"  Sources - GraphRAG: {len(graph_sources)}")
    
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
                    "sources": result["chat"]["sources"]
                })
            
        # Graph chat sources
        if "graph_chat" in results:
            for result in results["graph_chat"]:
                sources_data.append({
                    "question": result["question"],
                    "category": result.get("category", "Unknown"),
                    "mode": "graph",
                    "sources": result["chat"]["sources"]
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
                
                f.write("### Performance Metrics\n\n")
                f.write("| Metric | Classic Chat | GraphRAG Chat |\n")
                f.write("|--------|-------------|---------------|\n")
                f.write(f"| Average Response Time | {classic_avg_time:.2f}s | {graph_avg_time:.2f}s |\n")
                f.write(f"| Average Sources Used | {classic_avg_sources:.1f} | {graph_avg_sources:.1f} |\n\n")
            
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
                
                f.write("### Performance Metrics\n\n")
                f.write("| Metric | GraphRAG Chat |\n")
                f.write("|--------|---------------|\n")
                f.write(f"| Average Response Time | {graph_avg_time:.2f}s |\n")
                f.write(f"| Average Sources Used | {graph_avg_sources:.1f} |\n\n")
            
            # Write individual question results
            if TEST_MODE == "both":
                for i, result in enumerate(cat_results["chat"], 1):
                    graph_result = cat_results["graph_chat"][i-1]  # corresponding graph result
                    
                    f.write(f"### Question {i}: {result['question']}\n\n")
                    
                    # Classic chat result
                    f.write(f"**Classic Chat** ({result['chat']['time']:.2f}s, {len(result['chat']['sources'])} sources):\n")
                    f.write(f"```\n{result['chat']['answer']}\n```\n\n")
                    
                    # Graph chat result
                    f.write(f"**GraphRAG Chat** ({graph_result['chat']['time']:.2f}s, {len(graph_result['chat']['sources'])} sources):\n")
                    f.write(f"```\n{graph_result['chat']['answer']}\n```\n\n")
            
            elif TEST_MODE == "classic":
                for i, result in enumerate(cat_results["chat"], 1):
                    f.write(f"### Question {i}: {result['question']}\n\n")
                    f.write(f"**Classic Chat** ({result['chat']['time']:.2f}s, {len(result['chat']['sources'])} sources):\n")
                    f.write(f"```\n{result['chat']['answer']}\n```\n\n")
            
            elif TEST_MODE == "graph":
                for i, result in enumerate(cat_results["graph_chat"], 1):
                    f.write(f"### Question {i}: {result['question']}\n\n")
                    f.write(f"**GraphRAG Chat** ({result['chat']['time']:.2f}s, {len(result['chat']['sources'])} sources):\n")
                    f.write(f"```\n{result['chat']['answer']}\n```\n\n")
        
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
    
    print(f"\nTest completed! Results saved to {OUTPUT_DIR}/")
    print(f"Summary available at {OUTPUT_DIR}/summary.md")
    print(f"Raw data available at {OUTPUT_DIR}/test_results.json")
    print(f"Source details available at {OUTPUT_DIR}/sources.json")

if __name__ == "__main__":
    args = parse_arguments()
    print(f"Starting GraphRAG vs Basic RAG performance tests...")
    print(f"Configuration: Output={args.output_dir}")
    if args.description:
        print(f"Description: {args.description}")
    run_test_suite(args)
