#!/usr/bin/env python3
"""
contract_summary_test.py - Test script to evaluate question answering using a contract summary approach
"""

import argparse
import json
import time
import os
import sys
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Add src directory to path to correctly import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import questions from test_rag.py
from test_rag import TEST_QUESTIONS

# Load environment variables from config.env
load_dotenv("config.env")

# Default values from config.env
DEFAULT_MODEL = os.getenv("LLM_MODEL", "mistral-small3.1:latest")
DEFAULT_SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "mistral-large3:latest")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", 0.5))

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run tests using contract summary approach")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-file", 
        help="Text file containing the document content"
    )
    group.add_argument(
        "--pdf-file",
        help="PDF file to extract content from"
    )
    parser.add_argument(
        "--output-dir", 
        default="summary_test_results", 
        help="Directory where test results will be stored"
    )
    parser.add_argument(
        "--description", 
        default="", 
        help="Description of the test configuration"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model to use for answering questions (default from config.env: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--summary-model",
        default=DEFAULT_SUMMARY_MODEL,
        help=f"LLM model to use for creating the summary (default: {DEFAULT_SUMMARY_MODEL})"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Temperature for LLM generation (default from config.env: {DEFAULT_TEMPERATURE})"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="Maximum number of questions to test per category (0 = all questions)"
    )
    parser.add_argument(
        "--summary-context-window",
        type=int,
        default=30000,
        help="Context window size in tokens for summary generation (default: 30000)"
    )
    parser.add_argument(
        "--qa-context-window",
        type=int,
        default=8000,
        help="Context window size in tokens for Q&A (default: 8000)"
    )
    parser.add_argument(
        "--use-existing-summary",
        action="store_true",
        help="Use an existing summary file instead of generating a new one"
    )
    parser.add_argument(
        "--summary-file",
        help="Path to an existing summary file (requires --use-existing-summary)"
    )
    return parser.parse_args()

def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file using the project's PDF extractor
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        String containing the extracted text
    """
    try:
        # Try to import and use the main extract_pdf_text function
        from src.document_processing.pdf_extractor import extract_pdf_text, init_pdf_extractor_module
        
        # Initialize the extractor if needed
        init_pdf_extractor_module()
        
        print(f"📄 Extracting text from PDF using project's extractor: {pdf_path}")
        text, title = extract_pdf_text(pdf_path)
        print(f"📄 Extracted document title: {title}")
        return text
    except ImportError as e:
        print(f"Error: Could not import PDF extractor: {e}. Falling back to PyPDF2.")
        try:
            import PyPDF2
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            sys.exit(1)

def ask_llm(prompt: str, temperature: float = DEFAULT_TEMPERATURE, model: str = DEFAULT_MODEL, context_window: int = 0) -> str:
    """
    Ask a question to the LLM using Ollama
    
    Args:
        prompt: The prompt to send to the LLM
        temperature: Temperature for generation
        model: Model name to use
        context_window: Context window size in tokens (0 = use model default)
    
    Returns:
        The model's response
    """
    try:
        from src.document_processing.llm_chat import llm_chat_call_with_ollama
        
        # Pass context_window as a positional parameter
        return llm_chat_call_with_ollama(prompt, temperature, model, context_window)
    except ImportError as e:
        print(f"Error: Could not import llm_chat_call_with_ollama: {e}. Falling back to default implementation.")
        import subprocess
        
        # Use OLLAMA_URL from config.env if available
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        os.environ["OLLAMA_HOST"] = ollama_url
        
        # Fallback implementation using ollama CLI
        cmd = ["ollama", "run", model, "--temperature", str(temperature)]
        
        # Add context window parameter if specified
        if context_window > 0:
            cmd.extend(["--num-ctx", str(context_window)])
            
        cmd.append(prompt)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip()

def generate_contract_summary(document_content: str, model: str, temperature: float, context_window: int = 30000) -> str:
    """
    Generate a comprehensive summary of the contract
    
    Args:
        document_content: The full contract text
        model: The LLM model to use
        temperature: Temperature for generation
        context_window: Context window size in tokens
    
    Returns:
        A comprehensive summary of the contract
    """
    start_time = time.time()
    
    # Create the prompt template for generating a comprehensive summary
    prompt = """Tu es un assistant spécialisé dans l'analyse de contrats juridiques. Ta tâche est de créer un résumé exhaustif du contrat suivant. 

Ce résumé sera utilisé comme unique source d'information pour répondre à des questions spécifiques sur le contrat sans accès au document original.

INSTRUCTIONS:
1. Organise ton résumé par sections thématiques (parties, durée, paiement, obligations, résiliation, etc.)
2. Inclus TOUS les détails importants: dates précises, montants exacts, conditions spécifiques, délais, obligations des parties
3. Extrait et conserve les définitions et termes clés du contrat avec leurs définitions exactes
4. Note les clauses particulières, exceptions et conditions spéciales
5. Priorize les informations factuelles (qui, quoi, quand, combien) plutôt que les formulations juridiques générales
6. Préserve les numéros d'articles et références internes importantes
7. N'omets aucun élément qui pourrait être essentiel pour répondre à des questions sur le contrat
8. Résume de manière exhaustive sans interprétation personnelle

CONTRAT:
"""
    prompt += document_content
    
    # Estimate prompt token count for logging (rough estimation: ~1.3 tokens per word)
    word_count = len(prompt.split())
    estimated_tokens = int(word_count * 1.3)
    print(f"  Estimated prompt token count for summary generation: ~{estimated_tokens} tokens")
    
    print(f"🔍 Generating comprehensive contract summary using {model}...")
    
    # Get response from LLM
    response = ask_llm(prompt, temperature, model, context_window)
    
    generation_time = time.time() - start_time
    print(f"✅ Summary generated in {generation_time:.2f}s")
    
    return response, generation_time

def ask_question_with_summary(question: str, summary: str, model: str, temperature: float, context_window: int = 8000) -> Tuple[str, float]:
    """
    Ask a question using the contract summary
    
    Args:
        question: The question to ask
        summary: The contract summary
        model: The LLM model to use
        temperature: Temperature for generation
        context_window: Context window size in tokens
    
    Returns:
        Tuple of (answer, response_time)
    """
    start_time = time.time()
    
    # Create the prompt template
    prompt = f"""Tu es un assistant spécialisé dans l'analyse de contrats juridiques. Ton rôle est de répondre à des questions spécifiques sur un contrat en utilisant UNIQUEMENT le résumé fourni comme source d'information.

RÉSUMÉ DU CONTRAT:
{summary}

QUESTION DE L'UTILISATEUR: {question}

INSTRUCTIONS:
1. Réponds précisément à la question en te basant UNIQUEMENT sur les informations présentes dans le résumé fourni
2. Si l'information n'est pas disponible dans le résumé, indique clairement que "Cette information n'est pas disponible dans le résumé du contrat"
3. Ne fais aucune supposition ou déduction au-delà des informations explicitement présentes dans le résumé
4. Si pertinent, indique la section ou numéro d'article du contrat d'où provient l'information
5. N'inclus pas de phrases d'introduction ou de conclusion inutiles
"""
    
    # Estimate prompt token count for logging (rough estimation: ~1.3 tokens per word)
    word_count = len(prompt.split())
    estimated_tokens = int(word_count * 1.3)
    print(f"  Estimated prompt token count: ~{estimated_tokens} tokens")
    
    # Get response from LLM
    response = ask_llm(prompt, temperature, model, context_window)
    
    response_time = time.time() - start_time
    
    return response, response_time

def run_test_suite(args):
    """Run all tests and save results"""
    # Setup
    OUTPUT_DIR = args.output_dir
    DESCRIPTION = args.description
    MODEL = args.model
    SUMMARY_MODEL = args.summary_model
    TEMPERATURE = args.temperature
    MAX_QUESTIONS = args.max_questions
    SUMMARY_CONTEXT_WINDOW = args.summary_context_window
    QA_CONTEXT_WINDOW = args.qa_context_window
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get document content either from text file or PDF
    if args.input_file:
        INPUT_FILE = args.input_file
        try:
            with open(INPUT_FILE, 'r', encoding='utf-8') as f:
                document_content = f.read()
                print(f"📄 Loaded document from text file ({len(document_content)} characters)")
        except Exception as e:
            print(f"❌ Error reading input file: {e}")
            sys.exit(1)
    elif args.pdf_file:
        PDF_FILE = args.pdf_file
        document_content = extract_text_from_pdf(PDF_FILE)
        print(f"📄 Extracted content from PDF ({len(document_content)} characters)")
        
        # Optional: save the extracted text for reference
        extracted_text_file = os.path.join(OUTPUT_DIR, "extracted_text.txt")
        with open(extracted_text_file, 'w', encoding='utf-8') as f:
            f.write(document_content)
        print(f"💾 Saved extracted text to {extracted_text_file}")
    
    # Get contract summary
    if args.use_existing_summary and args.summary_file:
        # Use existing summary
        try:
            with open(args.summary_file, 'r', encoding='utf-8') as f:
                summary = f.read()
                summary_generation_time = 0
                print(f"📝 Loaded existing summary from {args.summary_file} ({len(summary)} characters)")
        except Exception as e:
            print(f"❌ Error reading summary file: {e}")
            sys.exit(1)
    else:
        # Generate new summary
        summary, summary_generation_time = generate_contract_summary(
            document_content, 
            SUMMARY_MODEL, 
            TEMPERATURE, 
            SUMMARY_CONTEXT_WINDOW
        )
        
        # Save the summary
        summary_file = os.path.join(OUTPUT_DIR, "contract_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"💾 Saved contract summary to {summary_file} ({len(summary)} characters)")
    
    # Initialize results
    results = {
        "summary_qa": []
    }
    
    metrics = {
        "summary_qa": {"total_time": 0, "question_count": 0}
    }
    
    # Get estimated token count for document and summary (rough estimation: ~1.3 tokens per word)
    doc_word_count = len(document_content.split())
    doc_estimated_tokens = int(doc_word_count * 1.3)
    summary_word_count = len(summary.split())
    summary_estimated_tokens = int(summary_word_count * 1.3)
    
    print(f"📊 Estimated document token count: ~{doc_estimated_tokens} tokens")
    print(f"📊 Estimated summary token count: ~{summary_estimated_tokens} tokens")
    print(f"📊 Compression ratio: {summary_word_count/doc_word_count:.2f}x ({summary_word_count} words / {doc_word_count} words)")
    
    # Add configuration information
    config = {
        "description": DESCRIPTION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL,
        "summary_model": SUMMARY_MODEL,
        "temperature": TEMPERATURE,
        "summary_context_window": SUMMARY_CONTEXT_WINDOW,
        "qa_context_window": QA_CONTEXT_WINDOW,
        "input_file": args.input_file if args.input_file else None,
        "pdf_file": args.pdf_file if args.pdf_file else None,
        "max_questions": MAX_QUESTIONS,
        "document_info": {
            "character_count": len(document_content),
            "word_count": doc_word_count,
            "estimated_token_count": doc_estimated_tokens
        },
        "summary_info": {
            "character_count": len(summary),
            "word_count": summary_word_count,
            "estimated_token_count": summary_estimated_tokens,
            "compression_ratio": f"{summary_word_count/doc_word_count:.2f}x",
            "generation_time": summary_generation_time
        },
        "config_env": {
            "llm_model": os.getenv("LLM_MODEL"),
            "temperature": os.getenv("TEMPERATURE"),
            "ollama_url": os.getenv("OLLAMA_URL")
        }
    }
    
    # Test with both question sets
    for category in ["questions T", "questions chatGPT"]:
        questions = TEST_QUESTIONS[category]
        
        # Limit the number of questions if specified
        if MAX_QUESTIONS > 0 and MAX_QUESTIONS < len(questions):
            print(f"\n== Testing {category.upper()} questions (limited to first {MAX_QUESTIONS}) ==")
            questions = questions[:MAX_QUESTIONS]
        else:
            print(f"\n== Testing {category.upper()} questions ==")
        
        for i, question in enumerate(questions, 1):
            print(f"\nQ{i}: {question}")
            
            # Run the test
            print(f"  Testing summary-based Q&A...")
            answer, response_time = ask_question_with_summary(
                question, 
                summary, 
                MODEL, 
                TEMPERATURE, 
                QA_CONTEXT_WINDOW
            )
            
            metrics["summary_qa"]["total_time"] += response_time
            metrics["summary_qa"]["question_count"] += 1
            
            # Save results
            results["summary_qa"].append({
                "question": question,
                "category": category,
                "qa": {
                    "answer": answer, 
                    "time": response_time
                }
            })
            
            print(f"  Response time: {response_time:.2f}s")
            print(f"  Answer preview: {answer[:100]}...")
            
            # Save results after each question in case of interruption
            with open(f"{OUTPUT_DIR}/test_results_partial.json", "w") as f:
                json.dump({
                    "config": config,
                    "results": results,
                    "metrics": metrics
                }, f, indent=2)
    
    # Calculate averages
    for mode in metrics:
        if metrics[mode]["question_count"] > 0:
            metrics[mode]["avg_time"] = metrics[mode]["total_time"] / metrics[mode]["question_count"]
    
    # Save all results
    with open(f"{OUTPUT_DIR}/test_results.json", "w") as f:
        json.dump({
            "config": config,
            "results": results,
            "metrics": metrics,
            "summary": summary  # Include the summary in the results for reference
        }, f, indent=2)
    
    # Generate summary
    generate_summary(results, metrics, config, OUTPUT_DIR, summary)

def generate_summary(results, metrics, config, output_dir, summary):
    """Generate a human-readable summary of test results"""
    
    with open(f"{output_dir}/summary.md", "w") as f:
        f.write("# Contract Summary Approach Performance Report\n\n")
        
        # Test configuration
        f.write("## Test Configuration\n\n")
        f.write(f"- **Date:** {config['timestamp']}\n")
        f.write(f"- **QA Model:** {config['model']}\n")
        f.write(f"- **Summary Model:** {config['summary_model']}\n")
        f.write(f"- **Temperature:** {config['temperature']}\n")
        f.write(f"- **Summary Context Window:** {config['summary_context_window']} tokens\n")
        f.write(f"- **QA Context Window:** {config['qa_context_window']} tokens\n")
        if config['input_file']:
            f.write(f"- **Input File:** {config['input_file']}\n")
        if config['pdf_file']:
            f.write(f"- **PDF File:** {config['pdf_file']}\n")
        f.write(f"- **Document Size:** {config['document_info']['character_count']} characters, {config['document_info']['word_count']} words, ~{config['document_info']['estimated_token_count']} tokens (est.)\n")
        f.write(f"- **Summary Size:** {config['summary_info']['character_count']} characters, {config['summary_info']['word_count']} words, ~{config['summary_info']['estimated_token_count']} tokens (est.)\n")
        f.write(f"- **Compression Ratio:** {config['summary_info']['compression_ratio']}\n")
        f.write(f"- **Summary Generation Time:** {config['summary_info']['generation_time']:.2f}s\n")
        if config['max_questions'] > 0:
            f.write(f"- **Max Questions Per Category:** {config['max_questions']}\n")
        if config['description']:
            f.write(f"- **Description:** {config['description']}\n")
        f.write("\n")
        
        # Add a section for config.env values
        f.write("### Environment Configuration\n\n")
        f.write(f"- **LLM Model (config.env):** {config['config_env']['llm_model']}\n")
        f.write(f"- **Temperature (config.env):** {config['config_env']['temperature']}\n")
        f.write(f"- **Ollama URL:** {config['config_env']['ollama_url']}\n\n")
        
        # Contract Summary Section
        f.write("## Contract Summary\n\n")
        f.write("```\n")
        # Limit the display of the summary to a reasonable length
        if len(summary) > 1000:
            f.write(f"{summary[:1000]}...\n[Summary truncated - see contract_summary.txt for full text]\n")
        else:
            f.write(f"{summary}\n")
        f.write("```\n\n")
        
        # Overall metrics
        f.write("## Overall Metrics\n\n")
        f.write("| Metric | Summary-based Q&A |\n")
        f.write("|--------|-------------------|\n")
        f.write(f"| Average Response Time | {metrics['summary_qa']['avg_time']:.2f}s |\n")
        f.write(f"| Total Questions | {metrics['summary_qa']['question_count']} |\n\n")
        
        # Group results by category
        categorized_results = {}
        for category in ["questions T", "questions chatGPT"]:
            categorized_results[category] = {}
            if "summary_qa" in results:
                categorized_results[category]["summary_qa"] = []
            
        # Sort results by category
        for result in results["summary_qa"]:
            category = result.get("category", "Uncategorized")
            if category in categorized_results:
                categorized_results[category]["summary_qa"].append(result)
        
        # Per-category summaries
        for category, cat_results in categorized_results.items():
            if not cat_results["summary_qa"]:
                continue  # Skip empty categories
                
            f.write(f"## {category}\n\n")
            
            if "summary_qa" in cat_results and cat_results["summary_qa"]:
                qa_time = sum(q["qa"]["time"] for q in cat_results["summary_qa"])
                qa_avg_time = qa_time / len(cat_results["summary_qa"])
                
                f.write("### Performance Metrics\n\n")
                f.write("| Metric | Summary-based Q&A |\n")
                f.write("|--------|-------------------|\n")
                f.write(f"| Average Response Time | {qa_avg_time:.2f}s |\n\n")
            
            # Write individual question results
            for i, result in enumerate(cat_results["summary_qa"], 1):
                f.write(f"### Question {i}: {result['question']}\n\n")
                f.write(f"**Summary-based Q&A** ({result['qa']['time']:.2f}s):\n")
                f.write(f"```\n{result['qa']['answer']}\n```\n\n")
        
        # Provide instructions for manual evaluation
        f.write("## Manual Evaluation Instructions\n\n")
        f.write("To evaluate the quality of responses, please score each answer on the following criteria:\n\n")
        f.write("1. **Accuracy (1-5)**: How factually correct is the answer based on the contract?\n")
        f.write("2. **Completeness (1-5)**: How comprehensive is the answer? Does it address all aspects of the question?\n")
        f.write("3. **Relevance (1-5)**: How relevant is the information provided to the question asked?\n")
        f.write("4. **Coherence (1-5)**: How well-structured and easy to understand is the answer?\n\n")
        
        f.write("## Comparison with Other Approaches\n\n")
        f.write("Compare these results with the RAG-based approach (if available) to evaluate:\n\n")
        f.write("1. Which approach produces more accurate answers?\n")
        f.write("2. Which approach is faster for response generation?\n")
        f.write("3. Does the summary-based approach miss any important details compared to the RAG approach?\n")
        f.write("4. Is the summary-based approach more consistent in its responses?\n\n")
        
    print(f"\nTest completed! Results saved to {output_dir}/")
    print(f"Summary available at {output_dir}/summary.md")
    print(f"Raw data available at {output_dir}/test_results.json")
    print(f"Contract summary available at {output_dir}/contract_summary.txt")

if __name__ == "__main__":
    args = parse_arguments()
    print(f"Starting Contract Summary-based tests...")
    print(f"Configuration: {'Input=' + args.input_file if args.input_file else 'PDF=' + args.pdf_file}, Output={args.output_dir}")
    print(f"QA Model={args.model}, Summary Model={args.summary_model}, Temperature={args.temperature}")
    print(f"Summary Context Window={args.summary_context_window} tokens, QA Context Window={args.qa_context_window} tokens")
    if args.use_existing_summary and args.summary_file:
        print(f"Using existing summary from {args.summary_file}")
    if args.description:
        print(f"Description: {args.description}")
    if args.max_questions > 0:
        print(f"Limited to {args.max_questions} questions per category")
    run_test_suite(args) 