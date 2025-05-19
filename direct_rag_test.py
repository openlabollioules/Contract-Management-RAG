#!/usr/bin/env python3
"""
direct_rag_test.py - Test script to evaluate question answering directly against a document chunk
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
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", 0.5))

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run RAG tests with direct document content")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-file", 
        help="Text file containing the document content (output from marker)"
    )
    group.add_argument(
        "--pdf-file",
        help="PDF file to extract content from"
    )
    parser.add_argument(
        "--output-dir", 
        default="direct_test_results", 
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
        help=f"LLM model to use (default from config.env: {DEFAULT_MODEL})"
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
        "--context-window",
        type=int,
        default=0,
        help="Context window size in tokens (0 = use model default)"
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
        from src.document_processing.pdf_extractor import extract_pdf_text, init
        
        # Initialize the extractor if needed
        init()
        
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
        from src.document_processing.llm_chat import ask_ollama
        
        # Pass context_window as a positional parameter now that it's defined in the function signature
        return ask_ollama(prompt, temperature, model, context_window)
    except ImportError as e:
        print(f"Error: Could not import ask_ollama: {e}. Falling back to default implementation.")
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

def run_test(question: str, document_content: str, model: str, temperature: float, context_window: int = 0) -> Tuple[str, float]:
    """
    Run a single test with a question directly against document content
    
    Args:
        question: The question to ask
        document_content: The document content as a string
        model: The LLM model to use
        temperature: Temperature for generation
        context_window: Context window size in tokens (0 = use model default)
    
    Returns:
        Tuple of (answer, response_time)
    """
    start_time = time.time()
    
    # Create the prompt template
    prompt = f"""You are an assistant specializing in contract analysis. Please analyze the following document content and answer the user's question.

DOCUMENT CONTENT:
{document_content}

USER'S QUESTION: {question}

Please provide a precise answer based on the document content provided. If the information is not available in the document, state that clearly. Answer only based on the provided document.
You have to generate the shortest answer that you can. Be as concise as possible but still include the essential response elements.
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
    TEMPERATURE = args.temperature
    MAX_QUESTIONS = args.max_questions
    CONTEXT_WINDOW = args.context_window
    
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
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        pdf_text_file = os.path.join(OUTPUT_DIR, "extracted_pdf_text.txt")
        with open(pdf_text_file, 'w', encoding='utf-8') as f:
            f.write(document_content)
        print(f"💾 Saved extracted PDF text to {pdf_text_file}")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize results
    results = {
        "direct_chat": []
    }
    
    metrics = {
        "direct_chat": {"total_time": 0, "question_count": 0}
    }
    
    # Get estimated token count for document (rough estimation: ~1.3 tokens per word)
    word_count = len(document_content.split())
    estimated_tokens = int(word_count * 1.3)
    print(f"📊 Estimated document token count: ~{estimated_tokens} tokens")
    
    # Add configuration information
    config = {
        "description": DESCRIPTION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL,
        "temperature": TEMPERATURE,
        "context_window": CONTEXT_WINDOW,
        "input_file": args.input_file if args.input_file else None,
        "pdf_file": args.pdf_file if args.pdf_file else None,
        "max_questions": MAX_QUESTIONS,
        "document_info": {
            "character_count": len(document_content),
            "word_count": word_count,
            "estimated_token_count": estimated_tokens
        },
        "config_env": {
            "llm_model": os.getenv("LLM_MODEL"),
            "temperature": os.getenv("TEMPERATURE"),
            "ollama_url": os.getenv("OLLAMA_URL"),
            "marker_dir": os.getenv("MARKER_DIR"),
            "use_mps": os.getenv("USE_MPS"),
            "use_half_precision": os.getenv("USE_HALF_PRECISION")
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
            print(f"  Testing direct chat...")
            answer, response_time = run_test(question, document_content, MODEL, TEMPERATURE, CONTEXT_WINDOW)
            
            metrics["direct_chat"]["total_time"] += response_time
            metrics["direct_chat"]["question_count"] += 1
            
            # Save results
            results["direct_chat"].append({
                "question": question,
                "category": category,
                "chat": {
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
            "metrics": metrics
        }, f, indent=2)
    
    # Generate summary
    generate_summary(results, metrics, config, OUTPUT_DIR)

def generate_summary(results, metrics, config, output_dir):
    """Generate a human-readable summary of test results"""
    
    with open(f"{output_dir}/summary.md", "w") as f:
        f.write("# Direct RAG Performance Summary\n\n")
        
        # Test configuration
        f.write("## Test Configuration\n\n")
        f.write(f"- **Date:** {config['timestamp']}\n")
        f.write(f"- **Model:** {config['model']}\n")
        f.write(f"- **Temperature:** {config['temperature']}\n")
        if config.get('context_window', 0) > 0:
            f.write(f"- **Context Window:** {config['context_window']} tokens\n")
        if config['input_file']:
            f.write(f"- **Input File:** {config['input_file']}\n")
        if config['pdf_file']:
            f.write(f"- **PDF File:** {config['pdf_file']}\n")
        f.write(f"- **Document Size:** {config['document_info']['character_count']} characters, {config['document_info']['word_count']} words, ~{config['document_info']['estimated_token_count']} tokens (est.)\n")
        if config['max_questions'] > 0:
            f.write(f"- **Max Questions Per Category:** {config['max_questions']}\n")
        if config['description']:
            f.write(f"- **Description:** {config['description']}\n")
        f.write("\n")
        
        # Add a section for config.env values
        f.write("### Environment Configuration\n\n")
        f.write(f"- **LLM Model (config.env):** {config['config_env']['llm_model']}\n")
        f.write(f"- **Temperature (config.env):** {config['config_env']['temperature']}\n")
        f.write(f"- **Ollama URL:** {config['config_env']['ollama_url']}\n")
        f.write(f"- **Marker Directory:** {config['config_env']['marker_dir']}\n")
        f.write(f"- **Use MPS:** {config['config_env']['use_mps']}\n")
        f.write(f"- **Use Half Precision:** {config['config_env']['use_half_precision']}\n\n")
        
        # Overall metrics
        f.write("## Overall Metrics\n\n")
        f.write("| Metric | Direct Chat |\n")
        f.write("|--------|-------------|\n")
        f.write(f"| Average Response Time | {metrics['direct_chat']['avg_time']:.2f}s |\n")
        f.write(f"| Total Questions | {metrics['direct_chat']['question_count']} |\n\n")
        
        # Group results by category
        categorized_results = {}
        for category in ["questions T", "questions chatGPT"]:
            categorized_results[category] = {}
            if "direct_chat" in results:
                categorized_results[category]["direct_chat"] = []
            
        # Sort results by category
        for result in results["direct_chat"]:
            category = result.get("category", "Uncategorized")
            if category in categorized_results:
                categorized_results[category]["direct_chat"].append(result)
        
        # Per-category summaries
        for category, cat_results in categorized_results.items():
            if not cat_results["direct_chat"]:
                continue  # Skip empty categories
                
            f.write(f"## {category}\n\n")
            
            if "direct_chat" in cat_results and cat_results["direct_chat"]:
                direct_time = sum(q["chat"]["time"] for q in cat_results["direct_chat"])
                direct_avg_time = direct_time / len(cat_results["direct_chat"])
                
                f.write("### Performance Metrics\n\n")
                f.write("| Metric | Direct Chat |\n")
                f.write("|--------|-------------|\n")
                f.write(f"| Average Response Time | {direct_avg_time:.2f}s |\n\n")
            
            # Write individual question results
            for i, result in enumerate(cat_results["direct_chat"], 1):
                f.write(f"### Question {i}: {result['question']}\n\n")
                f.write(f"**Direct Chat** ({result['chat']['time']:.2f}s):\n")
                f.write(f"```\n{result['chat']['answer']}\n```\n\n")
        
        # Provide instructions for manual evaluation
        f.write("## Manual Evaluation Instructions\n\n")
        f.write("To evaluate the quality of responses, please score each answer on the following criteria:\n\n")
        f.write("1. **Accuracy (1-5)**: How factually correct is the answer based on the contract?\n")
        f.write("2. **Completeness (1-5)**: How comprehensive is the answer? Does it address all aspects of the question?\n")
        f.write("3. **Relevance (1-5)**: How relevant is the information provided to the question asked?\n")
        f.write("4. **Coherence (1-5)**: How well-structured and easy to understand is the answer?\n\n")
        
    print(f"\nTest completed! Results saved to {output_dir}/")
    print(f"Summary available at {output_dir}/summary.md")
    print(f"Raw data available at {output_dir}/test_results.json")

if __name__ == "__main__":
    args = parse_arguments()
    print(f"Starting Direct RAG tests...")
    print(f"Configuration: {'Input=' + args.input_file if args.input_file else 'PDF=' + args.pdf_file}, Output={args.output_dir}, Model={args.model}")
    print(f"Using config from config.env (LLM_MODEL={os.getenv('LLM_MODEL')}, TEMPERATURE={os.getenv('TEMPERATURE')})")
    if args.context_window > 0:
        print(f"Context window set to {args.context_window} tokens")
    if args.description:
        print(f"Description: {args.description}")
    if args.max_questions > 0:
        print(f"Limited to {args.max_questions} questions per category")
    run_test_suite(args) 