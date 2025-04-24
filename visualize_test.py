"""
analyze_results.py - Generate visualizations and analysis from test results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
RESULTS_FILE = "test_results_full_db/test_results.json"
OUTPUT_DIR = "test_results_full_db/analysis"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_results():
    """Load test results from JSON file"""
    with open(RESULTS_FILE, 'r') as f:
        return json.load(f)

def analyze_response_times(results):
    """Analyze and visualize response times"""
    categories = list(results['results'].keys())
    basic_times = []
    graph_times = []
    
    for category in categories:
        category_results = results['results'][category]
        basic_category_times = [q['basic_rag']['time'] for q in category_results]
        graph_category_times = [q['graph_rag']['time'] for q in category_results]
        
        basic_times.append(np.mean(basic_category_times))
        graph_times.append(np.mean(graph_category_times))
    
    # Create bar chart
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    basic_bars = ax.bar(x - width/2, basic_times, width, label='Basic RAG')
    graph_bars = ax.bar(x + width/2, graph_times, width, label='GraphRAG')
    
    ax.set_title('Average Response Time by Question Category')
    ax.set_xlabel('Question Category')
    ax.set_ylabel('Response Time (seconds)')
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in categories])
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/response_times.png")
    plt.close()

def analyze_answer_length(results):
    """Analyze and visualize answer lengths"""
    categories = list(results['results'].keys())
    basic_lengths = []
    graph_lengths = []
    
    for category in categories:
        category_results = results['results'][category]
        basic_category_lengths = [len(q['basic_rag']['answer'].split()) for q in category_results]
        graph_category_lengths = [len(q['graph_rag']['answer'].split()) for q in category_results]
        
        basic_lengths.append(np.mean(basic_category_lengths))
        graph_lengths.append(np.mean(graph_category_lengths))
    
    # Create bar chart
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    basic_bars = ax.bar(x - width/2, basic_lengths, width, label='Basic RAG')
    graph_bars = ax.bar(x + width/2, graph_lengths, width, label='GraphRAG')
    
    ax.set_title('Average Answer Length by Question Category (word count)')
    ax.set_xlabel('Question Category')
    ax.set_ylabel('Word Count')
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in categories])
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/answer_lengths.png")
    plt.close()

def generate_report(results):
    """Generate a detailed analysis report"""
    with open(f"{OUTPUT_DIR}/analysis_report1.md", 'w') as f:
        f.write("# GraphRAG vs Basic RAG Analysis Report\n\n")
        
        # Overall statistics
        f.write("## Overall Performance\n\n")
        basic_avg = results['metrics']['basic_rag']['avg_time']
        graph_avg = results['metrics']['graph_rag']['avg_time']
        time_diff = graph_avg - basic_avg
        time_diff_percent = (time_diff / basic_avg) * 100
        
        f.write(f"- Basic RAG average response time: **{basic_avg:.2f}s**\n")
        f.write(f"- GraphRAG average response time: **{graph_avg:.2f}s**\n")
        f.write(f"- Difference: **{time_diff:.2f}s** ({time_diff_percent:.1f}%)\n\n")
        
        # Analysis by question type
        f.write("## Performance by Question Type\n\n")
        f.write("| Category | Basic RAG Time | GraphRAG Time | Difference | % Change |\n")
        f.write("|----------|---------------|---------------|------------|----------|\n")
        
        for category in results['results'].keys():
            category_results = results['results'][category]
            basic_times = [q['basic_rag']['time'] for q in category_results]
            graph_times = [q['graph_rag']['time'] for q in category_results]
            
            basic_avg = np.mean(basic_times)
            graph_avg = np.mean(graph_times)
            diff = graph_avg - basic_avg
            diff_percent = (diff / basic_avg) * 100
            
            f.write(f"| {category.title()} | {basic_avg:.2f}s | {graph_avg:.2f}s | {diff:.2f}s | {diff_percent:.1f}% |\n")
        
        f.write("\n## Analysis\n\n")
        
        # Compare performance by question type
        f.write("### Performance Analysis by Question Type\n\n")
        
        # Add interpretation and observations here based on the actual results
        f.write("The test results reveal important differences between Basic RAG and GraphRAG approaches:\n\n")
        
        if graph_avg > basic_avg:
            f.write("- GraphRAG shows higher response times overall, likely due to the additional processing required for graph traversal\n")
        else:
            f.write("- GraphRAG shows faster response times overall, which is an unexpected but positive finding\n")
        
        f.write("- For different question types, the performance varies:\n")
        
        for category in results['results'].keys():
            category_results = results['results'][category]
            basic_times = [q['basic_rag']['time'] for q in category_results]
            graph_times = [q['graph_rag']['time'] for q in category_results]
            
            basic_avg = np.mean(basic_times)
            graph_avg = np.mean(graph_times)
            diff_percent = ((graph_avg - basic_avg) / basic_avg) * 100
            
            if category == "relational" or category == "cross_reference":
                f.write(f"  - {category.title()} questions: GraphRAG is {abs(diff_percent):.1f}% {'slower' if diff_percent > 0 else 'faster'}, but likely provides more relevant context\n")
            else:
                f.write(f"  - {category.title()} questions: GraphRAG is {abs(diff_percent):.1f}% {'slower' if diff_percent > 0 else 'faster'}\n")
        
        f.write("\n### Quality Analysis Suggestions\n\n")
        f.write("While response time provides one metric, answer quality is equally important. When manually reviewing the results:\n\n")
        f.write("1. **For Factual Questions**: Check if GraphRAG introduces unnecessary information or if it provides more comprehensive context\n")
        f.write("2. **For Relational Questions**: Evaluate if GraphRAG better connects related clauses and provisions\n")
        f.write("3. **For Complex Questions**: Determine if GraphRAG provides more holistic answers drawing from multiple sections\n")
        f.write("4. **For Cross-Reference Questions**: Compare how effectively each approach finds connections across the document\n")
        f.write("5. **For Definition Questions**: Check if GraphRAG better connects term definitions with their usages\n\n")
        
        f.write("### Recommendations\n\n")
        f.write("Based on the performance metrics, consider:\n\n")
        f.write("- Using GraphRAG for questions requiring relational understanding and cross-document references\n")
        f.write("- Using Basic RAG for simple factual questions where response time is critical\n")
        f.write("- Implementing a hybrid approach where question type is classified first, then routed to the appropriate RAG method\n")

def main():
    print("Analyzing test results...")
    results = load_results()
    
    print("Generating response time analysis...")
    analyze_response_times(results)
    
    print("Generating answer length analysis...")
    analyze_answer_length(results)
    
    print("Generating detailed report...")
    generate_report(results)
    
    print(f"Analysis complete! Results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
