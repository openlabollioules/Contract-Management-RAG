import sys
from rag.pdf_loader import extract_text_contract
from rag.intelligent_splitter import IntelligentSplitter, Chunk
from rag.hierarchical_grouper import HierarchicalGrouper
from typing import List
import time
import os

def process_contract(filepath: str) -> List[Chunk]:
    """
    Process a contract file and return intelligent chunks
    
    Args:
        filepath: Path to the contract file
        
    Returns:
        List of Chunk objects
    """
    print("\n🔄 Début du traitement du document...")
    start_time = time.time()
    
    # 1. Load and extract text from PDF
    print("📄 Extraction du texte du PDF...")
    text, document_title = extract_text_contract(filepath)
    print(text)
    print(f"✅ Texte extrait ({len(text.split())} mots)")
    
    # 2. Split text into intelligent chunks
    print("\n🔍 Découpage du texte en chunks intelligents...")
    splitter = IntelligentSplitter(document_title=document_title)
    chunks = splitter.split(text)
    
    # 3. Group chunks hierarchically
    print("\n🔍 Regroupement hiérarchique des chunks...")
    grouper = HierarchicalGrouper()
    hierarchical_groups = grouper.group_chunks(chunks)
    
    # Print document metadata
    print("\nDocument Metadata:")
    print(f"- Title: {document_title}")
    print(f"- Author: Unknown")
    print(f"- Pages: Unknown")
    
    # Print content
    #print("\nContent:")
    #for chunk in chunks:
    #    print(chunk.content)
    
    return chunks

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python main.py <contract_file>")
        sys.exit(1)
        
    filepath = sys.argv[1]
    
    # Process the contract
    chunks = process_contract(filepath)