import json
import logging
import os
import uuid
from logging.handlers import RotatingFileHandler

import duckdb
import numpy as np
import requests
from dotenv import load_dotenv
import logging
from utils.logger import setup_logger
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path

logger = setup_logger(__file__)

load_dotenv()

# Initialize embedding model (lazy loading - will only load when first used)
_embedding_model = None

def _get_embedding_model(model_name="nomic-ai/nomic-embed-text-v2-moe"):
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {model_name}")
        models_dir = Path(os.getenv("MODELS_DIR", "offline_models/hf"))
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we should use GPU/MPS
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        
        logger.info(f"Using device: {device} for embeddings")
        
        # Load the model
        _embedding_model = SentenceTransformer(
            model_name,
            cache_folder=str(models_dir),
            device=device,
            trust_remote_code=True
        )
        
        # Use half precision if available for better performance
        if device in ["cuda", "mps"]:
            _embedding_model.half()
    
    return _embedding_model

# Function to obtain embedding using SentenceTransformer
def get_embedding(text, model="nomic-ai/nomic-embed-text-v2-moe"):
    logger.info(f"Fetching embedding for text: {text}")
    try:
        # Get the embedding model
        embedding_model = _get_embedding_model(model)
        
        # Generate embedding
        with torch.no_grad():
            embedding = embedding_model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        return embedding.tolist()
    except Exception as e:
        logger.exception(f"Embedding error: {e}")
        raise


# Function to calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    logger.debug("Calculating cosine similarity between two vectors")
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Function to configure the DuckDB database and `chat_history` table
def setup_history_database(path=None):
    logger.info(f"Initializing database at: {path}")
    
    # Ensure directory exists
    db_dir = os.path.dirname(path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        
    conn = duckdb.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT, 
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (conversation_id, role)  -- Composite primary key
        )
    """
    )
    conn.close()
    logger.info("✅ Database and `chat_history` table successfully initialized.")


# Function to add a conversation entry with embedding storage
def add_conversation_with_embedding(path, question, response, model="nomic-ai/nomic-embed-text-v2-moe"):
    logger.info("Adding a new conversation to the database")
    try:
        conn = duckdb.connect(path)
        conversation_id = str(uuid.uuid4())  # Generate unique conversation ID

        # Generate embeddings for question and response
        question_embedding = get_embedding(question, model)
        response_embedding = get_embedding(response, model)

        # Convert embeddings to JSON format
        question_embedding_json = json.dumps(question_embedding)
        response_embedding_json = json.dumps(response_embedding)

        # Insert user question into chat history
        conn.execute(
            """
            INSERT INTO chat_history (conversation_id, role, content, embedding) 
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, "user", question, question_embedding_json),
        )

        # Insert assistant response into chat history
        conn.execute(
            """
            INSERT INTO chat_history (conversation_id, role, content, embedding) 
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, "assistant", response, response_embedding_json),
        )
        conn.close()
        logger.info("✅ Conversation successfully added.")
    except Exception as e:
        logger.exception("Error adding conversation")
        raise


# Function to retrieve conversation history
def get_history(path):
    logger.info("Fetching conversation history")
    conn = duckdb.connect(path)
    history = conn.execute(
        """
        SELECT role, content, embedding FROM chat_history ORDER BY timestamp ASC
        """
    ).fetchall()
    conn.close()
    logger.info(f"Retrieved {len(history)} entries from history")
    return [
        {
            "role": role,
            "content": content,
            "embedding": json.loads(embedding) if embedding else None,
        }
        for role, content, embedding in history
    ]


# Function to retrieve similar conversations based on embedding similarity
def retrieve_similar_conversations(
    question, path, model="nomic-ai/nomic-embed-text-v2-moe", min_k=1, max_k=5, threshold=0.70
):
    logger.info(f"Searching for similar conversations for question: {question}")
    try:
        question_embedding = get_embedding(question, model)

        # Fetch stored user messages
        conn = duckdb.connect(path)
        user_messages = conn.execute(
            "SELECT conversation_id, content, embedding FROM chat_history WHERE role = 'user'"
        ).fetchall()
        conn.close()

        similarities = []
        conversations = {}

        # Compute similarity between input question and stored user questions
        for conversation_id, content, embedding in user_messages:
            if embedding:
                msg_embedding = json.loads(embedding)
                sim = cosine_similarity(question_embedding, msg_embedding)
                if sim >= threshold:
                    similarities.append((conversation_id, sim))
                    conversations[conversation_id] = {
                        "question": content,
                        "response": None,
                    }

        # Fetch stored assistant responses
        conn = duckdb.connect(path)
        assistant_messages = conn.execute(
            "SELECT conversation_id, content FROM chat_history WHERE role = 'assistant'"
        ).fetchall()
        conn.close()

        # Match responses to corresponding user questions
        for conv_id, content in assistant_messages:
            if conv_id in conversations:
                conversations[conv_id]["response"] = content

        similarities.sort(key=lambda x: x[1], reverse=True)
        logger.debug(f"Found {len(similarities)} similar conversations")
        return [
            conversations[conv_id]
            for conv_id, _ in similarities[: max(min_k, min(len(similarities), max_k))]
        ]
    except Exception as e:
        logger.exception("Error retrieving similar conversations")
        raise