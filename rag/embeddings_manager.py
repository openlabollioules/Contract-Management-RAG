from typing import List, Dict
import torch
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import platform

class EmbeddingsManager:
    def __init__(self, model_name: str = "BAAI/bge-m3", cache_dir: str = "offline_models/embeddings_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Détection du device
        is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'
        if is_apple_silicon and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🎮 Using MPS (Metal Performance Shaders) for embeddings")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"💻 Using {self.device} for embeddings")
            
        # Charger le modèle
        self.model = SentenceTransformer(model_name, cache_folder="offline_models/hf/hub")
        self.model.to(self.device)
        
        # Optimisations pour MPS
        if self.device.type == "mps":
            self.model.half()  # Utiliser la précision FP16 pour de meilleures performances
            
    def _get_cache_path(self, text: str) -> Path:
        """Génère un chemin de cache unique pour le texte"""
        # Utiliser un hash du texte comme nom de fichier
        text_hash = str(hash(text))
        return self.cache_dir / f"{text_hash}.npy"
        
    def _load_from_cache(self, cache_path: Path) -> np.ndarray:
        """Charge les embeddings depuis le cache"""
        return np.load(str(cache_path))
        
    def _save_to_cache(self, embeddings: np.ndarray, cache_path: Path):
        """Sauvegarde les embeddings dans le cache"""
        np.save(str(cache_path), embeddings)
        
    def get_embeddings(self, texts: List[str], use_cache: bool = True) -> List[np.ndarray]:
        """Génère ou récupère les embeddings pour une liste de textes"""
        results = []
        texts_to_embed = []
        cache_paths = []
        
        # Vérifier le cache pour chaque texte
        for text in texts:
            cache_path = self._get_cache_path(text)
            if use_cache and cache_path.exists():
                results.append(self._load_from_cache(cache_path))
            else:
                texts_to_embed.append(text)
                cache_paths.append(cache_path)
                
        # Générer les nouveaux embeddings si nécessaire
        if texts_to_embed:
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts_to_embed,
                    batch_size=32,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                
            # Sauvegarder dans le cache et ajouter aux résultats
            for emb, cache_path in zip(embeddings, cache_paths):
                self._save_to_cache(emb, cache_path)
                results.append(emb)
                
        return results
        
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calcule la similarité cosinus entre deux embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)) 