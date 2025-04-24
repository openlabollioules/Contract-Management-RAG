from dataclasses import dataclass

from utils.logger import setup_logger

from .embeddings_manager import EmbeddingsManager
from .intelligent_splitter import ChunkMetadata, ChunkType

# Configurer le logger pour ce module
logger = setup_logger(__file__)


@dataclass
class OptimizedChunk:
    """Représente un chunk optimisé avec ses métadonnées"""

    content: str
    metadata: ChunkMetadata
    chunk_type: ChunkType
    similarity_score: float = 0.0
    is_merged: bool = False


class ChunkOptimizer:
    """Optimise les chunks en utilisant la similarité sémantique et le Text Tiling"""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.embeddings_manager = EmbeddingsManager()
        logger.info(
            f"ChunkOptimizer initialisé avec similarity_threshold={similarity_threshold}"
        )
        logger.debug("EmbeddingsManager créé pour ChunkOptimizer")

    """def optimize_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        print("\n🔍 Optimisation des chunks...")
        start_time = time.time()
        
        # Convertir les chunks en OptimizedChunk
        optimized_chunks = [
            OptimizedChunk(
                content=chunk.content,
                metadata=chunk.metadata,
                chunk_type=chunk.chunk_type
            )
            for chunk in chunks
        ]
        
        # Générer les embeddings pour tous les chunks
        print("📊 Génération des embeddings...")
        chunk_embeddings = self.embeddings_manager.get_embeddings(
            [chunk.content for chunk in optimized_chunks]
        )
        
        # Calculer les similarités entre chunks consécutifs
        print("🔍 Calcul des similarités...")
        for i in tqdm(range(len(optimized_chunks) - 1), desc="Analyse des similarités"):
            similarity = self.embeddings_manager.compute_similarity(
                chunk_embeddings[i],
                chunk_embeddings[i + 1]
            )
            
            # Si la similarité est élevée, marquer pour fusion
            if similarity > self.similarity_threshold:
                optimized_chunks[i].is_merged = True
                optimized_chunks[i + 1].is_merged = True
                optimized_chunks[i].similarity_score = similarity
        
        # Fusionner les chunks marqués
        print("🔄 Fusion des chunks similaires...")
        merged_chunks = []
        current_chunk = None
        
        for chunk in optimized_chunks:
            if not chunk.is_merged:
                if current_chunk:
                    merged_chunks.append(current_chunk)
                current_chunk = Chunk(
                    content=chunk.content,
                    metadata=chunk.metadata,
                    chunk_type=chunk.chunk_type
                )
            else:
                if current_chunk:
                    # Fusionner le contenu
                    current_chunk.content += "\n" + chunk.content
                    # Mettre à jour les métadonnées
                    current_chunk.metadata.section_hierarchy.extend(chunk.metadata.section_hierarchy)
                    if chunk.metadata.section_title:
                        current_chunk.metadata.section_title = chunk.metadata.section_title
                else:
                    current_chunk = Chunk(
                        content=chunk.content,
                        metadata=chunk.metadata,
                        chunk_type=chunk.chunk_type
                    )
        
        if current_chunk:
            merged_chunks.append(current_chunk)
            
        print(f"\n✅ Optimisation terminée en {time.time() - start_time:.2f} secondes")
        print(f"📊 Résultats:")
        print(f"  - Chunks initiaux: {len(chunks)}")
        print(f"  - Chunks optimisés: {len(merged_chunks)}")
        print(f"  - Réduction: {((len(chunks) - len(merged_chunks)) / len(chunks) * 100):.1f}%")
        
        return merged_chunks"""
