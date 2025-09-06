#!/usr/bin/env python3

import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3
import pickle
from typing import List, Tuple, Optional
from database import DetectionDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticSearchAgent:
    """
    AI agent for semantic search of detection captions using SentenceTransformer.
    Allows natural language queries like "Find all red cars" to search through 
    detection captions and return relevant matches.
    """
    
    def __init__(self, db_path: str = "detections.db", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic search agent.
        
        Args:
            db_path: Path to SQLite database
            model_name: SentenceTransformer model to use
        """
        self.db_path = db_path
        self.model_name = model_name
        self.db = DetectionDatabase(db_path)
        
        # Initialize SentenceTransformer model
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Cache for embeddings to avoid recomputation
        self._embedding_cache = {}
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text, with caching."""
        if text not in self._embedding_cache:
            self._embedding_cache[text] = self.model.encode(text, normalize_embeddings=True)
        return self._embedding_cache[text]
    
    def index_captions(self, force_reindex: bool = False) -> int:
        """
        Create embeddings for all captions in the database.
        
        Args:
            force_reindex: If True, recreate all embeddings even if they exist
            
        Returns:
            Number of captions indexed
        """
        logger.info("Starting caption indexing...")
        
        # Get all detections with captions
        detections = self.db.get_detections_with_captions()
        
        if not detections:
            logger.warning("No detections with captions found")
            return 0
        
        indexed_count = 0
        batch_size = 50  # Process in batches to avoid memory issues
        
        for i in range(0, len(detections), batch_size):
            batch = detections[i:i + batch_size]
            
            # Prepare texts and IDs for batch processing
            texts_to_encode = []
            detection_ids = []
            
            for detection in batch:
                detection_id = detection[0]
                caption = detection[6]  # caption column
                
                if caption and (force_reindex or not self._has_embedding(detection_id)):
                    texts_to_encode.append(caption)
                    detection_ids.append(detection_id)
            
            if texts_to_encode:
                # Generate embeddings in batch
                logger.info(f"Encoding batch of {len(texts_to_encode)} captions...")
                embeddings = self.model.encode(texts_to_encode, normalize_embeddings=True)
                
                # Store embeddings in database
                for detection_id, embedding in zip(detection_ids, embeddings):
                    self._store_embedding(detection_id, embedding)
                    indexed_count += 1
        
        logger.info(f"Indexing complete. Indexed {indexed_count} captions.")
        return indexed_count
    
    def _has_embedding(self, detection_id: int) -> bool:
        """Check if detection already has an embedding."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT embeddings FROM detections WHERE id = ?", (detection_id,))
        result = cursor.fetchone()
        
        conn.close()
        return result and result[0] is not None
    
    def _store_embedding(self, detection_id: int, embedding: np.ndarray):
        """Store embedding in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize embedding
        embedding_blob = pickle.dumps(embedding)
        
        cursor.execute(
            "UPDATE detections SET embeddings = ? WHERE id = ?",
            (embedding_blob, detection_id)
        )
        
        conn.commit()
        conn.close()
    
    def search(self, query: str, top_k: int = 20, similarity_threshold: float = 0.3) -> List[Tuple[dict, float]]:
        """
        Search for detections matching the query.
        
        Args:
            query: Natural language query (e.g., "Find all red cars")
            top_k: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of tuples containing (detection_dict, similarity_score)
        """
        logger.info(f"Searching for: '{query}'")
        
        # Generate query embedding
        query_embedding = self._get_embedding(query)
        
        # Get all detections with embeddings
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, object_type, time, crop_of_object, original_video_link, 
                   frame_num_original_video, caption, embeddings, confidence,
                   bbox_x, bbox_y, bbox_width, bbox_height, created_at
            FROM detections 
            WHERE caption IS NOT NULL AND embeddings IS NOT NULL
        """)
        
        detections = cursor.fetchall()
        conn.close()
        
        if not detections:
            logger.warning("No detections with embeddings found. Run index_captions() first.")
            return []
        
        # Calculate similarities
        results = []
        
        for detection in detections:
            try:
                # Deserialize embedding
                embedding_blob = detection[7]  # embeddings column
                
                # Handle both pickled and raw numpy array formats
                if isinstance(embedding_blob, bytes) and len(embedding_blob) == 1536:
                    # Raw float32 array (384 dimensions * 4 bytes)
                    stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                else:
                    # Pickled format
                    stored_embedding = pickle.loads(embedding_blob)
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, stored_embedding)
                
                if similarity >= similarity_threshold:
                    # Convert to dictionary
                    detection_dict = {
                        'id': detection[0],
                        'object_type': detection[1],
                        'time': detection[2],
                        'crop_of_object': detection[3],
                        'original_video_link': detection[4],
                        'frame_num_original_video': detection[5],
                        'caption': detection[6],
                        'confidence': detection[8],
                        'bbox_x': detection[9],
                        'bbox_y': detection[10],
                        'bbox_width': detection[11],
                        'bbox_height': detection[12],
                        'created_at': detection[13]
                    }
                    
                    results.append((detection_dict, float(similarity)))
                    
            except Exception as e:
                logger.warning(f"Error processing detection {detection[0]}: {e}")
                continue
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        results = results[:top_k]
        
        logger.info(f"Found {len(results)} results for query: '{query}'")
        return results
    
    def get_similar_detections(self, detection_id: int, top_k: int = 10) -> List[Tuple[dict, float]]:
        """
        Find detections similar to a given detection.
        
        Args:
            detection_id: ID of the reference detection
            top_k: Maximum number of results to return
            
        Returns:
            List of tuples containing (detection_dict, similarity_score)
        """
        # Get the reference detection's embedding
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT embeddings FROM detections WHERE id = ?", (detection_id,))
        result = cursor.fetchone()
        
        if not result or not result[0]:
            logger.warning(f"No embedding found for detection {detection_id}")
            return []
        
        try:
            embedding_blob = result[0]
            # Handle both pickled and raw numpy array formats
            if isinstance(embedding_blob, bytes) and len(embedding_blob) == 1536:
                # Raw float32 array (384 dimensions * 4 bytes)
                reference_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            else:
                # Pickled format
                reference_embedding = pickle.loads(embedding_blob)
        except Exception as e:
            logger.error(f"Error loading embedding for detection {detection_id}: {e}")
            return []
        
        # Get all other detections with embeddings
        cursor.execute("""
            SELECT id, object_type, time, crop_of_object, original_video_link, 
                   frame_num_original_video, caption, embeddings, confidence,
                   bbox_x, bbox_y, bbox_width, bbox_height, created_at
            FROM detections 
            WHERE caption IS NOT NULL AND embeddings IS NOT NULL AND id != ?
        """, (detection_id,))
        
        detections = cursor.fetchall()
        conn.close()
        
        # Calculate similarities
        results = []
        
        for detection in detections:
            try:
                embedding_blob = detection[7]
                # Handle both pickled and raw numpy array formats
                if isinstance(embedding_blob, bytes) and len(embedding_blob) == 1536:
                    # Raw float32 array (384 dimensions * 4 bytes)
                    stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                else:
                    # Pickled format
                    stored_embedding = pickle.loads(embedding_blob)
                    
                similarity = np.dot(reference_embedding, stored_embedding)
                
                detection_dict = {
                    'id': detection[0],
                    'object_type': detection[1],
                    'time': detection[2],
                    'crop_of_object': detection[3],
                    'original_video_link': detection[4],
                    'frame_num_original_video': detection[5],
                    'caption': detection[6],
                    'confidence': detection[8],
                    'bbox_x': detection[9],
                    'bbox_y': detection[10],
                    'bbox_width': detection[11],
                    'bbox_height': detection[12],
                    'created_at': detection[13]
                }
                
                results.append((detection_dict, float(similarity)))
                
            except Exception as e:
                logger.warning(f"Error processing detection {detection[0]}: {e}")
                continue
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def suggest_queries(self, object_type: str = None) -> List[str]:
        """
        Suggest common search queries based on captions in the database.
        
        Args:
            object_type: Optional filter by object type
            
        Returns:
            List of suggested query strings
        """
        suggestions = [
            "red car",
            "person with backpack",
            "blue vehicle",
            "cat sitting",
            "dog running",
            "bicycle on street",
            "motorcycle parked",
            "person walking",
            "car in driveway",
            "bird flying"
        ]
        
        if object_type:
            # Filter suggestions by object type
            type_specific = {
                'car': ["red car", "blue car", "white car", "parked car", "moving car"],
                'person': ["person walking", "person with backpack", "person sitting", "person running"],
                'cat': ["cat sitting", "cat sleeping", "orange cat", "black cat"],
                'dog': ["dog running", "small dog", "large dog", "dog playing"],
                'bicycle': ["red bicycle", "bicycle on path", "parked bicycle"],
                'motorcycle': ["motorcycle parked", "red motorcycle", "motorcycle on road"]
            }
            return type_specific.get(object_type, suggestions[:5])
        
        return suggestions

def main():
    """Example usage and testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Semantic search for detection captions')
    parser.add_argument('--db', default='detections.db', help='Database path')
    parser.add_argument('--index', action='store_true', help='Index all captions')
    parser.add_argument('--search', type=str, help='Search query')
    parser.add_argument('--top-k', type=int, default=10, help='Number of results')
    parser.add_argument('--threshold', type=float, default=0.3, help='Similarity threshold')
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = SemanticSearchAgent(args.db)
    
    if args.index:
        print("Indexing captions...")
        count = agent.index_captions()
        print(f"Indexed {count} captions")
    
    if args.search:
        print(f"Searching for: '{args.search}'")
        results = agent.search(args.search, top_k=args.top_k, similarity_threshold=args.threshold)
        
        if results:
            print(f"\nFound {len(results)} results:")
            for i, (detection, similarity) in enumerate(results, 1):
                print(f"\n{i}. Similarity: {similarity:.3f}")
                print(f"   Type: {detection['object_type']}")
                print(f"   Time: {detection['time']}")
                print(f"   Caption: {detection['caption']}")
                print(f"   Confidence: {detection['confidence']:.2f}")
        else:
            print("No results found")
    
    if not args.index and not args.search:
        print("Usage examples:")
        print("  python ai_agent.py --index                    # Index all captions")
        print("  python ai_agent.py --search 'red car'         # Search for red cars")
        print("  python ai_agent.py --search 'person walking'  # Search for walking people")

if __name__ == "__main__":
    main()