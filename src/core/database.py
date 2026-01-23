import os
import yaml
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

# --- NEW EMBEDDING CLASS FOR OPTION 1 ---
class JinaEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        print("Loading Jina model (this may take a minute the first time)...")
        # This model supports 8,192 tokens (~32,000 characters).
        # It will easily handle your 11,000 character chunks.
        self.model = SentenceTransformer('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)

    def __call__(self, input: Documents) -> Embeddings:
        # Generate embeddings locally
        embeddings = self.model.encode(input).tolist()
        return embeddings

class DatabaseManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.client = chromadb.PersistentClient(path=self.config['embedding']['chroma_path'])
        
        # USE JINA INSTEAD OF GEMINI
        self.embedding_fn = JinaEmbeddingFunction()
        
        self.collection = self.client.get_or_create_collection(
            name=self.config['embedding']['collection_name'],
            embedding_function=self.embedding_fn
        )

    def query(self, query_text: str, n_results: int = 30):
        # Jina is simple: it doesn't need special 'task_type' flags like Google.
        # Just pass the text directly.
        results = self.collection.query(
            query_texts=[query_text], 
            n_results=n_results
        )
        
        formatted = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        return formatted