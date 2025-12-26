import os
import yaml
import chromadb
from google import genai
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "text-embedding-004" 

    def __call__(self, input: Documents) -> Embeddings:
        response = self.client.models.embed_content(
            model=self.model,
            contents=input,
            config={"task_type": "RETRIEVAL_DOCUMENT"}
        )
        return [item.values for item in response.embeddings]

class DatabaseManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Pull key name from config, then fetch from env
        env_var_name = self.config.get('gemini', {}).get('api_key_env', 'GOOGLE_API_KEY')
        api_key = os.getenv(env_var_name)
        
        self.client = chromadb.PersistentClient(path=self.config['embedding']['chroma_path'])
        self.gemini_ef = GeminiEmbeddingFunction(api_key=api_key)
        
        self.collection = self.client.get_or_create_collection(
            name=self.config['embedding']['collection_name'],
            embedding_function=self.gemini_ef
        )

    def query(self, query_text: str, n_results: int = 30):
        results = self.collection.query(query_texts=[query_text], n_results=n_results)
        formatted = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i]
                })
        return formatted