# src/tools/retriever.py
from sentence_transformers import CrossEncoder
from src.core.database import DatabaseManager

class RetrievalTool:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.db = DatabaseManager(config_path)
        # Load cross-encoder reranker (downloads on first use ~80MB)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("✅ Reranker model loaded")

    def search_10k(self, query: str) -> str:
        """
        Two-stage retrieval:
        1. Retrieve 30 candidates via vector search
        2. Rerank with cross-encoder, return top 10
        """
        print(f"🔍 [Stage 1] Retrieving candidates for: {query}")
        
        # Stage 1: Get more candidates (30-50)
        candidates = self.db.query(query_text=query, n_results=30)
        
        if not candidates:
            return "No relevant data found."
        
        # Stage 2: Rerank with cross-encoder
        print(f"🎯 [Stage 2] Reranking {len(candidates)} candidates...")
        
        pairs = [[query, c['text']] for c in candidates]
        scores = self.reranker.predict(pairs)
        
        # Sort by reranker score (higher = more relevant)
        ranked = sorted(
            zip(candidates, scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Take top 10 after reranking
        top_results = [r for r, score in ranked[:10]]
        
        # Log reranking effect
        print(f"📊 Top reranked score: {ranked[0][1]:.3f}, Bottom: {ranked[-1][1]:.3f}")

        # Format results
        formatted = []
        for i, r in enumerate(top_results):
            meta = r.get('metadata', {})
            page = meta.get('page_label', 'Unknown')
            section = meta.get('section_type', 'General')

            node_str = (
                f"<DATA_CHUNK ID='{i}' RELEVANCE='HIGH'>\n"
                f"SOURCE: Page {page}, Section: {section}\n"
                f"CONTENT: {r['text'].strip()}\n"
                f"</DATA_CHUNK>"
            )
            formatted.append(node_str)

        return "\n\n".join(formatted)
