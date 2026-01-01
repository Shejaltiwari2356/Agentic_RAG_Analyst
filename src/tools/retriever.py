# src/tools/retriever.py
from sentence_transformers import CrossEncoder
from src.core.database import DatabaseManager

class RetrievalTool:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.db = DatabaseManager(config_path)
        # Initialize reranker
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("✓ CrossEncoder reranker loaded")

    def search_10k(self, query: str) -> str:
        """
        Two-stage retrieval: vector search → reranking for precision.
        """
        print(f"🔍 [Tool: Retriever] Performing precision search for: {query}")

        # Stage 1: Retrieve MORE candidates (50 instead of 10)
        initial_results = self.db.query(query_text=query, n_results=50)
        print(f"   Retrieved {len(initial_results)} candidates for reranking")

        # Stage 2: Rerank with CrossEncoder
        pairs = [(query, r['text']) for r in initial_results]
        scores = self.reranker.predict(pairs)
        
        # Attach scores and sort
        for i, r in enumerate(initial_results):
            r['rerank_score'] = scores[i]
        
        reranked = sorted(initial_results, key=lambda x: x['rerank_score'], reverse=True)
        top_results = reranked[:7]  # Keep top 7 after reranking
        
        print(f"   Reranked → top score: {top_results[0]['rerank_score']:.3f}")

        # Format output
        formatted = []
        for i, r in enumerate(top_results):
            meta = r.get('metadata', {})
            page = meta.get('page_label', 'Unknown')
            section = meta.get('section_type', 'General')

            node_str = (
                f"<DATA_CHUNK ID='{i}' RELEVANCE='{r['rerank_score']:.3f}'>\n"
                f"SOURCE: Page {page}, Section: {section}\n"
                f"CONTENT: {r['text'].strip()}\n"
                f"</DATA_CHUNK>"
            )
            formatted.append(node_str)

        return "\n\n".join(formatted)
