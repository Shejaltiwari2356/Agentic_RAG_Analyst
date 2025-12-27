# src/tools/retriever.py
import os
from google import genai
from flashrank import Ranker, RerankRequest
from src.core.database import DatabaseManager

class RetrievalTool:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.db = DatabaseManager(config_path)
        # Use a more robust reranker model if possible, but MiniLM is fine for local
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def search_10k(self, query: str) -> str:
        # 1. Multi-Query Expansion (Forces the DB to look for different terms)
        # We ask Gemini to generate search terms that specifically target TABLES
        prompt = f"Generate 3 search queries to find the numerical tables for: '{query}'. Return ONLY queries."
        response = self.client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        queries = [query] + response.text.strip().split("\n")

        # 2. Broad Retrieval (Children)
        all_child_metas = []
        for q in queries[:3]:
            results = self.db.collection.query(
                query_texts=[q],
                n_results=40, # High recall to capture more candidates
                where={"type": "child"}
            )
            all_child_metas.extend(results['metadatas'][0])

        # 3. Parent Retrieval & "Signal-to-Noise" Filtering
        unique_parent_ids = list(set([m['parent_id'] for m in all_child_metas]))
        parents = self.db.collection.get(ids=unique_parent_ids)
        
        valid_passages = []
        for i, doc in enumerate(parents['documents']):
            # SENIOR HACK: The 'Table Density' check
            # Real financial data nodes have multiple pipes (|). Headers do not.
            pipe_count = doc.count("|")
            
            # If it's a financial question, prioritize tables. 
            # If it's a risk question, prioritize long paragraphs.
            if pipe_count > 5 or len(doc) > 1500:
                valid_passages.append({
                    "id": i, 
                    "text": doc, 
                    "meta": parents['metadatas'][i]
                })

        # 4. Reranking (The Judge)
        if not valid_passages:
            # Fallback to raw parents if filter is too strict
            valid_passages = [{"id": i, "text": d, "meta": m} for i, (d, m) in enumerate(zip(parents['documents'], parents['metadatas']))]

        rerank_request = RerankRequest(query=query, passages=valid_passages)
        reranked = self.ranker.rerank(rerank_request)

        # 5. Format Top 6 for Gemini
        # We explicitly tell the Agent which chunk had the highest Precision Score
        formatted = []
        for i, r in enumerate(reranked[:6]):
            meta = r.get('meta', {})
            formatted.append(
                f"<DATA_CHUNK ID='{i}' RERANK_SCORE='{round(r['score'], 4)}'>\n"
                f"SOURCE: Page {meta.get('page_label', 'NA')}\n"
                f"CONTENT: {r['text'].strip()}\n"
                f"</DATA_CHUNK>"
            )

        return "\n\n".join(formatted)