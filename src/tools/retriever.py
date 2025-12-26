# src/tools/retriever.py
from src.core.database import DatabaseManager

class RetrievalTool:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.db = DatabaseManager(config_path)

    def search_10k(self, query: str) -> str:
        """
        Retrieves financial data with high density. 
        Reduced count to 20 to prevent 'Context Noise' and 'Lost in the Middle' errors.
        """
        print(f"🔍 [Tool: Retriever] Performing precision search for: {query}")
        
        # Reduced to 20 to ensure Gemini stays focused on the most relevant data
        results = self.db.query(query_text=query, n_results=20)
        
        formatted = []
        for i, r in enumerate(results):
            meta = r.get('metadata', {})
            page = meta.get('page_label', 'Unknown')
            section = meta.get('section_type', 'General')
            
            # Using XML-style tags for better structural parsing by the LLM
            node_str = (
                f"<DATA_CHUNK ID='{i}'>\n"
                f"SOURCE: Page {page}, Section: {section}\n"
                f"CONTENT: {r['text'].strip()}\n"
                f"</DATA_CHUNK>"
            )
            formatted.append(node_str)
            
        return "\n\n".join(formatted)