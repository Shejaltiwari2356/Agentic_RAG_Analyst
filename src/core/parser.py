import os
import yaml
import nest_asyncio
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownNodeParser
from src.core.database import DatabaseManager

nest_asyncio.apply()

class PDFParser:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.db_manager = DatabaseManager(config_path)
        
        # EXACT PROMPT FROM YOUR GITHUB REPO
        self.financial_audit_prompt = """
        <role>Senior Financial Compliance Auditor</role>
        <instructions>
        1. Identify the 'Scale' of every table (e.g., 'In millions', 'In thousands'). 
        2. MANDATORY: Prepend the scale to the table markdown so it is never separated.
        3. Convert all tables to high-fidelity Markdown pipes (|).
        4. Preserve all section headers (#, ##, ###) for hierarchical splitting.
        </instructions>
        """

    def get_contextual_metadata(self, content: str):
        """Automatically tags nodes based on your existing logic."""
        content_lower = content.lower()
        metadata = {"section_type": "general_text"}
        
        if "balance sheets" in content_lower:
            metadata.update({"section_type": "financial_statement", "table_name": "balance_sheet"})
        elif "statements of operations" in content_lower or "income statement" in content_lower:
            metadata.update({"section_type": "financial_statement", "table_name": "income_statement"})
        elif "cash flows" in content_lower:
            metadata.update({"section_type": "financial_statement", "table_name": "cash_flow"})
        elif "risk factors" in content_lower:
            metadata.update({"section_type": "risk_analysis"})
            
        return metadata

    def run_smart_ingestion(self):
        print("🚀 Stage 1: LlamaParse Cloud Processing...")
        parser = LlamaParse(
            result_type="markdown",
            system_prompt_append=self.financial_audit_prompt,
            api_key=os.getenv("LLAMA_CLOUD_API_KEY")
        )
        documents = parser.load_data(self.config['data']['pdf_path'])
        
        print("📦 Stage 2: Hierarchical Markdown Splitting...")
        node_parser = MarkdownNodeParser()
        nodes = node_parser.get_nodes_from_documents(documents)
        
        print(f"📥 Stage 3: Ingesting {len(nodes)} nodes into ChromaDB...")
        for i, node in enumerate(nodes):
            custom_meta = self.get_contextual_metadata(node.text)
            custom_meta["page_label"] = node.metadata.get("page_label", "unknown")
            
            self.db_manager.collection.add(
                documents=[node.text], 
                metadatas=[custom_meta], 
                ids=[f"node_{i}"]
            )
        print("✅ Ingestion Complete with your Precision Prompts.")