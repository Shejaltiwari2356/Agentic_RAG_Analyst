# src/core/parser.py
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
        
        # PROMPT: Structured for high-density 10-K extraction
        self.financial_audit_prompt = """
        <role>Senior Financial Compliance Auditor & Data Architect</role>
        <context>Extracting Apple Inc. 2025 10-K for Agentic RAG.</context>
        <directives>
        1. Map ALL-CAPS BOLD to # and Mixed-Case Bold to ##.
        2. Preserve table scale (In Millions) in every column header.
        3. Convert (5,000) to -5000 and dashes (—) to 0.
        4. Append footnote text immediately after the related table.
        5. Tag paragraphs in Item 1A with [CATEGORY: RISK_FACTOR].
        </directives>
        """

    def get_contextual_metadata(self, content: str):
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
        
        print(f"📥 Stage 3: Small-to-Big Ingestion ({len(nodes)} Parents)...")
        for i, node in enumerate(nodes):
            parent_id = f"parent_{i}"
            parent_text = node.text
            custom_meta = self.get_contextual_metadata(parent_text)
            page_label = node.metadata.get("page_label", "unknown")

            # 1. STORE PARENT (The Big Context)
            self.db_manager.collection.add(
                documents=[parent_text],
                metadatas=[{**custom_meta, "type": "parent", "page_label": page_label}],
                ids=[parent_id]
            )
            
            # 2. STORE CHILDREN (Small 500-char Search Windows)
            # This improves retrieval precision for specific numbers/phrases
            child_size = 500
            for j in range(0, len(parent_text), child_size):
                child_text = parent_text[j : j + child_size]
                self.db_manager.collection.add(
                    documents=[child_text],
                    metadatas=[{"type": "child", "parent_id": parent_id}],
                    ids=[f"child_{i}_{j}"]
                )
        
        print("✅ Ingestion Complete: Parent-Child Hierarchy established.")