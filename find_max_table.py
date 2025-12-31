import os
import re
from dotenv import load_dotenv
from llama_parse import LlamaParse

# Load your API keys from .env
load_dotenv()

def find_magic_threshold():
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        print("❌ Error: LLAMA_CLOUD_API_KEY not found in environment.")
        return

    print("🚀 Connecting to LlamaParse to audit table sizes...")
    parser = LlamaParse(result_type="markdown", api_key=api_key)
    
    # We only need to check the main 10-K to find the pattern
    # Assuming your PDF is at this path
    pdf_path = "data/raw/apple_10k.pdf" 
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF not found at {pdf_path}")
        return

    documents = parser.load_data(pdf_path)
    
    max_table_len = 0
    total_tables = 0

    print("🧐 Analyzing document structure...")
    for doc in documents:
        # Tables in Markdown are blocks of text containing multiple '|'
        # We split by double newlines to isolate tables/paragraphs
        blocks = doc.text.split("\n\n")
        
        for block in blocks:
            if block.count("|") > 5:  # High confidence it's a table
                total_tables += 1
                length = len(block)
                if length > max_table_len:
                    max_table_len = length

    print("\n--- 📊 AUDIT RESULTS ---")
    print(f"Total Tables Detected: {total_tables}")
    print(f"Largest Table Length:  {max_table_len} characters")
    print(f"Suggested Chunk Size: {max_table_len + 1000} (Table + Buffer)")
    print("------------------------")

if __name__ == "__main__":
    find_magic_threshold()