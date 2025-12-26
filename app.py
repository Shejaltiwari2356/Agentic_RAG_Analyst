import argparse
import yaml
import os
from dotenv import load_dotenv
from src.agents.financial_auditor import FinancialAuditorAgent
from src.core.parser import PDFParser

load_dotenv()

class AgenticSystem:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.agent = FinancialAuditorAgent(self.config)
        self.parser = PDFParser(config_path)

    def run(self, query=None, ingest=False):
        if ingest:
            self.parser.run_smart_ingestion()
        elif query:
            print(self.agent.run(query))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--query", type=str)
    args = parser.parse_args()
    AgenticSystem().run(query=args.query, ingest=args.ingest)