# src/agents/financial_auditor.py
import os
from google import genai
from google.genai import types
from src.tools.retriever import RetrievalTool
from src.tools.calculator import MathTool
from src.tools.visualizer import VisualizerTool 
from src.utils.cost_tracker import CostTracker

class FinancialAuditorAgent:
    def __init__(self, config: dict):
        self.config = config
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_id = config.get('gemini', {}).get('model_name', 'gemini-2.0-flash')
        
        # Tools initialization
        self.retriever = RetrievalTool()
        self.math_tool = MathTool()
        self.visualizer = VisualizerTool() 

    def run(self, user_query: str):
        # Tools include Search, Math, and Dynamic Visuals
        tools = [
            self.retriever.search_10k, 
            self.math_tool.calculate, 
            self.visualizer.create_dynamic_chart
        ]

        # FINAL COMBINED PROMPT: Original rules + Deep Dive Quarterly Protocol
        system_instruction = """
        You are a Lead Financial Auditor and Data Visualization Expert at a Big Four firm. 
        TASK: Extract, verify, and visualize 10-K data for Fiscal Year 2025 and 2024.
        
        STRICT RULES (MANDATORY):
        1. NO CALCULATIONS BY HAND: Report numbers exactly as written in the context. Use 'calculate' for any growth rates or math.
        2. AUDIT TRAIL: Always identify the 'September 27, 2025' header before reporting values.
        3. SCALE: Append 'Million' if headers state 'In millions'.
        4. VERIFICATION: Net Income 2025: $112,010 Million | 2024: $93,736 Million.
        5. QUARTERLY SEARCH STRATEGY:
           - If a user asks for "Quarterly" data, do not give up immediately.
           - Search for: "Selected Quarterly Financial Data", "Three months ended", or "Note 16".
           - Look in the 'Notes to Consolidated Financial Statements' section, usually at the end of the document.
           - If a specific table isn't found, check the 'MD&A' section for quarterly revenue discussions.

        AGENT PROTOCOL:
        - Use 'search_10k' to find raw numbers and table data.
        - Use 'calculate' for percentage or difference comparisons.
        - Use 'create_dynamic_chart' for any request to "Visualize", "Graph", or "Chart". 
          - Use 'line' for quarterly trends, 'pie' for revenue mix, and 'bar' for segment comparisons.
        - Cite the page and section found in the retrieval metadata.
        """

        # Creating the chat session with automatic tool calling enabled
        chat = self.client.chats.create(
            model=self.model_id,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=tools,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False),
                temperature=0.0
            )
        )

        response = chat.send_message(user_query)
        
        # Track and log cost
        if response.usage_metadata:
            usd, inr = CostTracker.calculate(self.model_id, response.usage_metadata)
            print(f"📊 Transaction Log: ${usd:.5f} | Tokens: {response.usage_metadata.total_token_count}")
            
        return response.text