from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv  # Add this
import yaml

# Load environment variables BEFORE importing agent
load_dotenv()

from src.agents.financial_auditor import FinancialAuditorAgent

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
agent = FinancialAuditorAgent(config)

class QueryRequest(BaseModel):
    query: str

@app.post("/api/chat")
async def chat(request: QueryRequest):
    result = agent.run(request.query)
    return {"response": result.get("response", ""), "sources": result.get("sources", [])}
