# src/utils/cost_tracker.py

class CostTracker:
    # Pricing for Gemini 2.0 Flash
    PRICING = {
        "gemini-2.0-flash": {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
    }

    @staticmethod
    def calculate(model_id, usage):
        rates = CostTracker.PRICING.get("gemini-2.0-flash")
        
        if not usage:
            return 0.0, 0.0

        # Safe fetch using getattr with double null-safety
        p_tokens = getattr(usage, 'prompt_token_count', 0) or 0
        c_tokens = getattr(usage, 'candidates_token_count', 0) or 0
        
        in_cost = p_tokens * rates["input"]
        out_cost = c_tokens * rates["output"]
        
        total_usd = in_cost + out_cost
        return total_usd, total_usd * 90.0  # Dec 2025 USD to INR rate