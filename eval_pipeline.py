import os
import json
import yaml
import re
import asyncio
from datetime import datetime
from typing import List, Dict, Any
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import GeminiModel
from deepeval.evaluate import AsyncConfig

# ==========================================
# 1. HARDENED DIAGNOSTIC JUDGE (V2)
# ==========================================
class ArjunEvalJudge(GeminiModel):
    r"""
    An ultra-robust Judge for Gemini 2.0 that aggressively enforces JSON 
    formatting to prevent DeepEval parsing errors.
    """
    def __init__(self, model_name="gemini-2.0-flash"):
        # We set temperature to 0 for maximum consistency
        super().__init__(model=model_name, temperature=0)

    def _clean_json_string(self, res: str) -> str:
        """
        Uses multiple passes to extract and validate JSON content.
        """
        if not res or not isinstance(res, str):
            return res
        
        # Pass 1: Remove Markdown code blocks
        res = re.sub(r'```json\s*', '', res, flags=re.IGNORECASE)
        res = re.sub(r'```\s*', '', res)
        
        # Pass 2: Locate the first '{' and last '}'
        start_idx = res.find('{')
        end_idx = res.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            res = res[start_idx:end_idx + 1]
        
        # Pass 3: Clean whitespace and control characters
        res = res.strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

        # Pass 4: Final validation - if it's not valid JSON, we try to force it
        try:
            json.loads(res)
            return res
        except json.JSONDecodeError:
            # If still failing, return the raw stripped string and let DeepEval try
            # but log it for your diagnostics
            print(f"⚠️ Warning: Model returned potentially malformed JSON: {res[:100]}...")
            return res

    def generate(self, prompt: str) -> str:
        # Strict enforcement in the prompt
        system_instruction = (
            "INSTRUCTION: You are a financial auditor. Return ONLY a valid JSON object. "
            "Do not include any preamble, explanations, or markdown formatting outside the JSON. "
            "Note: $112,010 in a table labeled 'In Millions' is EQUAL to '$112,010 Million'.\n\n"
        )
        res = super().generate(system_instruction + prompt)
        return self._clean_json_string(res)

    async def a_generate(self, prompt: str) -> str:
        system_instruction = (
            "INSTRUCTION: You are a financial auditor. Return ONLY a valid JSON object. "
            "Do not include any preamble, explanations, or markdown formatting outside the JSON. "
            "Note: $112,010 in a table labeled 'In Millions' is EQUAL to '$112,010 Million'.\n\n"
        )
        res = await super().a_generate(system_instruction + prompt)
        return self._clean_json_string(res)

# ==========================================
# 2. FIXED DIAGNOSTIC REPORT GENERATOR
# ==========================================
class EvaluationDiagnoser:
    def __init__(self, output_dir="diagnostics"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_report(self, results_object):
        filename = f"diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        test_results_list = getattr(results_object, 'test_results', [])
        
        report = {
            "summary": {
                "total_tests": len(test_results_list),
                "passed": sum(1 for r in test_results_list if r.success),
                "failed": sum(1 for r in test_results_list if not r.success),
                "timestamp": str(datetime.now())
            },
            "detailed_results": []
        }

        for r in test_results_list:
            metrics_summary = {}
            m_data_list = getattr(r, 'metrics_data', [])
            for m_data in m_data_list:
                metrics_summary[m_data.name] = {
                    "score": m_data.score,
                    "threshold": m_data.threshold,
                    "reason": m_data.reason,
                    "success": m_data.success
                }

            report["detailed_results"].append({
                "input": r.input,
                "actual_output": r.actual_output,
                "expected_output": r.expected_output,
                "success": r.success,
                "metrics": metrics_summary,
                "retrieval_context_preview": r.retrieval_context[:1] if r.retrieval_context else []
            })

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"\n✅ Diagnostic report saved to: {filepath}")

# ==========================================
# 3. TEST DATA
# ==========================================
def get_100_questions():
    # Returning the subset you are currently testing
    return [
        # {"input": "What was the total value of 'Other non-current assets' in 2025?", "expected_output": "$83,727 Million "},
        # {"input": "What was the year-over-year growth percentage for Services in 2025?", "expected_output": "14% "},
        # {"input": "How much did iPhone sales grow compared to 2024?", "expected_output": "4% "},
        # {"input": "Mention specific cybersecurity threats identified as risks in the 10-K.", "expected_output": "Ransomware, computer viruses, or unauthorized access "},
        # {"input": "What is the trading symbol for Apple common stock?", "expected_output": "AAPL "},
        # {"input": "Who is the Chief Financial Officer (CFO) of Apple?", "expected_output": "Kevan Parekh "},
        # {"input": "Identify the Chief Executive Officer of the Company.", "expected_output": "Timothy D. Cook "},
        # {"input": "What is the potential maximum fine for a violation of the Digital Markets Act (DMA)?", "expected_output": "Up to 10% of annual worldwide net sales "},
        # {"input": "State the total 'Accounts receivable, net' for 2025.", "expected_output": "$39,777 Million "},
        # {"input": "What was the total liabilities reported for 2025?", "expected_output": "$285,508 Million "},
        # {"input": "Identify the 'Other non-current liabilities' balance for 2025.", "expected_output": "$41,549 Million "},
        # {"input": "Identify the specific fine amount for the Commission Article 5(4) investigation.", "expected_output": "€500 million "},
        # {"input": "What was the quarterly cash dividend per share declared in May 2025?", "expected_output": "$0.26 per share "},
        # {"input": "Identify the total 'Unrecognized compensation cost' related to RSUs as of September 2025.", "expected_output": "$21.8 Billion "},
        # {"input": "What was the aggregate market value of stock held by non-affiliates as of March 28, 2025?", "expected_output": "$3,253,431,000,000 "}

    
        # # GROUP 1: Revenue & Financial Performance
        # {"input": "What was Apple's total net sales for the fiscal year ended September 27, 2025?", "expected_output": "$416,161 Million [cite: 657, 793]"},
        # {"input": "Identify the total net income for 2025.", "expected_output": "$112,010 Million [cite: 793, 890]"},
        # {"input": "What was the percentage increase in total net sales from 2024 to 2025?", "expected_output": "6% [cite: 657, 672]"},
        # {"input": "What was the gross margin percentage for 2025?", "expected_output": "46.9% [cite: 687]"},
        # {"input": "What was the basic earnings per share in 2025?", "expected_output": "$7.49 [cite: 793, 890]"},
        # {"input": "What was the diluted earnings per share in 2025?", "expected_output": "$7.46 [cite: 793, 890]"},
        # {"input": "How much was the operating income for 2025?", "expected_output": "$133,050 Million [cite: 793, 1081]"},
        # {"input": "Identify the total cost of sales for 2025.", "expected_output": "$220,960 Million [cite: 793]"},
        # {"input": "State the income before provision for income taxes in 2025.", "expected_output": "$132,729 Million [cite: 793]"},
        # {"input": "What was the total provision for income taxes in 2025?", "expected_output": "$20,719 Million [cite: 793, 967]"},

        # # # GROUP 2: Products & Services
        # {"input": "What were the net sales for iPhone in 2025?", "expected_output": "$209,586 Million [cite: 672, 793]"},
        # {"input": "How much revenue did Services generate in 2025?", "expected_output": "$109,158 Million [cite: 672, 793]"},
        # {"input": "What was the net sales figure for Mac in 2025?", "expected_output": "$33,708 Million [cite: 672, 793]"},
        # {"input": "Identify the net sales for iPad in 2025.", "expected_output": "$28,023 Million [cite: 672, 793]"},
        # {"input": "What was the revenue for 'Wearables, Home and Accessories' in 2025?", "expected_output": "$35,686 Million [cite: 672, 793]"},
        # {"input": "Which product category saw a 4% decrease in sales in 2025?", "expected_output": "Wearables, Home and Accessories [cite: 672, 681]"},
        # {"input": "What was the year-over-year growth percentage for Services in 2025?", "expected_output": "14% [cite: 672]"},
        # {"input": "How much did iPhone sales grow compared to 2024?", "expected_output": "4% [cite: 672]"},
        # {"input": "What was the net sales growth for Mac in 2025?", "expected_output": "12% [cite: 672]"},
        # {"input": "Did iPad net sales increase or decrease in 2025, and by how much?", "expected_output": "Increased by 5% [cite: 672]"},

        # # # GROUP 3: Regional Segments
        # {"input": "What were the net sales for the Americas segment in 2025?", "expected_output": "$178,353 Million [cite: 657, 1081]"},
        # {"input": "How much revenue was generated in Europe in 2025?", "expected_output": "$111,032 Million [cite: 657, 1081]"},
        # {"input": "What were the net sales in Greater China in 2025?", "expected_output": "$64,377 Million [cite: 657, 1081]"},
        # {"input": "Identify the net sales for Japan in 2025?", "expected_output": "$28,703 Million [cite: 657, 1081]"},
        # {"input": "What was the net sales for 'Rest of Asia Pacific' in 2025?", "expected_output": "$33,696 Million [cite: 657, 1081]"},
        # {"input": "Which geographic segment had the highest operating income in 2025?", "expected_output": "Americas ($72,480 Million) [cite: 1081]"},
        # {"input": "State the operating income for the Europe segment in 2025.", "expected_output": "$47,739 Million [cite: 1081]"},
        # {"input": "What was the operating income for Greater China in 2025?", "expected_output": "$26,917 Million [cite: 1081]"},
        # {"input": "How much was the operating income for Japan in 2025?", "expected_output": "$13,955 Million [cite: 1081]"},
        # {"input": "What was the operating income for 'Rest of Asia Pacific' in 2025?", "expected_output": "$14,586 Million [cite: 1081]"},

        # # # GROUP 4: Operating Expenses((later) )
        # {"input": "What was the total Research and Development (R&D) expense in 2025?", "expected_output": "$34,550 Million [cite: 698, 793]"},
        # {"input": "How much did Apple spend on Selling, General and Administrative (SG&A) in 2025?", "expected_output": "$27,601 Million [cite: 698, 793]"},
        # {"input": "What were the total operating expenses in 2025?", "expected_output": "$62,151 Million [cite: 698, 793]"},
        # {"input": "What drove the increase in R&D expenses in 2025?", "expected_output": "Increases in headcount-related expenses and infrastructure-related costs [cite: 700]"},
        # {"input": "What drove the increase in SG&A expenses in 2025?", "expected_output": "Increases in headcount-related expenses and variable selling expenses [cite: 702]"},
        # {"input": "What was the R&D expense in 2024?", "expected_output": "$31,370 Million [cite: 698, 793]"},
        # {"input": "What was the SG&A expense in 2024?", "expected_output": "$26,097 Million [cite: 698, 793]"},
        # {"input": "Identify the 'Other income/(expense), net' for 2025.", "expected_output": "-$321 Million [cite: 793]"},
        # {"input": "What was the effective tax rate reported for 2025?", "expected_output": "15.6% [cite: 706, 971]"},
        # {"input": "What was the effective tax rate in 2024?", "expected_output": "24.1% [cite: 706, 971]"},

        # # # GROUP 5: Balance Sheet - Assets
        # {"input": "What was the total assets as of September 27, 2025?", "expected_output": "$359,241 Million [cite: 804]"},
        # {"input": "Identify the total cash and cash equivalents for 2025.", "expected_output": "$35,934 Million [cite: 804, 815]"},
        # {"input": "What was the value of current marketable securities in 2025?", "expected_output": "$18,763 Million [cite: 804, 896]"},
        # {"input": "State the total 'Accounts receivable, net' for 2025.", "expected_output": "$39,777 Million [cite: 804]"},
        # {"input": "What was the value of total inventories in 2025?", "expected_output": "$5,718 Million [cite: 804]"},
        # {"input": "What was the 'Vendor non-trade receivables' balance in 2025?", "expected_output": "$33,180 Million [cite: 804]"},
        # {"input": "State the total current assets for 2025.", "expected_output": "$147,957 Million [cite: 804]"},
        # {"input": "What was the value of non-current marketable securities in 2025?", "expected_output": "$77,723 Million [cite: 804, 896]"},
        # {"input": "Identify the 'Property, plant and equipment, net' in 2025.", "expected_output": "$49,834 Million [cite: 804, 947]"},
        # {"input": "What was the value of 'Other non-current assets' in 2025?", "expected_output": "$83,727 Million [cite: 804, 953]"},

        # # # GROUP 6: Balance Sheet - Liabilities & Equity
        # {"input": "What was the total liabilities for 2025?", "expected_output": "$285,508 Million [cite: 804]"},
        # {"input": "Identify the total 'Accounts payable' for 2025.", "expected_output": "$69,860 Million [cite: 804]"},
        # {"input": "What was the 'Other current liabilities' balance in 2025?", "expected_output": "$66,387 Million [cite: 804, 955]"},
        # {"input": "Identify the current portion of term debt in 2025.", "expected_output": "$12,350 Million [cite: 804, 1014]"},
        # {"input": "State the 'Total current liabilities' for 2025.", "expected_output": "$165,631 Million [cite: 804]"},
        # {"input": "What was the long-term debt (non-current) in 2025?", "expected_output": "$78,328 Million [cite: 804, 1014]"},
        # {"input": "Identify the 'Other non-current liabilities' in 2025.", "expected_output": "$41,549 Million [cite: 804, 993]"},
        # {"input": "What was the total shareholders' equity for 2025?", "expected_output": "$73,733 Million [cite: 804, 809]"},
        # {"input": "Identify the 'Accumulated deficit' as of Sept 27, 2025.", "expected_output": "-$14,264 Million [cite: 804, 809]"},
        # {"input": "What was the 'Common stock and additional paid-in capital' in 2025?", "expected_output": "$93,568 Million [cite: 804, 809]"},

        # # # # GROUP 7: Cash Flow
        # {"input": "How much cash was generated by operating activities in 2025?", "expected_output": "$111,482 Million [cite: 815]"},
        # {"input": "How much cash was generated by investing activities in 2025?", "expected_output": "$15,195 Million [cite: 815]"},
        # {"input": "How much cash was used in financing activities in 2025?", "expected_output": "$120,686 Million [cite: 815]"},
        # {"input": "How much did Apple pay for common stock repurchases in 2025?", "expected_output": "$90,711 Million [cite: 815]"},
        # {"input": "What were the dividends paid in 2025?", "expected_output": "$15,421 Million [cite: 815]"},
        # {"input": "State the 'Depreciation and amortization' for 2025.", "expected_output": "$11,698 Million [cite: 815]"},
        # {"input": "How much was the 'Share-based compensation expense' in 2025?", "expected_output": "$12,863 Million [cite: 815, 1046]"},
        # {"input": "What was the 'Proceeds from issuance of term debt, net' in 2025?", "expected_output": "$4,481 Million [cite: 815]"},
        # {"input": "How much did Apple spend on PPE (Capital Expenditures) in 2025?", "expected_output": "$12,715 Million [cite: 815]"},
        # {"input": "What was the cash paid for income taxes, net, in 2025?", "expected_output": "$43,369 Million [cite: 815]"},

        # # # GROUP 8: Risk Factors & Legal
        # {"input": "What is the new risk regarding tariffs mentioned in 2025?", "expected_output": "New U.S. tariffs on imports from China, India, Japan, South Korea, Taiwan, Vietnam and the EU [cite: 224, 648]"},
        # {"input": "Identify the fine amount for the Commission Article 5(4) investigation.", "expected_output": "€500 million [cite: 554]"},
        # {"input": "Who filed a civil antitrust lawsuit against Apple in March 2024?", "expected_output": "The DOJ and a number of state and district attorneys general [cite: 560]"},
        # {"input": "What is the potential maximum fine for a DMA violation?", "expected_output": "Up to 10% of annual worldwide net sales [cite: 556]"},
        # {"input": "Mention a cybersecurity risk identified in the 10-K.", "expected_output": "Ransomware, computer viruses, or unauthorized access [cite: 232, 360]"},
        # {"input": "What does the Epic Games lawsuit refer to regarding the 2021 injunction?", "expected_output": "Enjoining the Company from prohibiting developers from including buttons or external links for alternative purchasing mechanisms [cite: 566]"},
        # {"input": "What is the risk of having a single source for components?", "expected_output": "Subject to significant supply and pricing risks, and potential industry-wide shortage [cite: 142, 144, 286]"},
        # {"input": "How does the Digital Markets Act (DMA) affect Apple in the EU?", "expected_output": "Requires changes to iOS, iPadOS, the App Store and Safari [cite: 438]"},
        # {"input": "What impact does a strong U.S. dollar have on Apple?", "expected_output": "Adversely affects the U.S. dollar value of foreign currency-denominated sales and earnings [cite: 492]"},
        # {"input": "Identify the court hearing the Epic Games appeal in Oct 2025.", "expected_output": "U.S. Court of Appeals for the Ninth Circuit [cite: 574, 575]"},

        # # # GROUP 9: Employee & Corporate
        # {"input": "How many full-time equivalent employees did Apple have in 2025?", "expected_output": "166,000 [cite: 175]"},
        # {"input": "Where is Apple's principal executive office located?", "expected_output": "One Apple Park Way, Cupertino, California [cite: 18]"},
        # {"input": "What is the trading symbol for Apple common stock?", "expected_output": "AAPL [cite: 24, 586]"},
        # {"input": "Who is the Chief Executive Officer?", "expected_output": "Timothy D. Cook [cite: 1242]"},
        # {"input": "Who is the Chief Financial Officer (CFO)?", "expected_output": "Kevan Parekh [cite: 1181, 1242]"},
        # {"input": "How many shareholders of record were there on Oct 17, 2025?", "expected_output": "22,429 [cite: 588]"},
        # {"input": "What was the quarterly dividend declared in May 2025?", "expected_output": "$0.26 per share [cite: 733, 735]"},
        # {"input": "What was the amount of the new share repurchase program authorized in May 2025?", "expected_output": "$100 billion [cite: 594, 735]"},
        # {"input": "When did Apple's fiscal year 2025 end?", "expected_output": "September 27, 2025 [cite: 6, 65]"},
        # {"input": "Identify the exchange where Apple's common stock is registered.", "expected_output": "The Nasdaq Stock Market LLC [cite: 24, 586]"},

        # # # GROUP 10: Ratios & Synthesis
        # {"input": "Identify the 'Unrecognized compensation cost' for RSUs as of September 2025.", "expected_output": "$21.8 Billion [cite: 1047]"},
        # {"input": "What was the grant-date fair value per RSU for grants made in 2025?", "expected_output": "$226.68 [cite: 1038]"},
        # {"input": "What was the aggregate market value of stock held by non-affiliates as of March 28, 2025?", "expected_output": "$3,253,431,000,000 [cite: 56]"},
        # {"input": "How much was utilized under the May 2025 share repurchase program by Sept 27, 2025?", "expected_output": "$221 million [cite: 595]"},
        # {"input": "Identify the total number of common stock shares outstanding as of Oct 17, 2025.", "expected_output": "14,776,353,000 [cite: 59]"},
        # {"input": "Identify the total principal amount for term debt notes in 2025.", "expected_output": "$91,281 Million [cite: 713, 1014, 1019]"},
        # {"input": "What was the net change in cash for the year 2025?", "expected_output": "$5,991 Million increase [cite: 815]"},
        # {"input": "How much was the provision for income taxes in 2024?", "expected_output": "$29,749 Million [cite: 793, 967]"},
        # {"input": "What was the total gross property, plant and equipment value in 2025?", "expected_output": "$125,848 Million [cite: 947]"},
        # {"input": "What was the percentage of net sales through direct distribution channels in 2025?", "expected_output": "40% [cite: 124]"}
    ]
# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================
def run_arjun_evaluation():
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY environment variable not set.")
        return

    try:
        from src.agents.financial_auditor import FinancialAuditorAgent
    except ImportError:
        print("❌ Error: Could not import FinancialAuditorAgent. Ensure project structure is correct.")
        return
    
    agent = FinancialAuditorAgent({}) 
    judge_model = ArjunEvalJudge(model_name="gemini-2.0-flash")
    diagnoser = EvaluationDiagnoser()

    metrics = [
        FaithfulnessMetric(threshold=0.7, model=judge_model),
        AnswerRelevancyMetric(threshold=0.7, model=judge_model),
        ContextualPrecisionMetric(threshold=0.7, model=judge_model)
    ]

    test_cases = []
    raw_test_data = get_100_questions()
    
    print(f"🚀 Running Agent on {len(raw_test_data)} Test Cases...")
    
    for i, entry in enumerate(raw_test_data):
        print(f"[{i+1}/{len(raw_test_data)}] Query: {entry['input'][:60]}...")
        try:
            result = agent.run(entry["input"])
            
            test_case = LLMTestCase(
                input=entry["input"],
                actual_output=result.get("response", ""),
                expected_output=entry["expected_output"],
                retrieval_context=result.get("sources", [])
            )
            test_cases.append(test_case)
        except Exception as e:
            print(f"   ⚠️ Agent Execution Failed for input: {entry['input']}. Error: {e}")

    print("\n📊 Starting DeepEval Metrics Calculation...")
    try:
        # Reduced concurrency to 2 to avoid rate limits and parsing overlaps
        results = evaluate(
            test_cases=test_cases, 
            metrics=metrics, 
            async_config=AsyncConfig(max_concurrent=2) 
        )
        
        diagnoser.generate_report(results)
        
    except Exception as e:
        print(f"\n❌ Evaluation crashed: {e}")
        print("Tip: If the error persists, try switching model_name to 'gemini-1.5-pro' for evaluation.")

if __name__ == "__main__":
    run_arjun_evaluation()