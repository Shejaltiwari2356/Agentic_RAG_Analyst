import os
import json
from typing import List
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import GeminiModel

# 1. Custom Robust Wrapper for Gemini 2.0 Flash
class ArjunEvalJudge(GeminiModel):
    def __init__(self, model_name="gemini-2.0-flash"):
        # We pass the model name and temperature directly.
        # We avoid 'generation_config' to prevent the Client.__init__ error.
        super().__init__(
            model=model_name,
            temperature=0
        )

    # Fail-safe to clean JSON if the model adds markdown formatting
    def generate(self, prompt: str) -> str:
        res = super().generate(prompt)
        if isinstance(res, str):
            res = res.replace("```json", "").replace("```", "").strip()
        return res

    async def a_generate(self, prompt: str) -> str:
        res = await super().a_generate(prompt)
        if isinstance(res, str):
            res = res.replace("```json", "").replace("```", "").strip()
        return res

# 2. Dataset Initialization (Your 35 Test Cases)
raw_test_data =[
    # --- SEGMENT & PRODUCT PERFORMANCE (1-10) ---
    {
        "input": "Determine the percentage of total net sales contributed by the Americas geographic segment in 2025.",
        "expected_output": "42.86% ($178,353M Americas / $416,161M Total)",
        "context": ["Americas net sales: $178,353M. Total net sales: $416,161M."]
    },
    {
        "input": "Calculate the year-over-year growth rate for iPhone net sales between 2024 and 2025.",
        "expected_output": "5.2% growth ($211,013M in 2025 vs $200,583M in 2024).",
        "context": ["iPhone net sales: $211,013M (2025) and $200,583M (2024)."]
    },
    {
        "input": "Which product category saw the highest percentage growth in 2025?",
        "expected_output": "Services (approx. 12.8% growth).",
        "context": ["Product sales data: iPhone +5%, Services +12.8%, Wearables -2%."]
    },
    {
        "input": "What was the operating income for the 'Europe' segment in 2025?",
        "expected_output": "Data found in Note 15: $34,102 Million (hypothetical).",
        "context": ["Note 15 — Segment Information: Europe operating income table."]
    },
    {
        "input": "Compare iPad sales in 2025 to 2024. Did they meet the internal recovery targets mentioned in the MD&A?",
        "expected_output": "iPad sales were $28,023M (+4.9% vs 2024), aligning with the 'stabilization' goal noted in MD&A.",
        "context": ["MD&A Results of Operations: iPad segment performance."]
    },
    {
        "input": "What percentage of total revenue did 'Services' account for in 2025?",
        "expected_output": "23.1% ($96,105M / $416,161M).",
        "context": ["Services net sales: $96,105M. Total net sales: $416,161M."]
    },
    {
        "input": "Analyze the sales trend in Greater China for 2025. What were the primary headwinds?",
        "expected_output": "Sales flat at $72B; headwinds included regional competition and currency fluctuations.",
        "context": ["MD&A: Greater China regional performance and risk discussion."]
    },
    {
        "input": "Did 'Wearables, Home and Accessories' sales increase or decrease in 2025?",
        "expected_output": "Decreased by 1.5% to $39,241M.",
        "context": ["Wearables, Home and Accessories net sales table."]
    },
    {
        "input": "What was the average revenue per share for Apple in 2025?",
        "expected_output": "$26.83 ($416,161M Revenue / 15,511M Diluted Shares).",
        "context": ["Total sales: $416,161M. Weighted-average diluted shares: 15,511M."]
    },
    {
        "input": "Identify the 'Rest of Asia Pacific' net sales for 2025.",
        "expected_output": "$32,105 Million.",
        "context": ["Consolidated Statements of Operations: Regional Breakdown."]
    },

    # --- PROFITABILITY & MARGINS (11-20) ---
    {
        "input": "Calculate Apple's Gross Margin percentage for 2025.",
        "expected_output": "45.1% ($187,688M Gross Margin / $416,161M Sales).",
        "context": ["Gross margin: $187,688M. Total net sales: $416,161M."]
    },
    {
        "input": "How did the Gross Margin for 'Products' compare to 'Services' in 2025?",
        "expected_output": "Products: 37.5%, Services: 71.2%.",
        "context": ["Note 15: Cost of sales for products vs services."]
    },
    {
        "input": "Calculate the Operating Margin for 2025.",
        "expected_output": "31.8% ($132,333M Operating Income / $416,161M Sales).",
        "context": ["Operating income: $132,333M. Net sales: $416,161M."]
    },
    {
        "input": "What was the Net Profit Margin for Apple in 2025?",
        "expected_output": "26.9% ($112,010M Net Income / $416,161M Sales).",
        "context": ["Net income: $112,010M. Total revenue: $416,161M."]
    },
    {
        "input": "How much did Research and Development (R&D) as a percentage of revenue change from 2024 to 2025?",
        "expected_output": "Increased from 7.6% to 8.2%.",
        "context": ["R&D expense: $34,125M (2025) vs $29,915M (2024)."]
    },
    {
        "input": "Identify the impact of the $10.2B State Aid charge on the 2025 Net Income.",
        "expected_output": "Reduced Net Income by $10.2B, bringing it to $112,010M.",
        "context": ["Note 7 — Income Taxes: European Commission State Aid decision."]
    },
    {
        "input": "What was the 'Selling, General and Administrative' (SG&A) expense for 2025?",
        "expected_output": "$26,105 Million.",
        "context": ["Consolidated Statements of Operations: SG&A line item."]
    },
    {
        "input": "Calculate the Return on Assets (ROA) for 2025.",
        "expected_output": "31.2% ($112,010M Net Income / $359,241M Total Assets).",
        "context": ["Net Income: $112,010M. Total Assets: $359,241M."]
    },
    {
        "input": "Determine the effective tax rate for 2025.",
        "expected_output": "15.6% ($20,719M Provision / $132,729M Income before taxes).",
        "context": ["Provision for income taxes: $20,719M. Income before taxes: $132,729M."]
    },
    {
        "input": "What was the Diluted Earnings Per Share (EPS) for 2025?",
        "expected_output": "$7.22.",
        "context": ["Consolidated Statements of Operations: Diluted EPS line."]
    },

    # --- BALANCE SHEET & LIQUIDITY (21-30) ---
    {
        "input": "Calculate Apple's Current Ratio for 2025.",
        "expected_output": "0.98 ($142,353M Current Assets / $145,308M Current Liabilities).",
        "context": ["Total current assets: $142,353M. Total current liabilities: $145,308M."]
    },
    {
        "input": "What is the 'Quick Ratio' for Apple as of the 2025 fiscal year-end?",
        "expected_output": "0.85 (Excluding $7,102M Inventory from Current Assets).",
        "context": ["Current Assets: $142,353M. Inventory: $7,102M. Current Liab: $145,308M."]
    },
    {
        "input": "How much 'Cash and Cash Equivalents' did Apple hold at the end of 2025?",
        "expected_output": "$32,105 Million.",
        "context": ["Balance Sheet: Cash and cash equivalents."]
    },
    {
        "input": "What was the total value of 'Marketable Securities' (Current + Non-Current) in 2025?",
        "expected_output": "$165,303 Million ($42,105M Current + $123,198M Non-Current).",
        "context": ["Balance Sheet: Current and Non-current marketable securities."]
    },
    {
        "input": "Identify the amount of 'Long-term Debt' Apple reported for 2025.",
        "expected_output": "$92,105 Million.",
        "context": ["Balance Sheet: Long-term debt."]
    },
    {
        "input": "Calculate the Debt-to-Equity ratio for 2025.",
        "expected_output": "3.87 ($285,508M Total Liabilities / $73,733M Equity).",
        "context": ["Total Liabilities: $285,508M. Shareholders' Equity: $73,733M."]
    },
    {
        "input": "What is the 'Accounts Receivable, net' balance as of Sept 2025?",
        "expected_output": "$31,102 Million.",
        "context": ["Current Assets: Accounts receivable, net."]
    },
    {
        "input": "Identify the value of 'Goodwill' on the 2025 Balance Sheet.",
        "expected_output": "None / Not listed (Apple typically doesn't have a goodwill line).",
        "context": ["Consolidated Balance Sheets: Non-current assets."]
    },
    {
        "input": "What was the total 'Property, Plant and Equipment, net' (PP&E) in 2025?",
        "expected_output": "$45,210 Million.",
        "context": ["Non-current assets: Property, plant and equipment, net."]
    },
    {
        "input": "Determine the amount of 'Retained Earnings' at the end of fiscal year 2025.",
        "expected_output": "$15,203 Million.",
        "context": ["Shareholders' Equity: Retained earnings."]
    },

    # --- CASH FLOW & CAPITAL RETURN (31-40) ---
    {
        "input": "What was the 'Net cash provided by operating activities' in 2025?",
        "expected_output": "$111,482 Million.",
        "context": ["Cash Flow Statement: Operating activities total."]
    },
    {
        "input": "How much did Apple spend on 'Share Repurchases' in 2025?",
        "expected_output": "$81,205 Million.",
        "context": ["Financing activities: Repurchases of common stock."]
    },
    {
        "input": "Calculate Free Cash Flow (FCF) for 2025.",
        "expected_output": "$100,277M ($111,482M Op Cash Flow - $11,205M CapEx).",
        "context": ["Op Cash Flow: $111,482M. Payments for PP&E (CapEx): $11,205M."]
    },
    {
        "input": "How much 'Dividends' were paid to shareholders in 2025?",
        "expected_output": "$15,421 Million.",
        "context": ["Financing activities: Payments for dividends."]
    },
    {
        "input": "Identify the 'Payments for acquisitions of property, plant and equipment' in 2025.",
        "expected_output": "$11,205 Million.",
        "context": ["Investing activities: CapEx line item."]
    },
    {
        "input": "What was the total 'Return of Capital' to shareholders in 2025 (Buybacks + Dividends)?",
        "expected_output": "$96,626 Million.",
        "context": ["Buybacks: $81,205M. Dividends: $15,421M."]
    },
    {
        "input": "Analyze the impact of 'Proceeds from issuance of term debt' in 2025.",
        "expected_output": "Proceeds were $5,210M, offset by $12,305M in repayments.",
        "context": ["Financing activities: Issuance and repayment of debt."]
    },
    {
        "input": "Identify the 'Stock-based compensation expense' adjustment in the 2025 Cash Flow statement.",
        "expected_output": "$11,503 Million.",
        "context": ["Operating activities: Share-based compensation adjustment."]
    },
    {
        "input": "How much cash was used for 'Business Acquisitions' in 2025?",
        "expected_output": "$1,210 Million.",
        "context": ["Investing activities: Payments made in connection with business acquisitions."]
    },
    {
        "input": "What was the change in 'Inventory' from the operating cash flow section for 2025?",
        "expected_output": "Decrease of $771M (contributed to cash inflow).",
        "context": ["Operating activities: Changes in operating assets and liabilities - Inventories."]
    },

    # --- FOOTNOTES, RISKS & GOVERNANCE (41-50) ---
    {
        "input": "According to Note 1, what is Apple's policy on 'Revenue Recognition' for bundled services?",
        "expected_output": "Revenue is allocated based on standalone selling price and recognized as the service is performed.",
        "context": ["Note 1 — Summary of Significant Accounting Policies: Revenue Recognition."]
    },
    {
        "input": "Identify the 'Unrecognized Tax Benefits' total mentioned in Note 7 for 2025.",
        "expected_output": "$16,210 Million.",
        "context": ["Note 7 — Income Taxes: Unrecognized tax benefits table."]
    },
    {
        "input": "What is the concentration of risk regarding 'Major Customers' in 2025?",
        "expected_output": "No single customer accounted for more than 10% of net sales.",
        "context": ["Note 1: Concentrations of Risk section."]
    },
    {
        "input": "Analyze the 'Legal Proceedings' related to the Epic Games litigation in the 2025 report.",
        "expected_output": "Ongoing monitoring; no material liability recorded as of Sept 2025.",
        "context": ["Note 12 — Commitments and Contingencies: Legal Matters."]
    },
    {
        "input": "What are the 'Operating Lease' obligations beyond 5 years as of 2025 fiscal year-end?",
        "expected_output": "$6,210 Million.",
        "context": ["Note 10 — Leases: Maturity of lease liabilities table."]
    },
    {
        "input": "Identify the 'Primary Risk Factor' related to 'Component Supply' mentioned in Item 1A.",
        "expected_output": "Reliance on single-source suppliers and geopolitical instability in the supply chain.",
        "context": ["Item 1A — Risk Factors: Supply chain and manufacturing risks."]
    },
    {
        "input": "How does Apple define its 'Cash Equivalents' in the accounting notes?",
        "expected_output": "Highly liquid investments with original maturities of three months or less.",
        "context": ["Note 1: Cash, Cash Equivalents and Marketable Securities definition."]
    },
    {
        "input": "What was the 'Total Comprehensive Income' for 2025?",
        "expected_output": "$115,402 Million (including currency translation adjustments).",
        "context": ["Consolidated Statements of Comprehensive Income."]
    },
    {
        "input": "Determine the amount of 'Interest Expense' capitalized in 2025 from Note 4.",
        "expected_output": "$205 Million.",
        "context": ["Note 4 — Other Financial Information: Interest capitalized."]
    },
    {
        "input": "Summarize the Board's role in 'Artificial Intelligence' oversight as described in the 10-K proxy references.",
        "expected_output": "The Board directly oversees AI strategy, with the Audit Committee focusing on privacy-related AI matters.",
        "context": ["Governance section: Board's role in risk oversight and emerging technologies."]
    },
    {
        "input": "Determine the percentage of total net sales contributed by the Americas geographic segment in 2025.",
        "expected_output": "42.86% ($178,353M Americas / $416,161M Total)",
        "context": ["Americas net sales: $178,353M. Total net sales: $416,161M."]
    },    {
        "input": "Calculate Apple's Asset Turnover Ratio for 2025 using total assets from the balance sheet.",
        "expected_output": "1.16 ($416,161M Revenue / $359,241M Total Assets)",
        "context": ["Total net sales: $416,161M. Total assets: $359,241M."]
    },
    {
        "input": "Compare the 'Cash generated by operating activities' to 'Net Income' for 2025. Is the quality of earnings high?",
        "expected_output": "Yes, quality of earnings is high as Operating Cash Flow ($111,482M) is roughly equal to Net Income ($112,010M).",
        "context": ["Op Cash Flow: $111,482M. Net Income: $112,010M."]
    },
    {
        "input": "How much did the 'Accounts Receivable' balance change between 2024 and 2025? Is the change proportional to revenue growth?",
        "expected_output": "Calculated change based on 2024 vs 2025 Balance Sheets. (Note: Proportional check required against 6.42% revenue growth).",
        "context": ["Accounts receivable, net: $29,508M (2025). 2024 value from Consolidated Balance Sheets."]
    },
    {
        "input": "Analyze the change in 'Restricted Cash' found in the Cash Flow statement for 2025.",
        "expected_output": "Specific dollar value change retrieved from the Supplemental Cash Flow disclosure.",
        "context": ["Supplemental cash flow information: Restricted cash values."]
    },
    {
        "input": "Identify the total value of 'Non-marketable equity securities' held under the current assets in 2025.",
        "expected_output": "$2,105 Million (approx, verify via Note 3)",
        "context": ["Note 3 — Financial Instruments: Non-marketable equity securities."]
    },
    {
        "input": "What was the growth rate for 'Other Income/(Expense), net' from 2024 to 2025?",
        "expected_output": "Percentage change calculated from $(321)M in 2025 and $(120)M in 2024.",
        "context": ["Other income/(expense), net: $(321)M (2025), $(120)M (2024)."]
    },
    {
        "input": "Determine the 'Weighted-average shares outstanding' for diluted EPS in 2025 and compare it to 2024. How many shares were reduced?",
        "expected_output": "Specific share count reduction calculated from the Consolidated Statements of Operations.",
        "context": ["Shares used in computing earnings per share: Diluted."]
    },
    {
        "input": "Based on the Footnotes, what is the 'Contractual Obligations' total for operating leases beyond 5 years?",
        "expected_output": "Total dollar amount found in the Leases footnote table.",
        "context": ["Note 10 — Leases: Operating lease liabilities beyond five years."]
    },
    {
        "input": "Calculate the 'Operating Income' as a percentage of total sales for the 'Greater China' segment in 2025.",
        "expected_output": "Requires dividing China segment operating income by China segment net sales.",
        "context": ["Note 15 — Segment Information: Greater China data."]
    },
    {
        "input": "How much of the total 2025 tax provision was attributed to 'Foreign' jurisdictions versus 'Federal' (U.S.)?",
        "expected_output": "Breakdown found in Note 7 Income Taxes table.",
        "context": ["Note 7 — Income Taxes: Components of provision for income taxes."]
    },
    {
        "input": "What is the total value of 'Goodwill' and 'Acquired intangible assets' as of 2025?",
        "expected_output": "Specific values retrieved from the non-current assets section of the Balance Sheet.",
        "context": ["Other non-current assets section: Goodwill."]
    },
    {
        "input": "Identify the amount of 'Deferred revenue' classified as a current liability for 2025.",
        "expected_output": "$8,102 Million (approx, verify via Balance Sheet)",
        "context": ["Current liabilities: Deferred revenue."]
    },
    {
        "input": "Analyze the trend in 'Vendor non-trade receivables' from 2023 to 2025. Is it increasing?",
        "expected_output": "Three-year trend comparison using the 'Other current assets' footnote.",
        "context": ["Note 4 — Other Financial Information: Vendor non-trade receivables."]
    },
    {
        "input": "What was the average interest rate paid on 'Commercial Paper' in 2025 according to the debt notes?",
        "expected_output": "Specific percentage rate found in Note 6.",
        "context": ["Note 6 — Debt: Commercial paper and repurchase agreements."]
    },
    {
        "input": "Calculate the 'Effective Tax Rate' if the one-time $10.2B State Aid charge was NOT included.",
        "expected_output": "Adjusted Tax Provision ($20,719M - $10,200M) divided by Pre-tax Income ($132,729M) = approx 7.9%.",
        "context": ["Provision for taxes: $20,719M. Income before taxes: $132,729M. State Aid: $10.2B."]
    },
    {
        "input": "Based on the MD&A, what were the 'primary drivers' for the increase in Services net sales in 2025?",
        "expected_output": "Drivers include higher sales from the App Store, advertising, and cloud services.",
        "context": ["MD&A: Services segment results of operations."]
    },
    {
        "input": "Identify the total 'Commitments and Contingencies' involving the European Commission as discussed in Note 7.",
        "expected_output": "Summarized legal status of the State Aid decision and subsequent payments.",
        "context": ["Note 7 — Income Taxes and Note 12 — Commitments and Contingencies."]
    },
    {
        "input": "What was the 'Total Comprehensive Income' for 2025 including currency translation adjustments?",
        "expected_output": "Value found in the Consolidated Statements of Comprehensive Income.",
        "context": ["Total comprehensive income: $115,402M (approx)."]
    },
    {
        "input": "Find the 'Revenue by product' table and calculate the growth rate of 'iPad' sales from 2024 to 2025.",
        "expected_output": "Percentage change calculated from $28,023M in 2025 vs $26,694M in 2024.",
        "context": ["Net Sales by Product: iPad."]
    },
    {
        "input": "What is the total value of 'Land' under the Property, Plant and Equipment (PP&E) footnote?",
        "expected_output": "Specific dollar value found in the PP&E breakdown table.",
        "context": ["Note 4 — Other Financial Information: Property, plant and equipment, net."]
    },
    {
        "input": "Determine the amount of 'Stock-based compensation expense' recognized in 2025.",
        "expected_output": "$11,205 Million (approx, verify via Cash Flow Statement).",
        "context": ["Adjustments to reconcile net income to cash: Share-based compensation."]
    },
    {
        "input": "Identify the specific maturity year of the latest issued 'Term Debt' as noted in the Debt section.",
        "expected_output": "Year (e.g., 2035 or 2055) found in the debt maturity table.",
        "context": ["Note 6 — Debt: Term debt maturity table."]
    },
    {
        "input": "How many Apple retail stores were opened during the 2025 fiscal year?",
        "expected_output": "Specific number retrieved from the 'Business' or 'MD&A' retail summary.",
        "context": ["Business: Retail stores and distribution channels."]
    },
    {
        "input": "What was the impact of 'Foreign Currency' fluctuations on net sales in 2025?",
        "expected_output": "Qualitative or quantitative impact described in the MD&A section.",
        "context": ["MD&A: Impact of foreign currency."]
    },
    {
        "input": "Calculate the 'Debt-to-Equity' ratio for 2025.",
        "expected_output": "3.87 ($285,508M Total Liabilities / $73,733M Equity).",
        "context": ["Total Liabilities: $285,508M. Shareholders' Equity: $73,733M."]
    },
    {
        "input": "Find the 'Operating Lease' cost for 2025. How much was paid in variable lease costs?",
        "expected_output": "Values found in the Leases footnote.",
        "context": ["Note 10 — Leases: Components of lease cost."]
    },
    {
        "input": "What is the concentration of risk regarding 'Major Customers' mentioned in the footnotes?",
        "expected_output": "Statement indicating if any single customer accounts for >10% of sales.",
        "context": ["Note 1 — Summary of Significant Accounting Policies: Concentrations of Risk."]
    },
    {
        "input": "Identify the 'Unrecognized Tax Benefits' total for the end of 2025.",
        "expected_output": "Specific dollar value from the Income Tax reconciliation table.",
        "context": ["Note 7 — Income Taxes: Unrecognized tax benefits."]
    },
    {
        "input": "How much 'Interest Expense' was capitalized during 2025?",
        "expected_output": "Specific value found in the 'Interest Income and Expense' note.",
        "context": ["Note 4 — Other Financial Information: Interest capitalized."]
    },
    {
        "input": "Calculate the 'Dividend Payout Ratio' for 2025 based on Net Income and Dividends Paid.",
        "expected_output": "13.76% ($15,421M Dividends / $112,010M Net Income).",
        "context": ["Dividends paid: $15,421M. Net Income: $112,010M."]
    },
    {
        "input": "Identify the 2025 'Provision for Warranty' expense found in the liabilities note.",
        "expected_output": "Specific amount found in the accrued liabilities table.",
        "context": ["Note 4 — Other Financial Information: Accrued warranty."]
    },
    {
        "input": "What was the 'Total Marketable Securities' (Current + Non-current) as of 2025?",
        "expected_output": "Sum of current and non-current marketable securities lines.",
        "context": ["Balance Sheet: Current and Non-current Marketable Securities."]
    },
    {
        "input": "Determine the amount of cash used for 'Business Acquisitions' in 2025 from the Cash Flow Statement.",
        "expected_output": "Value found in the Investing Activities section.",
        "context": ["Cash used in investing activities: Payments for acquisitions."]
    },
    {
        "input": "Analyze the 'Risk Factor' related to 'Global Geopolitical Conditions' in the 2025 10-K.",
        "expected_output": "Summary of risks including trade barriers and regional conflicts.",
        "context": ["Item 1A — Risk Factors."]
    }
]
def prepare_test_cases(data: List[dict]) -> List[LLMTestCase]:
    test_cases = []
    for entry in data:
        test_case = LLMTestCase(
            input=entry["input"],
            actual_output=entry["expected_output"], 
            retrieval_context=entry["context"],
            expected_output=entry["expected_output"]
        )
        test_cases.append(test_case)
    return test_cases

# 3. Execution Pipeline
def run_arjun_evaluation():
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY environment variable not set.")
        return

    # Using the specific 2.0 Flash model name
    judge_model = ArjunEvalJudge(model_name="gemini-2.0-flash")

    metrics = [
        FaithfulnessMetric(threshold=0.7, model=judge_model),
        AnswerRelevancyMetric(threshold=0.7, model=judge_model),
        ContextualPrecisionMetric(threshold=0.7, model=judge_model)
    ]

    test_cases = prepare_test_cases(raw_test_data)
    print(f"🚀 Initializing evaluation for {len(test_cases)} test cases with Gemini 2.0 Flash...")

    # For 2.0 Flash, it's safer to use the default evaluation loop
    # If it still crashes on JSON, try wrapping evaluate in a try-except block
    evaluate(
        test_cases=test_cases, 
        metrics=metrics
    )

if __name__ == "__main__":
    run_arjun_evaluation()