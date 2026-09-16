import os
import json
import time
import pandas as pd
from google import genai

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set")

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-3.6-flash"

df = pd.read_csv("judge_calibration_30.csv")

results = []

RUBRIC = """
Score the customer-support reply independently.

Relevance:
1 = unrelated
2 = mostly unrelated
3 = partially relevant
4 = relevant
5 = directly addresses the customer's issue

Helpfulness:
1 = useless
2 = minimally useful
3 = somewhat useful
4 = useful
5 = highly useful / gives an appropriate next step

Grounding:
1 = unsupported by the supplied historical evidence
2 = weakly grounded
3 = partially grounded
4 = well grounded
5 = directly supported by the historical evidence

Non-hallucination:
1 = contains serious unsupported claims
2 = contains notable unsupported claims
3 = minor uncertainty or unsupported detail
4 = essentially safe
5 = fully supported / no invented facts

Tone:
1 = inappropriate
2 = poor
3 = acceptable
4 = professional
5 = professional and empathetic

Overall acceptable:
YES if the reply is reasonable to send to the customer.
NO if it is materially misleading, irrelevant, unsafe, or inadequate.

Escalation:
Judge whether the system's escalation decision is appropriate for the customer's issue.
Return YES, NO, or UNCLEAR.
"""

for i, row in df.iterrows():

    prompt = f"""
You are evaluating an AI customer-support agent.

{RUBRIC}

IMPORTANT:
- Do not infer or use any hidden gold label.
- Evaluate only the information provided below.
- The historical reply is evidence, not automatically correct.
- A generic escalation fallback may be appropriate for sensitive/account-specific issues.
- Return ONLY valid JSON.

CUSTOMER MESSAGE:
{row['customer_text']}

SYSTEM PREDICTED INTENT:
{row['predicted_intent']}

SYSTEM ESCALATION DECISION:
{row['should_escalate']}

SYSTEM ESCALATION REASON:
{row['escalation_reason']}

RETRIEVED HISTORICAL CUSTOMER MESSAGE:
{row['selected_evidence_customer']}

RETRIEVED HISTORICAL SUPPORT REPLY:
{row['selected_evidence_reply']}

GENERATED SYSTEM REPLY:
{row['reply']}

Return exactly this JSON structure:

{{
  "relevance": 1,
  "helpfulness": 1,
  "grounding": 1,
  "non_hallucination": 1,
  "tone": 1,
  "overall_acceptable": "YES",
  "escalation_appropriate": "UNCLEAR",
  "reason": "brief explanation"
}}
"""

    print(f"Judging {i + 1}/30: {row['case_id']}")

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json"
            }
        )

        text = response.text.strip()
        result = json.loads(text)

        results.append({
            "case_id": row["case_id"],
            **result
        })

    except Exception as e:
        print("ERROR:", e)

        results.append({
            "case_id": row["case_id"],
            "relevance": None,
            "helpfulness": None,
            "grounding": None,
            "non_hallucination": None,
            "tone": None,
            "overall_acceptable": None,
            "escalation_appropriate": None,
            "reason": str(e)
        })

    time.sleep(1)

out = pd.DataFrame(results)

out.to_csv("llm_judge_30_results.csv", index=False)
out.to_excel("llm_judge_30_results.xlsx", index=False)

print()
print("DONE")
print(f"Results saved: {len(out)}")
print("Files:")
print("  llm_judge_30_results.csv")
print("  llm_judge_30_results.xlsx")