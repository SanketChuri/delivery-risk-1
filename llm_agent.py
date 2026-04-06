import os
import json
from groq import Groq
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()


def _get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Set it in your environment or .env file before running."
        )
    return Groq(api_key=api_key)


def build_fallback_brief(row, raw_content=""):
    job_id = row.get("job_id", "Unknown")
    delay = row.get("delay", 0)
    priority = str(row.get("priority", "")).lower()
    traffic = str(row.get("traffic_level", "")).lower()
    status = str(row.get("status", "")).lower()
    risk_level = row.get("risk_level", "Medium")
    ops_action = row.get("ops_action", "Review job manually.")

    reasons = []

    if delay and delay > 0:
        reasons.append(f"it is already {delay} minutes behind schedule")
    if priority == "high":
        reasons.append("it is a high-priority delivery")
    if traffic == "heavy":
        reasons.append("traffic conditions are heavy")
    if status in ["delayed", "in_transit", "picked_up", "on_route"]:
        reasons.append(f"the current status is {status.replace('_', ' ')}")

    if reasons:
        explanation = f"Job {job_id} is {risk_level.lower()} risk because " + ", ".join(reasons) + "."
    else:
        explanation = f"Job {job_id} is being monitored due to potential delivery risk."

    if risk_level == "High":
        customer_message = (
            "Your delivery is currently experiencing a delay. "
            "Our operations team is actively reviewing it and working to keep the delivery on track."
        )
    elif risk_level == "Medium":
        customer_message = (
            "Your delivery may be slightly delayed due to live operating conditions. "
            "We are monitoring it closely."
        )
    else:
        customer_message = (
            "Your delivery is on track, and our team is continuing to monitor it in real time."
        )

    return {
        "risk_explanation": raw_content if raw_content else explanation,
        "ops_recommendation": ops_action,
        "customer_message": customer_message,
    }


@traceable(name="generate_delivery_ai_brief")
def generate_ai_brief(row):
    client = _get_groq_client()

    prompt = f"""
You are an operations copilot for a same-day courier company.

Given this delivery job, return a JSON object with exactly these keys:
- risk_explanation
- ops_recommendation
- customer_message

Rules:
- Keep each value short, practical, and clear
- Make the answer specific to the job data
- customer_message must be professional and customer-friendly
- Return valid JSON only
- Do not include markdown fences

Job Data:
Job ID: {row.get('job_id', '')}
Driver ID: {row.get('driver_id', '')}
Delay: {row.get('delay', '')}
Risk Score: {row.get('risk_score', '')}
Risk Level: {row.get('risk_level', '')}
Alert Level: {row.get('alert_level', '')}
Priority: {row.get('priority', '')}
Traffic Level: {row.get('traffic_level', '')}
Status: {row.get('status', '')}
Recommended Action: {row.get('recommended_action', '')}
Ops Action: {row.get('ops_action', '')}
ETA Drift: {row.get('eta_drift', '')}
Expected Delivery Time: {row.get('expected_delivery_time', '')} minutes
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except Exception:
        return build_fallback_brief(row, raw_content=content)