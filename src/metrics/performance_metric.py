import re
import os
import json
from google import genai
from src.models.types import EvalContext
import logging


# Configure Gemini API key from environment variable
api_key = os.getenv("GOOGLE_API_KEY")
# if not api_key:
#     raise RuntimeError("Could not find GOOGLE_API_KEY. Make sure to set it as an environment variable.")
# genai.configure(api_key=api_key)


async def metric(ctx: EvalContext) -> float:
    gh = ctx.gh_data[0]
    """
    Analyze the README content from EvalContext for performance claims
    and evidence quality using Gemini.
    Returns a float score (average of overall_evidence_quality and overall_specificity).
    """

    readme_content = gh.get("readme_text")

    prompt = f"""
    Analyze the following README content for performance claims and evidence quality. 

    Look for:
    1. Quantitative benchmark results (specific numbers, scores, metrics)
    2. Comparisons with other models (with data)
    3. Dataset evaluations and test results
    4. Performance metrics (accuracy, F1, BLEU, etc.)
    5. Subjective claims without evidence ("best in class", "state-of-the-art" without data)

    Rate each type of claim on evidence quality and specificity.

    README Content:
    ---
    {readme_content}
    ---

    Please provide a structured analysis in the following JSON format:
    {{
        "claims_found": [
            {{
                "claim_text": "extracted claim text",
                "claim_type": "benchmark|metric|comparison|subjective",
                "evidence_quality": 0.0-1.0,
                "specificity": 0.0-1.0,
                "datasets_mentioned": ["dataset1", "dataset2"],
                "metrics_mentioned": ["metric1", "metric2"]
            }}
        ],
        "summary": {{
            "total_claims": number,
            "quantitative_claims": number,
            "benchmark_count": number,
            "has_tables_or_charts": boolean,
            "overall_evidence_quality": 0.0-1.0,
            "overall_specificity": 0.0-1.0
        }}
    }}

    Focus on being precise about what constitutes good evidence vs. vague claims.
    """

    client = genai.Client()
    logging.info("Calling Gemini with prompt for performance metric: %s", prompt[:500].replace("\n", " ") + "...")

    # Call Gemini (this is synchronous)
    response = client.models.generate_content(
        model = "gemini-2.0-flash", 
        contents = prompt
        )

    # Extract text response
    analysis_text = response.text
    # logging.info("Gemini response: %s", analysis_text)

    # Try parsing into JSON
    try:
        raw = response.text
        cleaned = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.DOTALL)
        analysis_json = json.loads(cleaned)
    except json.JSONDecodeError:
        raise RuntimeError(f"Gemini response was not valid JSON:\n{analysis_text}")

    # Compute score from summary
    summary = analysis_json.get("summary", {})
    quality = summary.get("overall_evidence_quality", 0.0)
    specificity = summary.get("overall_specificity", 0.0)
    print(f"Performance Metric - Quality: {quality}, Specificity: {specificity}")
    return float((quality + specificity) / 2.0)