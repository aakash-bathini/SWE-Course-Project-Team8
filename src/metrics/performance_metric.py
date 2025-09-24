import re
import os
import json
try:
    import google.generativeai as genai
except ImportError:
    logging.warning("google.generativeai package not found, Gemini calls will fail")
from src.models.types import EvalContext
import logging
import requests

# Configure Gemini API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")
purdue_api_key = os.getenv("GEN_AI_STUDIO_API_KEY")
    
# genai.configure(api_key=api_key)


async def metric(ctx: EvalContext) -> float:
    if not api_key and not purdue_api_key:
        logging.error("GOOGLE_API_KEY and GEN_AI_STUDIO_API_KEY environment variables not set, performance_metric will fail")
        return 0.0
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
    Only  respond with the JSON object, no additional text.
    """
    
    if api_key:
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
        try:
            raw = response.text
            cleaned = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.DOTALL)
            analysis_json = json.loads(cleaned)
        except json.JSONDecodeError:
            logging.error(f"LLM response was not valid JSON:\n{analysis_text}")
            return (0.0)
    else:
        logging.info("Calling Purdue GenAI with prompt for performance metric: %s", prompt[:500].replace("\n", " ") + "...")

        url = "https://genai.rcac.purdue.edu/api/chat/completions"
        headers = {
            "Authorization": f"Bearer {purdue_api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": "llama4:latest",
            "messages": [
            {
                "role": "a very needed Engineer looking at README files for performance claims",
                "content": prompt
            }
            ],
            "stream": False
    }
        response = requests.post(url, headers=headers, json=body)
        analysis_text = response.json()['choices'][0]['message']['content']
        # Try parsing into JSON
        try:
            cleaned = re.sub(r"^```json\s*|\s*```$", "", analysis_text.strip(), flags=re.DOTALL)
            analysis_json = json.loads(cleaned)
        except json.JSONDecodeError:
            logging.error(f"LLM response was not valid JSON:\n{analysis_text}")
            return (0.0)

    # logging.info("LLM response: %s", analysis_text)
    # Compute score from summary
    summary = analysis_json.get("summary", {})
    quality = summary.get("overall_evidence_quality", 0.0)
    specificity = summary.get("overall_specificity", 0.0)
    logging.info(
        f"Performance Metric -> Quality: {quality}, Specificity: {specificity}, "
        f"Total Claims: {summary.get('total_claims', 0)}, Quantitative Claims: {summary.get('quantitative_claims', 0)}, "
        f"Benchmark Count: {summary.get('benchmark_count', 0)}, Has Tables/Charts: {summary.get('has_tables_or_charts', False)}"
    )
    # print(f"Performance Metric - Quality: {quality}, Specificity: {specificity}")
    return float((quality + specificity) / 2.0)