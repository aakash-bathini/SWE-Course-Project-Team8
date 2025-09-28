import re
import os
import json
import logging
import requests
from src.models.types import EvalContext

MAX_INPUT_CHARS = 7000
api_key = os.getenv("GEMINI_API_KEY")
purdue_api_key = os.getenv("GEN_AI_STUDIO_API_KEY")

async def metric(ctx: EvalContext) -> float:
    if not api_key and not purdue_api_key:
        logging.error("GOOGLE_API_KEY and GEN_AI_STUDIO_API_KEY not set, performance_metric will fail")
        return 0.0

    readme_content = ""
    try:
        if ctx.hf_data and isinstance(ctx.hf_data, list) and ctx.hf_data:
            hf = ctx.hf_data[0] or {}
            readme_content = hf.get("readme_text") or ""
            logging.info("Performance metric using Hugging Face README")
        elif ctx.gh_data and isinstance(ctx.gh_data, list) and ctx.gh_data:
            gh = ctx.gh_data[0] or {}
            readme_content = gh.get("readme_text") or ""
            logging.info("Performance metric using GitHub README")
        else:
            logging.info("No README available (HF or GitHub), skipping performance metric")
            return 0.0
    except Exception as e:
        logging.error("Performance metric: error selecting README source: %s", e)
        return 0.0

    readme_content = readme_content[:MAX_INPUT_CHARS]

    prompt = f"""
    You are analyzing the README of an ML model or dataset to detect performance-related claims.
    Your job is to extract only claims about performance, benchmarks, or datasets, and score their evidence.

    Step 1: Identify all performance-related claims. Examples include:
    - Quantitative results (accuracy, F1, BLEU, latency, throughput, etc.)
    - Comparisons to other models ("better than BERT", "state-of-the-art", etc.)
    - Dataset usage or evaluation ("evaluated on SQuAD", "trained on 1M examples")

    Step 2: For each claim, record:
    - claim_text: the raw claim from the README
    - claim_type: one of ["metric", "benchmark", "comparison", "subjective"]
    - evidence_quality: score in [0.0, 1.0] → how well-supported is the claim (numbers, citations, datasets)?
    - specificity: score in [0.0, 1.0] → how precise is the claim (exact metrics vs vague words)?
    - datasets_mentioned: list of dataset names if any
    - metrics_mentioned: list of metric names if any

    Step 3: Provide a summary object:
    - total_claims: number of claims found
    - overall_evidence_quality: average of evidence_quality values
    - overall_specificity: average of specificity values

    Return ONLY valid JSON (no extra commentary, no markdown). 
    Use this schema:

    {{
    "claims_found": [
        {{
        "claim_text": "...",
        "claim_type": "...",
        "evidence_quality": 0.0,
        "specificity": 0.0,
        "datasets_mentioned": [],
        "metrics_mentioned": []
        }}
    ],
    "summary": {{
        "total_claims": 0,
        "overall_evidence_quality": 0.0,
        "overall_specificity": 0.0
    }}
    }}

    README Content:
    ---
    {readme_content}
    ---
    """

    analysis_json = None
    for attempt in range(1, 4):  # up to 3 attempts
        try:
            if api_key:
                from google import genai
                client = genai.Client()
                logging.info("Performance metric attempt %d with Gemini", attempt)
                response = client.models.generate_content(
                    model="gemini-2.0-flash", 
                    contents=prompt
                )
                raw = response.text
            else:
                url = "https://genai.rcac.purdue.edu/api/chat/completions"
                headers = {
                    "Authorization": f"Bearer {purdue_api_key}",
                    "Content-Type": "application/json"
                }
                body = {
                    "model": "llama4:latest",
                    "messages": [
                        {"role": "system", "content": "You are a very needed engineer analyzing README files for performance claims."},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "max_tokens": 1024
                }
                logging.info("Performance metric attempt %d with Purdue GenAI", attempt)
                response = requests.post(url, headers=headers, json=body)
                raw = response.json()['choices'][0]['message']['content']

            # clean + parse
            cleaned = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.DOTALL)
            analysis_json = json.loads(cleaned)
            logging.info("Performance metric JSON parse succeeded on attempt %d", attempt)
            break  # ✅ success, stop retrying

        except Exception as e:
            logging.debug("Performance metric attempt %d failed: %s", attempt, e)
            continue  # try again

    if not analysis_json:
        logging.error("Performance metric failed all 3 attempts, returning 0.0")
        return 0.0

    # Compute score from summary
    summary = analysis_json.get("summary", {})
    quality = summary.get("overall_evidence_quality", 0.0)
    specificity = summary.get("overall_specificity", 0.0)
    logging.info(
        "Performance Metric -> Quality: %s, Specificity: %s, Total Claims: %s",
        quality, specificity, summary.get("total_claims", 0)
    )
    return float(round(((quality + specificity) / 2.0), 2))