import re
import os
import json
import logging
import requests
from src.models.types import EvalContext

MAX_INPUT_CHARS = 7750
api_key = os.getenv("GEMINI_API_KEY")
purdue_api_key = os.getenv("GEN_AI_STUDIO_API_KEY")

async def metric(ctx: EvalContext) -> float:
    if not api_key and not purdue_api_key:
        logging.error("GOOGLE_API_KEY and GEN_AI_STUDIO_API_KEY not set, performance_metric will fail")
        return 0.0

    try:
        gh = ctx.gh_data[0]
    except (IndexError, KeyError):
        logging.info("No GitHub data in EvalContext, skipping performance metric")
        return 0.0

    readme_content = (gh.get("readme_text") or "")[:MAX_INPUT_CHARS]

    prompt = f"""
    Analyze the following README content for performance claims and evidence quality. 
    ...
    README Content:
    ---
    {readme_content}
    ---
    Please provide a structured analysis in JSON format only.
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
            logging.warning("Performance metric attempt %d failed: %s", attempt, e)
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