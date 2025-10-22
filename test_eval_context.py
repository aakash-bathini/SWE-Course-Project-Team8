# test_prep_eval_context.py
import os
from src.api.prep_eval_context import prepare_eval_context

# Optional: isolate cache so you don’t pollute your main project cache
os.environ["PROJECT_ROOT"] = os.getcwd()

ctx = prepare_eval_context("https://github.com/kabirraymalik/gello-bimanual-ur3")
# ctx = prepare_eval_context("https://huggingface.co/facebook/MobileLLM-R1-950M")

print("URL:", ctx.url)
print("Category:", ctx.category)
print("HF Data count:", len(ctx.hf_data or []))
print("GH Data count:", len(ctx.gh_data or []))
if ctx.gh_data:
    gh = ctx.gh_data[0]
    print("Repo ID:", gh.get("repo_id"))
    print("Stars:", gh.get("stars"))
    print("Forks:", gh.get("forks"))
    print("Open Issues:", gh.get("open_issues"))
    print("Readme excerpt:", (gh.get("readme_text") or "")[:200].replace("\n", " "))
