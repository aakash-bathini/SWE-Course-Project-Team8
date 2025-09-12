#src/api/prep_eval_context.py
from src.models.types import EvalContext, Category

def prepare_eval_context(url: str | None = None) -> EvalContext:
    """
    Dummy context builder for testing.
    Returns a simple EvalContext with filler data so metrics can run.
    """
    url = url or "https://example.com/model/demo"
    cat: Category | None = "MODEL"

    # Filler data you can expand later and call API's
    hf_data = {"card": {"license": "apache-2.0"}, "downloads": 1234}
    gh_data = None
    repo_dir = None

    return EvalContext(url=url, category=cat, hf_data=hf_data, gh_data=gh_data, repo_dir=repo_dir)