# fmt: off
import pytest
import sys
import os
from unittest.mock import patch, AsyncMock


@pytest.mark.parametrize("model_name", ["org/good-model"])
def test_good_model_ingest_and_rate_passes_threshold(model_name):
    """
    Integration test:
    - Mock HF and GitHub scrapers to return rich metadata for a 'good' model
    - POST /models/ingest should succeed (>=0.5 on all applicable non-latency metrics)
    - GET /artifact/model/{id}/rate should return a high net score and healthy submetrics
    """
    # Mock HF data with high engagement and clear README/metadata
    hf_data = {
        "url": f"https://huggingface.co/{model_name}",
        "repo_id": model_name,
        "repo_type": "model",
        "license": "mit",
        "downloads": 2_500_000,
        "likes": 5000,
        "pipeline_tag": "text-classification",
        "readme_text": """
            # Good Model
            Achieves 92% accuracy on GLUE and 90% on MNLI.
            Evaluated on SQuAD and CoLA with strong F1 and precision/recall.
            | Benchmark | Score |
            | GLUE      | 0.92  |
            | MNLI      | 0.90  |

            ## Usage

            ```python
            from transformers import AutoModel, AutoTokenizer
            model = AutoModel.from_pretrained("org/good-model")
            tokenizer = AutoTokenizer.from_pretrained("org/good-model")
            ```
            """,
        "github_links": ["https://github.com/example/good-model"],
        "files": [{"path": "weights.bin", "size": 1024}],
        "datasets": ["glue", "squad"],
        "tags": ["benchmark:glue", "dataset:squad", "task:classification"],
    }

    # Mock GitHub profile with balanced contributors and license info
    gh_profile = {
        "repo": "example/good-model",
        "license_spdx": "MIT",
        "contributors": {"alice": 500, "bob": 450, "carol": 400},
        "readme_text": "This repo provides the Good Model. MIT License.",
        "doc_texts": {"LICENSE": "MIT License", "CONTRIBUTING": "Please contribute."},
    }

    # Ensure project root on sys.path for `import app`
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    )
    from fastapi.testclient import TestClient
    from app import app
    from src.auth.jwt_auth import auth

    # Valid admin+upload token
    token = auth.create_access_token(
        {
            "sub": "tester",
            "permissions": ["upload", "search", "download", "admin"],
        }
    )
    headers = {"X-Authorization": f"bearer {token}"}

    # Mock reproducibility metric to return 1.0 (perfect execution) to pass threshold
    with patch("app.scrape_hf_url", return_value=(hf_data, "model")), patch(
        "app.scrape_github_url", return_value=gh_profile
    ), patch("src.metrics.reproducibility.metric", new=AsyncMock(return_value=1.0)):
        # Ingest the model
        client = TestClient(app)
        resp = client.post(f"/models/ingest?model_name={model_name}", headers=headers)
        assert resp.status_code in (200, 201), resp.text
        data = resp.json()
        art_id = data["metadata"]["id"]
        assert data["metadata"]["type"] == "model"
        assert art_id

        # Rate the model
        rate = client.get(f"/artifact/model/{art_id}/rate", headers=headers)
        assert rate.status_code == 200, rate.text
        rating = rate.json()

        # Net score should be comfortably above threshold due to high-quality metadata
        assert rating["net_score"] >= 0.5

        # Spot-check a few sub-metrics likely to be >= 0.5 with our mocked data
        for key in ["license", "bus_factor", "code_quality", "performance_claims"]:
            assert key in rating
            # Only check non-negative metrics; sentinel negatives are allowed for N/A
            # reviewedness can be -1.0 (sentinel) if no GitHub repo
            if isinstance(rating[key], (int, float)):
                if key == "reviewedness" and rating[key] == -1.0:
                    continue
                if rating[key] >= 0.0:
                    assert rating[key] >= 0.5
# fmt: on
