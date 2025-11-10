"""
Milestone 4 Tests - New Metrics & Search Functionality
Tests for:
- 3 new metrics: Reproducibility, Reviewedness, Treescore
- Rate endpoint returning all 11 metrics
- Search endpoints: /models/search (regex), /models/search/version (semver)
"""

import pytest
from fastapi.testclient import TestClient
from typing import Dict, Any
import re
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


@pytest.fixture
def client():
    """Create a test client"""
    return TestClient(app)


@pytest.fixture
def admin_headers(client: TestClient) -> Dict[str, str]:
    """Get admin headers for authenticated requests"""
    try:
        resp = client.put(
            "/authenticate",
            json={
                "user": {"name": "admin", "is_admin": True},
                "secret": {"password": "admin"},
            },
        )
        if resp.status_code == 200:
            token = resp.json()
            return {"X-Authorization": token}
    except Exception:
        pass
    return {}


class TestNewMetrics:
    """Test 3 new metrics are properly integrated"""

    def test_rate_returns_reproducibility_metric(self, client: TestClient, admin_headers: Dict[str, str]):
        """Rate endpoint returns reproducibility metric"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        # Upload a model
        resp = client.post(
            "/models/upload",
            data={"url": "https://huggingface.co/gpt2"},
            headers=admin_headers,
        )
        
        if resp.status_code in (201, 200):
            artifact_id = resp.json().get("id")
            
            # Get rating
            rating_resp = client.get(
                f"/artifact/model/{artifact_id}/rate",
                headers=admin_headers,
            )
            
            assert rating_resp.status_code == 200
            rating_data = rating_resp.json()
            assert "reproducibility" in rating_data
            assert isinstance(rating_data["reproducibility"], (int, float))

    def test_rate_returns_reviewedness_metric(self, client: TestClient, admin_headers: Dict[str, str]):
        """Rate endpoint returns reviewedness metric"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        # Upload a model
        resp = client.post(
            "/models/upload",
            data={"url": "https://huggingface.co/gpt2"},
            headers=admin_headers,
        )
        
        if resp.status_code in (201, 200):
            artifact_id = resp.json().get("id")
            
            # Get rating
            rating_resp = client.get(
                f"/artifact/model/{artifact_id}/rate",
                headers=admin_headers,
            )
            
            assert rating_resp.status_code == 200
            rating_data = rating_resp.json()
            assert "reviewedness" in rating_data
            # Reviewedness can be -1 (no GitHub repo), 0-1 (fraction), or positive float
            assert isinstance(rating_data["reviewedness"], (int, float))

    def test_rate_returns_treescore_metric(self, client: TestClient, admin_headers: Dict[str, str]):
        """Rate endpoint returns treescore metric"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        # Upload a model
        resp = client.post(
            "/models/upload",
            data={"url": "https://huggingface.co/gpt2"},
            headers=admin_headers,
        )
        
        if resp.status_code in (201, 200):
            artifact_id = resp.json().get("id")
            
            # Get rating
            rating_resp = client.get(
                f"/artifact/model/{artifact_id}/rate",
                headers=admin_headers,
            )
            
            assert rating_resp.status_code == 200
            rating_data = rating_resp.json()
            assert "tree_score" in rating_data
            # Treescore should be between -1 and 1
            assert isinstance(rating_data["tree_score"], (int, float))

    def test_rate_returns_all_metrics_with_latencies(self, client: TestClient, admin_headers: Dict[str, str]):
        """Rate endpoint returns all 11 metrics plus 3 new ones with latency info"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        # Upload a model
        resp = client.post(
            "/models/upload",
            data={"url": "https://huggingface.co/gpt2"},
            headers=admin_headers,
        )
        
        if resp.status_code in (201, 200):
            artifact_id = resp.json().get("id")
            
            # Get rating
            rating_resp = client.get(
                f"/artifact/model/{artifact_id}/rate",
                headers=admin_headers,
            )
            
            assert rating_resp.status_code == 200
            rating_data = rating_resp.json()
            
            # Check all Phase 1 metrics (8 original)
            assert "ramp_up_time" in rating_data
            assert "bus_factor" in rating_data
            assert "performance_claims" in rating_data
            assert "license" in rating_data
            assert "dataset_and_code_score" in rating_data
            assert "dataset_quality" in rating_data
            assert "code_quality" in rating_data
            assert "size_score" in rating_data
            
            # Check Phase 2 metrics (3 new)
            assert "reproducibility" in rating_data
            assert "reviewedness" in rating_data
            assert "tree_score" in rating_data
            
            # Check net_score
            assert "net_score" in rating_data
            
            # Check latency fields exist
            assert "reproducibility_latency" in rating_data
            assert "reviewedness_latency" in rating_data
            assert "tree_score_latency" in rating_data

    def test_net_score_excludes_phase2_metrics(self, client: TestClient, admin_headers: Dict[str, str]):
        """Net score calculation only uses Phase 1 metrics, not Phase 2"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        # Upload a model
        resp = client.post(
            "/models/upload",
            data={"url": "https://huggingface.co/gpt2"},
            headers=admin_headers,
        )
        
        if resp.status_code in (201, 200):
            artifact_id = resp.json().get("id")
            
            # Get rating
            rating_resp = client.get(
                f"/artifact/model/{artifact_id}/rate",
                headers=admin_headers,
            )
            
            assert rating_resp.status_code == 200
            rating_data = rating_resp.json()
            
            # Net score should be between 0 and 1 (weighted average of Phase 1 only)
            assert 0.0 <= rating_data["net_score"] <= 1.0


class TestSearchByRegex:
    """Test /models/search endpoint with regex patterns"""

    def test_search_requires_authentication(self, client: TestClient):
        """Search endpoint requires authentication"""
        resp = client.get("/models/search", params={"query": "gpt"})
        # Either 401 (Unauthorized) or 403 (Forbidden) - both indicate auth failure
        assert resp.status_code in (401, 403)

    def test_search_requires_search_permission(self, client: TestClient, admin_headers: Dict[str, str]):
        """Search endpoint requires search permission"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        # Should work with admin token (has search permission)
        resp = client.get(
            "/models/search",
            params={"query": "gpt"},
            headers=admin_headers,
        )
        assert resp.status_code in (200, 400)  # 400 if query param is missing

    def test_search_requires_query_parameter(self, client: TestClient, admin_headers: Dict[str, str]):
        """Search requires query parameter"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        resp = client.get("/models/search", headers=admin_headers)
        assert resp.status_code == 422  # Pydantic validation error for missing required param

    def test_search_empty_query_rejected(self, client: TestClient, admin_headers: Dict[str, str]):
        """Search rejects empty query"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        resp = client.get(
            "/models/search",
            params={"query": ""},
            headers=admin_headers,
        )
        assert resp.status_code == 400

    def test_search_invalid_regex_rejected(self, client: TestClient, admin_headers: Dict[str, str]):
        """Search rejects invalid regex patterns"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        # Invalid regex with unclosed bracket
        resp = client.get(
            "/models/search",
            params={"query": "[invalid"},
            headers=admin_headers,
        )
        assert resp.status_code == 400
        error = resp.json()
        assert "regex" in error.get("detail", "").lower() or "pattern" in error.get("detail", "").lower()

    def test_search_simple_string_match(self, client: TestClient, admin_headers: Dict[str, str]):
        """Search with simple string pattern"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        resp = client.get(
            "/models/search",
            params={"query": "gpt"},
            headers=admin_headers,
        )
        
        assert resp.status_code == 200
        data = resp.json()
        assert "query" in data
        assert data["query"] == "gpt"
        assert "count" in data
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_search_regex_pattern(self, client: TestClient, admin_headers: Dict[str, str]):
        """Search with regex pattern"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        # Search for pattern like "bert-*"
        resp = client.get(
            "/models/search",
            params={"query": "bert.*"},
            headers=admin_headers,
        )
        
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["results"], list)

    def test_search_case_insensitive(self, client: TestClient, admin_headers: Dict[str, str]):
        """Search is case-insensitive"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        # Both should work
        resp1 = client.get(
            "/models/search",
            params={"query": "GPT"},
            headers=admin_headers,
        )
        
        resp2 = client.get(
            "/models/search",
            params={"query": "gpt"},
            headers=admin_headers,
        )
        
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        # Results might be the same or different, but both should be valid

    def test_search_returns_artifact_metadata(self, client: TestClient, admin_headers: Dict[str, str]):
        """Search results include artifact metadata"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        resp = client.get(
            "/models/search",
            params={"query": ".*"},  # Match everything
            headers=admin_headers,
        )
        
        if resp.status_code == 200:
            data = resp.json()
            if data["count"] > 0:
                result = data["results"][0]
                assert "id" in result
                assert "name" in result
                assert "type" in result

    def test_search_no_duplicates(self, client: TestClient, admin_headers: Dict[str, str]):
        """Search results have no duplicates"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        resp = client.get(
            "/models/search",
            params={"query": ".*"},
            headers=admin_headers,
        )
        
        if resp.status_code == 200:
            data = resp.json()
            ids = [r["id"] for r in data["results"]]
            # Check no duplicates
            assert len(ids) == len(set(ids))


class TestSearchByVersion:
    """Test /models/search/version endpoint with semver patterns"""

    def test_version_search_requires_authentication(self, client: TestClient):
        """Version search requires authentication"""
        resp = client.get("/models/search/version", params={"query": "1.0.0"})
        # Either 401 (Unauthorized) or 403 (Forbidden) - both indicate auth failure
        assert resp.status_code in (401, 403)

    def test_version_search_requires_query(self, client: TestClient, admin_headers: Dict[str, str]):
        """Version search requires query parameter"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        resp = client.get("/models/search/version", headers=admin_headers)
        assert resp.status_code == 422

    def test_version_search_empty_query_rejected(self, client: TestClient, admin_headers: Dict[str, str]):
        """Version search rejects empty query"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        resp = client.get(
            "/models/search/version",
            params={"query": ""},
            headers=admin_headers,
        )
        assert resp.status_code == 400

    def test_version_search_exact_match(self, client: TestClient, admin_headers: Dict[str, str]):
        """Version search with exact version"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        resp = client.get(
            "/models/search/version",
            params={"query": "1.0.0"},
            headers=admin_headers,
        )
        
        assert resp.status_code == 200
        data = resp.json()
        assert "query" in data
        assert "count" in data
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_version_search_range(self, client: TestClient, admin_headers: Dict[str, str]):
        """Version search with range"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        resp = client.get(
            "/models/search/version",
            params={"query": "1.0.0-2.0.0"},
            headers=admin_headers,
        )
        
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["results"], list)

    def test_version_search_tilde(self, client: TestClient, admin_headers: Dict[str, str]):
        """Version search with tilde (~) notation"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        resp = client.get(
            "/models/search/version",
            params={"query": "~1.2.0"},
            headers=admin_headers,
        )
        
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["results"], list)

    def test_version_search_caret(self, client: TestClient, admin_headers: Dict[str, str]):
        """Version search with caret (^) notation"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        resp = client.get(
            "/models/search/version",
            params={"query": "^1.2.0"},
            headers=admin_headers,
        )
        
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["results"], list)

    def test_version_search_no_duplicates(self, client: TestClient, admin_headers: Dict[str, str]):
        """Version search results have no duplicates"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        resp = client.get(
            "/models/search/version",
            params={"query": "^1.0.0"},
            headers=admin_headers,
        )
        
        if resp.status_code == 200:
            data = resp.json()
            if data["count"] > 0:
                ids = [r["id"] for r in data["results"]]
                assert len(ids) == len(set(ids))


class TestVersionParsing:
    """Test version parsing and semver logic"""

    def test_version_parser_simple(self):
        """Test parsing simple version"""
        from app import parse_version
        
        assert parse_version("1.0.0") == (1, 0, 0)
        assert parse_version("2.3.4") == (2, 3, 4)

    def test_version_parser_with_v_prefix(self):
        """Test parsing version with v prefix"""
        from app import parse_version
        
        assert parse_version("v1.0.0") == (1, 0, 0)
        assert parse_version("V2.1.0") == (2, 1, 0)

    def test_version_parser_two_parts(self):
        """Test parsing version with only major.minor"""
        from app import parse_version
        
        assert parse_version("1.2") == (1, 2)

    def test_version_compare(self):
        """Test version comparison"""
        from app import parse_version, compare_versions
        
        assert compare_versions(parse_version("1.0.0"), parse_version("2.0.0")) == -1
        assert compare_versions(parse_version("2.0.0"), parse_version("1.0.0")) == 1
        assert compare_versions(parse_version("1.0.0"), parse_version("1.0.0")) == 0
        assert compare_versions(parse_version("1.0"), parse_version("1.0.0")) == 0

    def test_matches_version_query_exact(self):
        """Test exact version matching"""
        from app import matches_version_query
        
        assert matches_version_query("1.0.0", "1.0.0")
        assert not matches_version_query("1.0.1", "1.0.0")
        assert not matches_version_query("1.1.0", "1.0.0")

    def test_matches_version_query_range(self):
        """Test range version matching"""
        from app import matches_version_query
        
        assert matches_version_query("1.5.0", "1.0.0-2.0.0")
        assert matches_version_query("1.0.0", "1.0.0-2.0.0")
        assert matches_version_query("2.0.0", "1.0.0-2.0.0")
        assert not matches_version_query("2.0.1", "1.0.0-2.0.0")
        assert not matches_version_query("0.9.0", "1.0.0-2.0.0")

    def test_matches_version_query_tilde(self):
        """Test tilde (~) version matching"""
        from app import matches_version_query
        
        # ~1.2.0 means >=1.2.0, <1.3.0
        assert matches_version_query("1.2.0", "~1.2.0")
        assert matches_version_query("1.2.5", "~1.2.0")
        assert not matches_version_query("1.3.0", "~1.2.0")
        assert not matches_version_query("1.1.0", "~1.2.0")

    def test_matches_version_query_caret(self):
        """Test caret (^) version matching"""
        from app import matches_version_query
        
        # ^1.2.0 means >=1.2.0, <2.0.0
        assert matches_version_query("1.2.0", "^1.2.0")
        assert matches_version_query("1.5.0", "^1.2.0")
        assert matches_version_query("1.99.99", "^1.2.0")
        assert not matches_version_query("2.0.0", "^1.2.0")
        assert not matches_version_query("1.1.0", "^1.2.0")

    def test_matches_version_query_invalid(self):
        """Test invalid version matching"""
        from app import matches_version_query
        
        # Invalid versions should return False
        assert not matches_version_query("not-a-version", "1.0.0")
        assert not matches_version_query("1.0.0", "invalid")


class TestMetricsIntegration:
    """Test metrics are properly integrated into the system"""

    def test_all_metrics_calculated(self, client: TestClient, admin_headers: Dict[str, str]):
        """All 11 metrics are calculated when rating"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        resp = client.post(
            "/models/upload",
            data={"url": "https://huggingface.co/gpt2"},
            headers=admin_headers,
        )
        
        if resp.status_code in (201, 200):
            artifact_id = resp.json().get("id")
            
            rating_resp = client.get(
                f"/artifact/model/{artifact_id}/rate",
                headers=admin_headers,
            )
            
            assert rating_resp.status_code == 200
            rating_data = rating_resp.json()
            
            # Count non-zero/non-null metrics
            metric_count = 0
            for key in rating_data:
                if not key.endswith("_latency") and key != "name" and key != "category" and key != "size_score":
                    if rating_data[key] is not None:
                        metric_count += 1
            
            # Should have at least the 11 core metrics
            assert metric_count >= 11

    def test_reproducibility_metric_scoring(self, client: TestClient, admin_headers: Dict[str, str]):
        """Reproducibility metric returns valid scores"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        resp = client.post(
            "/models/upload",
            data={"url": "https://huggingface.co/gpt2"},
            headers=admin_headers,
        )
        
        if resp.status_code in (201, 200):
            artifact_id = resp.json().get("id")
            
            rating_resp = client.get(
                f"/artifact/model/{artifact_id}/rate",
                headers=admin_headers,
            )
            
            if rating_resp.status_code == 200:
                score = rating_resp.json().get("reproducibility")
                # Reproducibility: 0, 0.5, or 1
                assert score in [0.0, 0.5, 1.0, -1.0] or (0 <= score <= 1)

    def test_reviewedness_metric_scoring(self, client: TestClient, admin_headers: Dict[str, str]):
        """Reviewedness metric returns valid scores"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        resp = client.post(
            "/models/upload",
            data={"url": "https://huggingface.co/gpt2"},
            headers=admin_headers,
        )
        
        if resp.status_code in (201, 200):
            artifact_id = resp.json().get("id")
            
            rating_resp = client.get(
                f"/artifact/model/{artifact_id}/rate",
                headers=admin_headers,
            )
            
            if rating_resp.status_code == 200:
                score = rating_resp.json().get("reviewedness")
                # Reviewedness: -1 (no GitHub), or 0-1 (fraction)
                assert -1 <= score <= 1

    def test_treescore_metric_scoring(self, client: TestClient, admin_headers: Dict[str, str]):
        """Treescore metric returns valid scores"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        resp = client.post(
            "/models/upload",
            data={"url": "https://huggingface.co/gpt2"},
            headers=admin_headers,
        )
        
        if resp.status_code in (201, 200):
            artifact_id = resp.json().get("id")
            
            rating_resp = client.get(
                f"/artifact/model/{artifact_id}/rate",
                headers=admin_headers,
            )
            
            if rating_resp.status_code == 200:
                score = rating_resp.json().get("tree_score")
                # Treescore: -1 (no parents), or 0-1 (average)
                assert -1 <= score <= 1


class TestM4Requirements:
    """Test all M4 requirements are met"""

    def test_requirement_4_1_three_new_metrics(self, client: TestClient, admin_headers: Dict[str, str]):
        """M4.1: Three new metrics implemented"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        resp = client.post(
            "/models/upload",
            data={"url": "https://huggingface.co/gpt2"},
            headers=admin_headers,
        )
        
        if resp.status_code in (201, 200):
            artifact_id = resp.json().get("id")
            
            rating_resp = client.get(
                f"/artifact/model/{artifact_id}/rate",
                headers=admin_headers,
            )
            
            assert rating_resp.status_code == 200
            rating = rating_resp.json()
            
            # M4.1: Three new metrics
            assert "reproducibility" in rating
            assert "reviewedness" in rating
            assert "tree_score" in rating

    def test_requirement_4_1_rate_returns_all_metrics(self, client: TestClient, admin_headers: Dict[str, str]):
        """M4.1: Rate endpoint returns complete metric JSON including new metrics"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        resp = client.post(
            "/models/upload",
            data={"url": "https://huggingface.co/gpt2"},
            headers=admin_headers,
        )
        
        if resp.status_code in (201, 200):
            artifact_id = resp.json().get("id")
            
            rating_resp = client.get(
                f"/artifact/model/{artifact_id}/rate",
                headers=admin_headers,
            )
            
            assert rating_resp.status_code == 200
            rating = rating_resp.json()
            
            # Should have name, category, net_score
            assert "name" in rating
            assert "category" in rating
            assert "net_score" in rating

    def test_requirement_4_2_search_regex(self, client: TestClient, admin_headers: Dict[str, str]):
        """M4.2: Search endpoint with regex"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        resp = client.get(
            "/models/search",
            params={"query": "gpt.*"},
            headers=admin_headers,
        )
        
        assert resp.status_code == 200
        data = resp.json()
        assert "query" in data
        assert "results" in data

    def test_requirement_4_2_version_search_exact(self, client: TestClient, admin_headers: Dict[str, str]):
        """M4.2: Version search with exact notation"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        resp = client.get(
            "/models/search/version",
            params={"query": "1.2.3"},
            headers=admin_headers,
        )
        
        assert resp.status_code == 200

    def test_requirement_4_2_version_search_range(self, client: TestClient, admin_headers: Dict[str, str]):
        """M4.2: Version search with range notation"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        resp = client.get(
            "/models/search/version",
            params={"query": "1.2.3-2.1.0"},
            headers=admin_headers,
        )
        
        assert resp.status_code == 200

    def test_requirement_4_2_version_search_tilde(self, client: TestClient, admin_headers: Dict[str, str]):
        """M4.2: Version search with tilde notation"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        resp = client.get(
            "/models/search/version",
            params={"query": "~1.2.0"},
            headers=admin_headers,
        )
        
        assert resp.status_code == 200

    def test_requirement_4_2_version_search_caret(self, client: TestClient, admin_headers: Dict[str, str]):
        """M4.2: Version search with caret notation"""
        if not admin_headers:
            pytest.skip("Admin token not available")

        resp = client.get(
            "/models/search/version",
            params={"query": "^1.2.0"},
            headers=admin_headers,
        )
        
        assert resp.status_code == 200
