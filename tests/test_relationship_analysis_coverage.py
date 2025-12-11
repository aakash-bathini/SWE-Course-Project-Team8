"""
Tests for relationship_analysis.py to increase code coverage.
Focuses on the new provider-agnostic LLM helper and heuristic fallbacks.
"""

import pytest
import json
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics.relationship_analysis import analyze_artifact_relationships, _heuristic_relationship_extraction


class TestRelationshipAnalysisLLM:
    """Test relationship analysis with provider-agnostic LLM helper"""

    @pytest.mark.asyncio
    async def test_extract_relationships_llm_success(self):
        """Test relationship extraction when cached_llm_chat returns valid JSON"""
        readme_text = "This model uses the SQuAD dataset and code from https://github.com/test/repo"
        hf_data = {}

        mock_response = json.dumps(
            {
                "relationships": [
                    {"relationship_type": "dataset", "name_or_url": "SQuAD", "confidence": 0.9},
                    {"relationship_type": "code_repo", "name_or_url": "https://github.com/test/repo", "confidence": 0.8},
                ]
            }
        )

        with patch("src.metrics.relationship_analysis.cached_llm_chat", return_value=mock_response):
            result = await analyze_artifact_relationships(readme_text, hf_data)

        assert "SQuAD" in result["linked_datasets"]
        assert "https://github.com/test/repo" in result["linked_code_repos"]
        assert result["relationship_confidence"] > 0.0

    @pytest.mark.asyncio
    async def test_extract_relationships_llm_parse_error_fallback(self):
        """Test relationship extraction falls back when LLM response cannot be parsed"""
        readme_text = "This model uses SQuAD dataset and https://github.com/test/repo"
        hf_data = {}

        with patch("src.metrics.relationship_analysis.cached_llm_chat", return_value="Invalid JSON response"), patch(
            "src.metrics.relationship_analysis._heuristic_relationship_extraction"
        ) as mock_heuristic:
            mock_heuristic.return_value = {
                "linked_datasets": ["SQuAD"],
                "linked_code_repos": ["https://github.com/test/repo"],
                "relationship_confidence": 0.5,
            }

            result = await analyze_artifact_relationships(readme_text, hf_data)

        assert result["linked_datasets"] == ["SQuAD"]
        assert result["linked_code_repos"] == ["https://github.com/test/repo"]

    @pytest.mark.asyncio
    async def test_extract_relationships_llm_unavailable(self):
        """Test relationship extraction when no LLM provider is configured"""
        readme_text = "README content"
        hf_data = {}

        with patch("src.metrics.relationship_analysis.cached_llm_chat", return_value=None), patch(
            "src.metrics.relationship_analysis._heuristic_relationship_extraction", return_value={"linked_datasets": [], "linked_code_repos": [], "relationship_confidence": 0.0}
        ) as mock_heuristic:
            result = await analyze_artifact_relationships(readme_text, hf_data)

        mock_heuristic.assert_called_once()
        assert "linked_datasets" in result
        assert "linked_code_repos" in result

    @pytest.mark.asyncio
    async def test_extract_relationships_filters_invalid_relationships(self):
        """Test relationship extraction filters invalid relationship types and entries"""
        readme_text = "Test README"
        hf_data = {}

        mock_response = json.dumps(
            {
                "relationships": [
                    {"relationship_type": "dataset", "name_or_url": "valid", "confidence": 0.8},
                    {"relationship_type": "invalid_type", "name_or_url": "invalid", "confidence": 0.5},
                    {"relationship_type": "code_repo", "name_or_url": "", "confidence": 0.7},
                    "not_a_dict",
                ]
            }
        )

        with patch("src.metrics.relationship_analysis.cached_llm_chat", return_value=mock_response):
            result = await analyze_artifact_relationships(readme_text, hf_data)

        assert result["linked_datasets"] == ["valid"]
        assert result["linked_code_repos"] == []

    @pytest.mark.asyncio
    async def test_extract_relationships_empty_confidence_scores(self):
        """Test relationship extraction when LLM returns empty relationship list"""
        readme_text = "Test README"
        hf_data = {}

        mock_response = json.dumps({"relationships": []})

        with patch("src.metrics.relationship_analysis.cached_llm_chat", return_value=mock_response):
            result = await analyze_artifact_relationships(readme_text, hf_data)

        assert result["relationship_confidence"] == 0.0
        assert len(result["linked_datasets"]) == 0
        assert len(result["linked_code_repos"]) == 0


class TestHeuristicRelationshipExtraction:
    """Test heuristic relationship extraction fallback"""

    def test_heuristic_extraction_github_urls(self):
        """Test heuristic extraction finds GitHub URLs"""
        readme_text = "Check out https://github.com/user/repo and https://github.com/org/project"
        result = _heuristic_relationship_extraction(readme_text)

        assert len(result["linked_code_repos"]) >= 2
        assert "https://github.com/user/repo" in result["linked_code_repos"]
        assert "https://github.com/org/project" in result["linked_code_repos"]

    def test_heuristic_extraction_dataset_urls(self):
        """Test heuristic extraction finds dataset URLs"""
        readme_text = (
            "Dataset: https://huggingface.co/datasets/squad "
            "and https://kaggle.com/datasets/test "
            "and https://zenodo.org/record/123"
        )
        result = _heuristic_relationship_extraction(readme_text)

        assert len(result["linked_datasets"]) >= 3

    def test_heuristic_extraction_common_datasets(self):
        """Test heuristic extraction finds common dataset names"""
        readme_text = "This model was trained on SQuAD, ImageNet, and CIFAR datasets"
        result = _heuristic_relationship_extraction(readme_text)

        assert "squad" in result["linked_datasets"] or "SQuAD" in result["linked_datasets"]
        assert "imagenet" in result["linked_datasets"] or "ImageNet" in result["linked_datasets"]
        assert "cifar" in result["linked_datasets"] or "CIFAR" in result["linked_datasets"]

    def test_heuristic_extraction_removes_duplicates(self):
        """Test heuristic extraction removes duplicate entries"""
        readme_text = (
            "Repo: https://github.com/user/repo. " "Also see https://github.com/user/repo. " "Dataset: SQuAD and squad"
        )
        result = _heuristic_relationship_extraction(readme_text)

        # Should have unique entries
        assert len(result["linked_code_repos"]) == 1
        assert len(set(result["linked_datasets"])) == len(result["linked_datasets"])

    def test_heuristic_extraction_no_matches(self):
        """Test heuristic extraction with no matches"""
        readme_text = "This is a plain README with no URLs or dataset names"
        result = _heuristic_relationship_extraction(readme_text)

        assert len(result["linked_datasets"]) == 0
        assert len(result["linked_code_repos"]) == 0
        assert result["relationship_confidence"] == 0.0

    def test_heuristic_extraction_with_matches(self):
        """Test heuristic extraction confidence when matches found"""
        readme_text = "Dataset: SQuAD. Code: https://github.com/test/repo"
        result = _heuristic_relationship_extraction(readme_text)

        assert result["relationship_confidence"] == 0.5
        assert len(result["linked_datasets"]) > 0 or len(result["linked_code_repos"]) > 0

    def test_heuristic_extraction_with_hf_data(self):
        """Test heuristic extraction with hf_data parameter"""
        readme_text = "Test README"
        hf_data = {"some": "data"}

        result = _heuristic_relationship_extraction(readme_text, hf_data)

        assert "linked_datasets" in result
        assert "linked_code_repos" in result
        assert "relationship_confidence" in result
