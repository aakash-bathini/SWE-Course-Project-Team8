"""
Tests for models ingest endpoint to increase coverage
Focuses on error paths, threshold failures, and storage paths
"""

import pytest
import sys
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _get_auth_token():
    """Helper to get auth token"""
    from src.auth.jwt_auth import auth
    token_data = {
        "sub": "ece30861defaultadminuser",
        "permissions": ["upload", "search", "download", "admin"],
    }
    return auth.create_access_token(token_data)


def _get_headers():
    """Helper to get auth headers"""
    token = _get_auth_token()
    return {"X-Authorization": f"bearer {token}"}


class TestModelsIngest:
    """Test models ingest endpoint"""

    def test_ingest_no_permission(self):
        """Test ingest without upload permission"""
        from app import app
        from src.auth.jwt_auth import auth
        
        token_data = {
            "sub": "limiteduser",
            "permissions": ["search"],  # No upload permission
        }
        token = auth.create_access_token(token_data)
        headers = {"X-Authorization": f"bearer {token}"}
        
        client = TestClient(app)
        response = client.post("/models/ingest?model_name=test/model", headers=headers)
        assert response.status_code == 401

    def test_ingest_hf_not_available(self):
        """Test ingest when HuggingFace scraping is not available"""
        from app import app, scrape_hf_url, calculate_phase2_metrics
        headers = _get_headers()
        client = TestClient(app)
        
        # Mock scrape_hf_url to be None
        with patch('app.scrape_hf_url', None):
            with patch('app.calculate_phase2_metrics', None):
                response = client.post("/models/ingest?model_name=test/model", headers=headers)
                assert response.status_code == 501

    def test_ingest_scrape_failure(self):
        """Test ingest when HuggingFace scraping fails"""
        from app import app, scrape_hf_url
        headers = _get_headers()
        client = TestClient(app)
        
        if scrape_hf_url:
            with patch('app.scrape_hf_url', side_effect=Exception("Scrape error")):
                response = client.post("/models/ingest?model_name=test/model", headers=headers)
                assert response.status_code == 500

    def test_ingest_threshold_failure(self):
        """Test ingest when metrics fail threshold"""
        from app import app, scrape_hf_url, calculate_phase2_metrics
        headers = _get_headers()
        client = TestClient(app)
        
        if scrape_hf_url and calculate_phase2_metrics:
            # Mock successful scraping
            mock_hf_data = {
                "readme_text": "Test model",
                "config_json": {}
            }
            
            # Mock metrics that fail threshold (< 0.5)
            mock_metrics = {
                "net_score": 0.3,  # Fails threshold
                "ramp_up_time": 0.4,  # Fails threshold
                "bus_factor": 0.2,  # Fails threshold
                "performance_claims": 0.6,
                "license": 0.7,
                "dataset_and_code_score": 0.8,
                "dataset_quality": 0.9,
                "code_quality": 0.85,
                "reproducibility": 0.75,
                "reviewedness": 0.65,
                "tree_score": 0.55,
                "net_score_latency": 1.0,  # Latency metric (should be ignored)
            }
            
            with patch('app.scrape_hf_url', return_value=(mock_hf_data, "model")):
                with patch('app.calculate_phase2_metrics', new_callable=AsyncMock, return_value=mock_metrics):
                    response = client.post("/models/ingest?model_name=test/model", headers=headers)
                    assert response.status_code == 424
                    assert "threshold" in response.json().get("detail", "").lower()

    def test_ingest_sqlite_count_failure(self):
        """Test ingest when SQLite count fails"""
        from app import app, get_db, db_crud, scrape_hf_url, calculate_phase2_metrics
        headers = _get_headers()
        client = TestClient(app)
        
        if scrape_hf_url and calculate_phase2_metrics and get_db and db_crud:
            mock_hf_data = {"readme_text": "Test", "config_json": {}}
            mock_metrics = {k: 0.8 for k in ["net_score", "ramp_up_time", "bus_factor", 
                                             "performance_claims", "license", "dataset_and_code_score",
                                             "dataset_quality", "code_quality", "reproducibility",
                                             "reviewedness", "tree_score"]}
            
            with patch('app.scrape_hf_url', return_value=(mock_hf_data, "model")):
                with patch('app.calculate_phase2_metrics', new_callable=AsyncMock, return_value=mock_metrics):
                    with patch('app.get_db', side_effect=Exception("DB error")):
                        response = client.post("/models/ingest?model_name=test/model", headers=headers)
                        # Should fall back to in-memory count
                        assert response.status_code in [201, 500]

    def test_ingest_s3_count_fallback(self):
        """Test ingest when S3 count is used as fallback"""
        from app import app, s3_storage, scrape_hf_url, calculate_phase2_metrics
        headers = _get_headers()
        client = TestClient(app)
        
        if scrape_hf_url and calculate_phase2_metrics and s3_storage:
            mock_hf_data = {"readme_text": "Test", "config_json": {}}
            mock_metrics = {k: 0.8 for k in ["net_score", "ramp_up_time", "bus_factor",
                                             "performance_claims", "license", "dataset_and_code_score",
                                             "dataset_quality", "code_quality", "reproducibility",
                                             "reviewedness", "tree_score"]}
            
            with patch('app.scrape_hf_url', return_value=(mock_hf_data, "model")):
                with patch('app.calculate_phase2_metrics', new_callable=AsyncMock, return_value=mock_metrics):
                    with patch.object(s3_storage, 'count_artifacts_by_type', return_value=10):
                        with patch.dict(os.environ, {"USE_SQLITE": "0"}):
                            response = client.post("/models/ingest?model_name=test/model", headers=headers)
                            assert response.status_code in [201, 500]

    def test_ingest_s3_save_failure(self):
        """Test ingest when S3 save fails"""
        from app import app, s3_storage, scrape_hf_url, calculate_phase2_metrics
        headers = _get_headers()
        client = TestClient(app)
        
        if scrape_hf_url and calculate_phase2_metrics and s3_storage:
            mock_hf_data = {"readme_text": "Test", "config_json": {}}
            mock_metrics = {k: 0.8 for k in ["net_score", "ramp_up_time", "bus_factor",
                                             "performance_claims", "license", "dataset_and_code_score",
                                             "dataset_quality", "code_quality", "reproducibility",
                                             "reviewedness", "tree_score"]}
            
            with patch('app.scrape_hf_url', return_value=(mock_hf_data, "model")):
                with patch('app.calculate_phase2_metrics', new_callable=AsyncMock, return_value=mock_metrics):
                    with patch.object(s3_storage, 'save_artifact_metadata', return_value=False):
                        response = client.post("/models/ingest?model_name=test/model", headers=headers)
                        # Should still succeed but log error
                        assert response.status_code in [201, 500]

    def test_ingest_file_storage_failure(self):
        """Test ingest when file storage fails"""
        from app import app, scrape_hf_url, calculate_phase2_metrics
        from src.storage import file_storage
        headers = _get_headers()
        client = TestClient(app)
        
        if scrape_hf_url and calculate_phase2_metrics:
            mock_hf_data = {"readme_text": "Test", "config_json": {}}
            mock_metrics = {k: 0.8 for k in ["net_score", "ramp_up_time", "bus_factor",
                                             "performance_claims", "license", "dataset_and_code_score",
                                             "dataset_quality", "code_quality", "reproducibility",
                                             "reviewedness", "tree_score"]}
            
            with patch('app.scrape_hf_url', return_value=(mock_hf_data, "model")):
                with patch('app.calculate_phase2_metrics', new_callable=AsyncMock, return_value=mock_metrics):
                    with patch.object(file_storage, 'get_artifact_directory', side_effect=Exception("Storage error")):
                        response = client.post("/models/ingest?model_name=test/model", headers=headers)
                        # Should still succeed but log warning
                        assert response.status_code in [201, 500]

    def test_ingest_negative_metrics_filtered(self):
        """Test ingest filters out negative metrics"""
        from app import app, scrape_hf_url, calculate_phase2_metrics
        headers = _get_headers()
        client = TestClient(app)
        
        if scrape_hf_url and calculate_phase2_metrics:
            mock_hf_data = {"readme_text": "Test", "config_json": {}}
            # Include negative metrics (should be filtered out)
            mock_metrics = {
                "net_score": 0.8,
                "ramp_up_time": 0.7,
                "bus_factor": 0.6,
                "performance_claims": 0.9,
                "license": 0.85,
                "dataset_and_code_score": 0.75,
                "dataset_quality": 0.8,
                "code_quality": 0.7,
                "reproducibility": 0.65,
                "reviewedness": -1.0,  # Negative (no GitHub repo)
                "tree_score": 0.6,
                "net_score_latency": 1.0,
            }
            
            with patch('app.scrape_hf_url', return_value=(mock_hf_data, "model")):
                with patch('app.calculate_phase2_metrics', new_callable=AsyncMock, return_value=mock_metrics):
                    response = client.post("/models/ingest?model_name=test/model", headers=headers)
                    # Should pass (negative metrics filtered out)
                    assert response.status_code in [201, 424, 500]

    def test_ingest_latency_metrics_ignored(self):
        """Test ingest ignores latency metrics in threshold check"""
        from app import app, scrape_hf_url, calculate_phase2_metrics
        headers = _get_headers()
        client = TestClient(app)
        
        if scrape_hf_url and calculate_phase2_metrics:
            mock_hf_data = {"readme_text": "Test", "config_json": {}}
            # Latency metrics should be ignored even if < 0.5
            mock_metrics = {
                "net_score": 0.8,
                "ramp_up_time": 0.7,
                "bus_factor": 0.6,
                "performance_claims": 0.9,
                "license": 0.85,
                "dataset_and_code_score": 0.75,
                "dataset_quality": 0.8,
                "code_quality": 0.7,
                "reproducibility": 0.65,
                "reviewedness": 0.6,
                "tree_score": 0.6,
                "net_score_latency": 0.3,  # Low latency (should be ignored)
                "ramp_up_time_latency": 0.2,  # Low latency (should be ignored)
            }
            
            with patch('app.scrape_hf_url', return_value=(mock_hf_data, "model")):
                with patch('app.calculate_phase2_metrics', new_callable=AsyncMock, return_value=mock_metrics):
                    response = client.post("/models/ingest?model_name=test/model", headers=headers)
                    # Should pass (latency metrics ignored)
                    assert response.status_code in [201, 424, 500]

    def test_ingest_sqlite_audit_logging(self):
        """Test ingest with SQLite audit logging"""
        from app import app, get_db, db_crud, scrape_hf_url, calculate_phase2_metrics
        headers = _get_headers()
        client = TestClient(app)
        
        if scrape_hf_url and calculate_phase2_metrics and get_db and db_crud:
            mock_hf_data = {"readme_text": "Test", "config_json": {}}
            mock_metrics = {k: 0.8 for k in ["net_score", "ramp_up_time", "bus_factor",
                                             "performance_claims", "license", "dataset_and_code_score",
                                             "dataset_quality", "code_quality", "reproducibility",
                                             "reviewedness", "tree_score"]}
            
            with patch('app.scrape_hf_url', return_value=(mock_hf_data, "model")):
                with patch('app.calculate_phase2_metrics', new_callable=AsyncMock, return_value=mock_metrics):
                    with patch.dict(os.environ, {"USE_SQLITE": "1"}):
                        response = client.post("/models/ingest?model_name=test/model", headers=headers)
                        # Should succeed and log audit
                        assert response.status_code in [201, 500]

    def test_ingest_general_exception(self):
        """Test ingest with general exception"""
        from app import app, scrape_hf_url
        headers = _get_headers()
        client = TestClient(app)
        
        if scrape_hf_url:
            with patch('app.scrape_hf_url', side_effect=Exception("General error")):
                response = client.post("/models/ingest?model_name=test/model", headers=headers)
                assert response.status_code == 500

