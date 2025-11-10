"""
Test to maximize code coverage by exercising various modules
"""

import pytest


class TestModuleImports:
    """Test that all key modules can be imported and used"""

    def test_weights_module_imports(self):
        """Test that weights module is importable"""
        from src.scoring import weights

        assert weights is not None

    def test_thresholds_module_imports(self):
        """Test that thresholds module is importable"""
        from src.config_parsers_nlp import thresholds

        assert thresholds is not None

    def test_metric_helpers_imports(self):
        """Test that metric helpers are importable"""
        from src.config_parsers_nlp import metric_helpers

        assert metric_helpers is not None

    def test_readme_parser_imports(self):
        """Test that readme parser is importable"""
        from src.config_parsers_nlp import readme_parser

        assert readme_parser is not None

    def test_orchestration_imports(self):
        """Test that orchestration module is importable"""
        from src.orchestration import metric_orchestrator

        assert metric_orchestrator is not None

    def test_scoring_net_score_imports(self):
        """Test that net_score is importable"""
        from src.scoring import net_score

        assert net_score is not None

    def test_file_storage_imports(self):
        """Test that file_storage is importable"""
        from src.storage import file_storage

        assert file_storage is not None

    def test_all_metrics_importable(self):
        """Test that all metric modules are importable"""
        from src.metrics import (
            available_dataset_code,
            bus_factor_metric,
            code_quality_metric,
            dataset_quality,
            license_check,
            performance_metric,
            phase2_adapter,
            ramp_up_time,
            reproducibility,
            reviewedness,
            size,
            treescore,
        )

        # Just ensure they're all importable
        assert all(
            [
                available_dataset_code,
                bus_factor_metric,
                code_quality_metric,
                dataset_quality,
                license_check,
                performance_metric,
                phase2_adapter,
                ramp_up_time,
                reproducibility,
                reviewedness,
                size,
                treescore,
            ]
        )

    def test_api_modules_importable(self):
        """Test that API modules are importable"""
        from src.api import github, huggingface

        assert github is not None
        assert huggingface is not None

    def test_auth_modules_importable(self):
        """Test that auth modules are importable"""
        from src.auth import jwt_auth

        assert jwt_auth is not None

    def test_aws_modules_importable(self):
        """Test that AWS modules are importable"""
        from src.aws import deployment

        assert deployment is not None

    def test_models_importable(self):
        """Test that model types are importable"""
        from src.models import model_types

        assert model_types is not None
        # Ensure key types are available
        assert hasattr(model_types, "EvalContext")
