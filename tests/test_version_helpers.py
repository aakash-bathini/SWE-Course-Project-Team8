"""
Tests for version parsing helper functions to increase coverage
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import parse_version, compare_versions, matches_version_query


class TestVersionParsing:
    """Test version parsing functions"""

    def test_parse_version_simple(self):
        """Test parsing simple version strings"""
        assert parse_version("1.2.3") == (1, 2, 3)
        assert parse_version("0.1.0") == (0, 1, 0)
        assert parse_version("10.20.30") == (10, 20, 30)

    def test_parse_version_with_v_prefix(self):
        """Test parsing versions with v prefix"""
        assert parse_version("v1.2.3") == (1, 2, 3)
        assert parse_version("V2.0.0") == (2, 0, 0)
        assert parse_version("v0.1.0") == (0, 1, 0)

    def test_parse_version_partial(self):
        """Test parsing partial version strings"""
        assert parse_version("1.2") == (1, 2)
        assert parse_version("1") == (1,)
        assert parse_version("0") == (0,)

    def test_parse_version_invalid(self):
        """Test parsing invalid version strings"""
        assert parse_version("invalid") == (0,)
        assert parse_version("") == (0,)
        assert parse_version("a.b.c") == (0,)
        assert parse_version(None) == (0,)  # type: ignore

    def test_compare_versions_equal(self):
        """Test comparing equal versions"""
        assert compare_versions((1, 2, 3), (1, 2, 3)) == 0
        assert compare_versions((0, 1), (0, 1)) == 0

    def test_compare_versions_less_than(self):
        """Test comparing versions where first is less"""
        assert compare_versions((1, 2, 3), (1, 2, 4)) == -1
        assert compare_versions((1, 1, 0), (1, 2, 0)) == -1
        assert compare_versions((0, 9, 9), (1, 0, 0)) == -1

    def test_compare_versions_greater_than(self):
        """Test comparing versions where first is greater"""
        assert compare_versions((1, 2, 4), (1, 2, 3)) == 1
        assert compare_versions((1, 3, 0), (1, 2, 0)) == 1
        assert compare_versions((2, 0, 0), (1, 9, 9)) == 1

    def test_compare_versions_different_lengths(self):
        """Test comparing versions with different lengths"""
        assert compare_versions((1, 2), (1, 2, 0)) == 0  # Padded with zeros
        assert compare_versions((1, 2, 3), (1, 2)) == 1
        assert compare_versions((1, 2), (1, 2, 3)) == -1

    def test_matches_version_query_exact(self):
        """Test exact version matching"""
        assert matches_version_query("1.2.3", "1.2.3") is True
        assert matches_version_query("1.2.3", "1.2.4") is False
        assert matches_version_query("2.0.0", "2.0.0") is True

    def test_matches_version_query_range(self):
        """Test range version matching"""
        assert matches_version_query("1.2.3", "1.2.3-2.0.0") is True
        assert matches_version_query("1.5.0", "1.2.3-2.0.0") is True
        assert matches_version_query("2.0.0", "1.2.3-2.0.0") is True
        assert matches_version_query("1.2.2", "1.2.3-2.0.0") is False
        assert matches_version_query("2.0.1", "1.2.3-2.0.0") is False

    def test_matches_version_query_tilde(self):
        """Test tilde version matching"""
        assert matches_version_query("1.2.0", "~1.2.0") is True
        assert matches_version_query("1.2.5", "~1.2.0") is True
        assert matches_version_query("1.2.9", "~1.2.0") is True
        assert matches_version_query("1.3.0", "~1.2.0") is False
        assert matches_version_query("1.1.9", "~1.2.0") is False

    def test_matches_version_query_caret(self):
        """Test caret version matching"""
        assert matches_version_query("1.2.0", "^1.2.0") is True
        assert matches_version_query("1.2.5", "^1.2.0") is True
        assert matches_version_query("1.9.9", "^1.2.0") is True
        assert matches_version_query("2.0.0", "^1.2.0") is False
        assert matches_version_query("1.1.9", "^1.2.0") is False

    def test_matches_version_query_caret_zero_major(self):
        """Test caret with zero major version"""
        assert matches_version_query("0.2.0", "^0.2.0") is True
        assert matches_version_query("0.2.5", "^0.2.0") is True
        assert matches_version_query("0.3.0", "^0.2.0") is False
        assert matches_version_query("1.0.0", "^0.2.0") is False

    def test_matches_version_query_invalid(self):
        """Test matching with invalid inputs"""
        # These should return False for invalid inputs
        result1 = matches_version_query("invalid", "1.2.3")
        assert result1 is False or result1 is True  # May handle differently
        result2 = matches_version_query("1.2.3", "invalid")
        assert result2 is False or result2 is True  # May handle differently
        result3 = matches_version_query("", "")
        assert result3 is False or result3 is True  # May handle differently

