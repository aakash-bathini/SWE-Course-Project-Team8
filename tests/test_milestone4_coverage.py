"""
Additional coverage tests to reach 50%+ for Milestone 4
Focus on testing logic paths that improve coverage without auth complexity
"""

import pytest
from typing import Dict, Any, List


class TestHelperFunctionsCoverage:
    """Direct tests of helper functions to improve coverage"""

    def test_parse_version_various_formats(self):
        """Test parse_version with many different formats"""
        from app import parse_version

        # 3-part versions
        assert parse_version("0.0.0") == (0, 0, 0)
        assert parse_version("1.0.0") == (1, 0, 0)
        assert parse_version("10.20.30") == (10, 20, 30)

        # 2-part versions
        assert parse_version("1.0") == (1, 0)
        assert parse_version("5.10") == (5, 10)

        # 4+ part versions
        assert parse_version("1.2.3.4") == (1, 2, 3, 4)
        assert parse_version("1.2.3.4.5") == (1, 2, 3, 4, 5)

        # With v prefix (various cases)
        assert parse_version("v1.0.0") == (1, 0, 0)
        assert parse_version("V1.0.0") == (1, 0, 0)
        assert parse_version("v10.20.30") == (10, 20, 30)

        # Invalid formats
        assert parse_version("") == (0,)
        assert parse_version("abc") == (0,)
        assert parse_version("1.a.0") == (0,)
        assert parse_version("...") == (0,)

    def test_compare_versions_all_directions(self):
        """Test compare_versions in all comparison directions"""
        from app import compare_versions

        # Less than
        assert compare_versions((1, 0, 0), (1, 0, 1)) == -1
        assert compare_versions((1, 0, 0), (1, 1, 0)) == -1
        assert compare_versions((1, 0, 0), (2, 0, 0)) == -1
        assert compare_versions((0, 0, 1), (0, 1, 0)) == -1

        # Equal
        assert compare_versions((1, 0, 0), (1, 0, 0)) == 0
        assert compare_versions((0, 0, 0), (0, 0, 0)) == 0
        assert compare_versions((2, 3, 4), (2, 3, 4)) == 0

        # Greater than
        assert compare_versions((1, 0, 1), (1, 0, 0)) == 1
        assert compare_versions((1, 1, 0), (1, 0, 0)) == 1
        assert compare_versions((2, 0, 0), (1, 0, 0)) == 1

        # Different length padding
        assert compare_versions((1, 0), (1, 0, 0)) == 0
        assert compare_versions((1,), (1, 0)) == 0
        assert compare_versions((1, 1), (1, 0, 0)) == 1
        assert compare_versions((1,), (1, 1)) == -1

    def test_matches_version_exact_comprehensive(self):
        """Comprehensive test of exact version matching"""
        from app import matches_version_query

        # Single version matches
        assert matches_version_query("1.0.0", "1.0.0") is True
        assert matches_version_query("0.0.0", "0.0.0") is True
        assert matches_version_query("99.99.99", "99.99.99") is True

        # Non-matches
        assert matches_version_query("1.0.0", "1.0.1") is False
        assert matches_version_query("2.0.0", "1.0.0") is False
        assert matches_version_query("0.0.1", "0.0.0") is False

    def test_matches_version_range_comprehensive(self):
        """Comprehensive test of range version matching"""
        from app import matches_version_query

        # Simple ranges
        assert matches_version_query("1.5.0", "1.0.0-2.0.0") is True
        assert matches_version_query("1.0.0", "1.0.0-1.5.0") is True
        assert matches_version_query("1.5.0", "1.0.0-1.5.0") is True

        # Boundary tests
        assert matches_version_query("0.9.9", "1.0.0-2.0.0") is False
        assert matches_version_query("2.0.1", "1.0.0-2.0.0") is False
        assert matches_version_query("1.0.0", "1.0.0-2.0.0") is True
        assert matches_version_query("2.0.0", "1.0.0-2.0.0") is True

        # Wide ranges
        assert matches_version_query("1.0.0", "0.0.0-99.99.99") is True
        assert matches_version_query("50.50.50", "0.0.0-99.99.99") is True

    def test_matches_version_tilde_comprehensive(self):
        """Comprehensive test of tilde version matching"""
        from app import matches_version_query

        # Basic tilde
        assert matches_version_query("1.2.0", "~1.2.0") is True
        assert matches_version_query("1.2.1", "~1.2.0") is True
        assert matches_version_query("1.2.999", "~1.2.0") is True

        # Tilde boundaries
        assert matches_version_query("1.1.999", "~1.2.0") is False
        assert matches_version_query("1.3.0", "~1.2.0") is False

        # Tilde with different base versions
        assert matches_version_query("2.0.5", "~2.0.0") is True
        assert matches_version_query("2.0.999", "~2.0.0") is True
        assert matches_version_query("2.1.0", "~2.0.0") is False

    def test_matches_version_caret_comprehensive(self):
        """Comprehensive test of caret version matching"""
        from app import matches_version_query

        # Normal caret (major > 0)
        assert matches_version_query("1.2.0", "^1.2.0") is True
        assert matches_version_query("1.9.9", "^1.2.0") is True
        assert matches_version_query("1.99.99", "^1.2.0") is True

        # Caret boundaries (major > 0)
        assert matches_version_query("1.1.999", "^1.2.0") is False
        assert matches_version_query("2.0.0", "^1.2.0") is False

        # Special case: major = 0
        assert matches_version_query("0.2.0", "^0.2.0") is True
        assert matches_version_query("0.2.999", "^0.2.0") is True
        assert matches_version_query("0.3.0", "^0.2.0") is False
        assert matches_version_query("1.0.0", "^0.2.0") is False

        # Multiple major versions with caret
        assert matches_version_query("2.5.0", "^2.0.0") is True
        assert matches_version_query("3.0.0", "^2.0.0") is False

    def test_matches_version_with_v_prefix(self):
        """Test version matching with v prefix in both version and query"""
        from app import matches_version_query

        # v in version only
        assert matches_version_query("v1.0.0", "1.0.0") is True

        # v in query only
        assert matches_version_query("1.0.0", "v1.0.0") is True

        # v in both
        assert matches_version_query("v1.0.0", "v1.0.0") is True

        # v with ranges
        assert matches_version_query("v1.5.0", "v1.0.0-v2.0.0") is True
        assert matches_version_query("1.5.0", "1.0.0-2.0.0") is True

        # v with tilde
        assert matches_version_query("v1.2.5", "~v1.2.0") is True
        assert matches_version_query("1.2.5", "~1.2.0") is True

        # v with caret
        assert matches_version_query("v1.5.0", "^v1.2.0") is True
        assert matches_version_query("1.5.0", "^1.2.0") is True

    def test_matches_version_edge_cases(self):
        """Test edge cases in version matching"""
        from app import matches_version_query

        # Zero versions
        assert matches_version_query("0.0.0", "0.0.0") is True
        assert matches_version_query("0.0.1", "0.0.0-0.1.0") is True

        # Large version numbers
        assert matches_version_query("999.999.999", "999.999.999") is True
        assert matches_version_query("999.999.999", "900.0.0-1000.0.0") is True

        # Single digit versions
        assert matches_version_query("1", "1") is True
        assert matches_version_query("1.0", "1.0") is True

    def test_parse_version_consistency(self):
        """Test that parse_version produces consistent results"""
        from app import parse_version

        # Multiple calls should return identical results
        v1 = parse_version("1.2.3")
        v2 = parse_version("1.2.3")
        assert v1 == v2
        assert v1 is not v2  # Different object

        # Edge cases return consistent (0,)
        assert parse_version("invalid1") == (0,)
        assert parse_version("invalid2") == (0,)

    def test_compare_versions_consistency(self):
        """Test that compare_versions is consistent"""
        from app import compare_versions

        # Antisymmetry: if compare(a, b) == -1, then compare(b, a) == 1
        v1 = (1, 0, 0)
        v2 = (2, 0, 0)

        assert compare_versions(v1, v2) == -1
        assert compare_versions(v2, v1) == 1

        # Transitivity: if a < b and b < c, then a < c
        a = (1, 0, 0)
        b = (2, 0, 0)
        c = (3, 0, 0)

        assert compare_versions(a, b) == -1
        assert compare_versions(b, c) == -1
        assert compare_versions(a, c) == -1

    def test_matches_version_query_return_type(self):
        """Ensure matches_version_query always returns bool"""
        from app import matches_version_query

        # Should return bool for all cases
        test_cases = [
            ("1.0.0", "1.0.0"),
            ("1.5.0", "1.0.0-2.0.0"),
            ("1.5.0", "~1.2.0"),
            ("1.5.0", "^1.0.0"),
            ("invalid", "invalid"),
            ("", ""),
        ]

        for version, query in test_cases:
            result = matches_version_query(version, query)
            assert isinstance(result, bool), f"Expected bool for {version}, {query}, got {type(result)}"
