"""
Test script to verify ArtifactType enum conversion works correctly
for both in-memory and S3 storage scenarios.

This test ensures that artifacts stored in S3 (as JSON strings) can be
correctly converted to ArtifactType enums when retrieved, preventing
S3/memory mismatches that could cause autograder failures.
"""

import pytest
from app import ArtifactType, ArtifactMetadata


def test_artifact_type_enum_conversion():
    """Test that string types can be converted to ArtifactType enum"""
    # Test cases: string -> ArtifactType enum
    test_cases = [
        ("model", ArtifactType.MODEL),
        ("dataset", ArtifactType.DATASET),
        ("code", ArtifactType.CODE),
    ]

    for type_str, expected_enum in test_cases:
        artifact_type_enum = ArtifactType(type_str)
        assert (
            artifact_type_enum == expected_enum
        ), f"Expected {expected_enum}, got {artifact_type_enum}"


def test_artifact_metadata_with_enum():
    """Test creating ArtifactMetadata with ArtifactType enum"""
    metadata = ArtifactMetadata(
        name="test-artifact",
        id="test-1-1234567890",
        type=ArtifactType.MODEL,
    )
    assert metadata.type == ArtifactType.MODEL
    assert metadata.type.value == "model"


def test_artifact_metadata_with_string():
    """Test that Pydantic can convert string to ArtifactType enum"""
    # Pydantic should auto-convert strings to enums
    metadata = ArtifactMetadata(
        name="test-artifact",
        id="test-1-1234567890",
        type="model",  # String should be auto-converted
    )
    assert metadata.type == ArtifactType.MODEL
    assert isinstance(metadata.type, ArtifactType)


def test_s3_memory_type_consistency():
    """
    Test that types from S3 (stored as JSON strings) can be converted
    to ArtifactType enums, simulating the production scenario where
    artifacts are stored in S3 and retrieved in a different Lambda invocation.
    """
    # Simulate S3 data structure (as it would be stored in JSON)
    s3_artifact_data = {
        "metadata": {
            "name": "test-model",
            "id": "model-1-1234567890",
            "type": "model",  # Stored as string in JSON
        },
        "data": {"url": "https://example.com/model"},
    }

    # Simulate in-memory data structure
    memory_artifact_data = {
        "metadata": {
            "name": "test-model",
            "id": "model-1-1234567890",
            "type": "model",  # Also stored as string
        },
        "data": {"url": "https://example.com/model"},
    }

    # Both should be convertible to ArtifactType enum
    for source_name, artifact_data in [("S3", s3_artifact_data), ("memory", memory_artifact_data)]:
        stored_type = artifact_data["metadata"]["type"]
        assert isinstance(stored_type, str), f"{source_name}: type should be string"

        # Convert to enum (as done in artifact_retrieve)
        artifact_type_enum = ArtifactType(stored_type)
        assert isinstance(
            artifact_type_enum, ArtifactType
        ), f"{source_name}: should be ArtifactType enum"
        assert artifact_type_enum == ArtifactType.MODEL, f"{source_name}: should be MODEL"

        # Create ArtifactMetadata (should work for both)
        metadata_dict = artifact_data["metadata"].copy()
        metadata_dict["type"] = artifact_type_enum
        metadata = ArtifactMetadata(**metadata_dict)
        assert metadata.type == ArtifactType.MODEL, f"{source_name}: metadata type should be MODEL"


def test_invalid_artifact_type():
    """Test that invalid artifact types raise ValueError"""
    with pytest.raises(ValueError):
        ArtifactType("invalid_type")
