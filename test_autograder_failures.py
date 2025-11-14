#!/usr/bin/env python3
"""
Local test script to reproduce autograder failures:
- Artifact Read Test Group (ByName/ByID)
- Regex Tests Group (Exact Match)
- Rate models concurrently
"""

import sys
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from app import app
from src.auth.jwt_auth import auth


def get_auth_headers() -> Dict[str, str]:
    """Get authentication headers with admin permissions"""
    token = auth.create_access_token(
        {"sub": "test_user", "permissions": ["upload", "search", "download", "admin"]}
    )
    return {"X-Authorization": f"bearer {token}"}


def test_by_name_scenarios():
    """Test various ByName scenarios that are failing"""
    client = TestClient(app)
    headers = get_auth_headers()

    print("\n=== Testing ByName Scenarios ===")

    # Create test artifacts with various name formats
    test_artifacts = [
        {"name": "simple-model", "type": "model", "url": "https://huggingface.co/simple-model"},
        {"name": "org/model-name", "type": "model", "url": "https://huggingface.co/org/model-name"},
        {
            "name": "model_with_underscores",
            "type": "model",
            "url": "https://huggingface.co/model_with_underscores",
        },
        {
            "name": "Model-With-Mixed-Case",
            "type": "model",
            "url": "https://huggingface.co/Model-With-Mixed-Case",
        },
        {
            "name": "model.with.dots",
            "type": "model",
            "url": "https://huggingface.co/model.with.dots",
        },
    ]

    created_ids = []
    for artifact in test_artifacts:
        resp = client.post(
            f"/artifact/{artifact['type']}", json={"url": artifact["url"]}, headers=headers
        )
        if resp.status_code in (200, 201, 202):
            data = resp.json()
            created_ids.append((data["metadata"]["id"], artifact["name"]))
            print(f"✓ Created: {artifact['name']} -> {data['metadata']['id']}")
        else:
            print(f"✗ Failed to create {artifact['name']}: {resp.status_code} - {resp.text}")

    # Test ByName queries
    test_names = [
        "simple-model",
        "org/model-name",  # URL-encoded
        "Model-With-Mixed-Case",  # Case-insensitive should work
        "MODEL_WITH_UNDERSCORES",  # Case-insensitive
        "nonexistent-model-12345",  # Should return 404
    ]

    for name in test_names:
        # URL encode slashes
        encoded_name = name.replace("/", "%2F")
        resp = client.get(f"/artifact/byName/{encoded_name}", headers=headers)
        print(f"\nByName('{name}'):")
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            results = resp.json()
            print(f"  Results: {len(results)} matches")
            for r in results:
                print(f"    - {r['name']} ({r['id']})")
        elif resp.status_code == 404:
            print(f"  ✓ Correctly returned 404 (not found)")
        else:
            print(f"  ✗ Unexpected status: {resp.text[:200]}")


def test_by_id_scenarios():
    """Test various ByID scenarios that are failing"""
    client = TestClient(app)
    headers = get_auth_headers()

    print("\n=== Testing ByID Scenarios ===")

    # Create test artifacts
    test_artifacts = [
        {"type": "model", "url": "https://huggingface.co/test-model-1"},
        {"type": "dataset", "url": "https://huggingface.co/test-dataset-1"},
        {"type": "code", "url": "https://github.com/test/code-1"},
    ]

    created_ids = []
    for artifact in test_artifacts:
        resp = client.post(
            f"/artifact/{artifact['type']}", json={"url": artifact["url"]}, headers=headers
        )
        if resp.status_code in (200, 201, 202):
            data = resp.json()
            art_id = data["metadata"]["id"]
            art_type = data["metadata"]["type"]
            created_ids.append((art_id, art_type))
            print(f"✓ Created: {art_type} -> {art_id}")
        else:
            print(f"✗ Failed to create {artifact['type']}: {resp.status_code}")

    # Test ByID queries
    for art_id, art_type in created_ids:
        # Test both /artifacts/{type}/{id} and /artifact/{type}/{id}
        for route_prefix in ["/artifacts", "/artifact"]:
            resp = client.get(f"{route_prefix}/{art_type}/{art_id}", headers=headers)
            print(f"\nByID({art_type}/{art_id}) via {route_prefix}:")
            print(f"  Status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                print(f"  ✓ Found: {data['metadata']['name']}")
            else:
                print(f"  ✗ Failed: {resp.text[:200]}")

    # Test /package/{id} alias
    for art_id, art_type in created_ids:
        if art_type == "model":  # /package/{id} should work for models
            resp = client.get(f"/package/{art_id}", headers=headers)
            print(f"\nByID via /package/{art_id}:")
            print(f"  Status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                print(f"  ✓ Found: {data['metadata']['name']}")
            else:
                print(f"  ✗ Failed: {resp.text[:200]}")


def test_regex_exact_match():
    """Test regex exact match that's failing"""
    client = TestClient(app)
    headers = get_auth_headers()

    print("\n=== Testing Regex Exact Match ===")

    # Create test artifacts with known names
    test_names = ["exact-match-test", "prefix-exact-match-test-suffix", "other-model"]

    for name in test_names:
        resp = client.post(
            "/artifact/model", json={"url": f"https://huggingface.co/{name}"}, headers=headers
        )
        if resp.status_code in (200, 201, 202):
            print(f"✓ Created: {name}")
        else:
            print(f"✗ Failed to create {name}: {resp.status_code}")

    # Test exact match regex (^name$)
    exact_pattern = "^exact-match-test$"
    resp = client.post("/artifact/byRegEx", json={"regex": exact_pattern}, headers=headers)
    print(f"\nRegex exact match ('{exact_pattern}'):")
    print(f"  Status: {resp.status_code}")
    if resp.status_code == 200:
        results = resp.json()
        print(f"  Results: {len(results)} matches")
        for r in results:
            print(f"    - {r['name']} ({r['id']})")
        # Should only match "exact-match-test", not "prefix-exact-match-test-suffix"
        matched_names = [r["name"] for r in results]
        if (
            "exact-match-test" in matched_names
            and "prefix-exact-match-test-suffix" not in matched_names
        ):
            print("  ✓ Correct exact match behavior")
        else:
            print(f"  ✗ Incorrect matches: {matched_names}")
    elif resp.status_code == 404:
        print("  ✗ No matches found (might be expected if artifacts not persisted)")
    else:
        print(f"  ✗ Unexpected status: {resp.text[:200]}")


def test_concurrent_rate():
    """Test concurrent rate requests that are failing"""
    client = TestClient(app)
    headers = get_auth_headers()

    print("\n=== Testing Concurrent Rate Requests ===")

    # Create multiple model artifacts
    model_ids = []
    for i in range(5):
        resp = client.post(
            "/artifact/model",
            json={"url": f"https://huggingface.co/test-model-{i}"},
            headers=headers,
        )
        if resp.status_code in (200, 201, 202):
            data = resp.json()
            model_id = data["metadata"]["id"]
            model_ids.append(model_id)
            print(f"✓ Created model {i}: {model_id}")
        else:
            print(f"✗ Failed to create model {i}: {resp.status_code}")

    if not model_ids:
        print("  ✗ No models created, cannot test rate")
        return

    # Test /package/{id}/rate endpoint (autograder uses this)
    def rate_model(model_id: str) -> Dict[str, Any]:
        """Rate a single model"""
        resp = client.get(f"/package/{model_id}/rate", headers=headers)
        return {
            "id": model_id,
            "status": resp.status_code,
            "data": resp.json() if resp.status_code == 200 else resp.text[:200],
        }

    # Test concurrent requests
    print(f"\nTesting concurrent rate requests for {len(model_ids)} models...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(rate_model, mid) for mid in model_ids]
        results = [f.result() for f in futures]

    success_count = sum(1 for r in results if r["status"] == 200)
    print(f"\nResults: {success_count}/{len(results)} successful")
    for r in results:
        if r["status"] == 200:
            data = r["data"]
            net_score = data.get("net_score", "N/A")
            print(f"  ✓ {r['id']}: net_score={net_score}")
        else:
            print(f"  ✗ {r['id']}: status={r['status']}, error={r['data']}")


def main():
    """Run all test scenarios"""
    print("=" * 60)
    print("Autograder Failure Reproduction Tests")
    print("=" * 60)

    # Reset system first
    client = TestClient(app)
    headers = get_auth_headers()
    reset_resp = client.delete("/reset", headers=headers)
    print(f"\nReset system: {reset_resp.status_code}")

    try:
        test_by_name_scenarios()
        test_by_id_scenarios()
        test_regex_exact_match()
        test_concurrent_rate()
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Tests completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
