#!/usr/bin/env python3
"""
Local test script to reproduce autograder failures:
- Artifact Read Test Group (ByName/ByID)
- Regex Tests Group (Exact Match)
- Rate models concurrently

This script tests the exact scenarios that the autograder tests.
"""

import sys
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from urllib.parse import quote

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


def run_by_name_tests(client: TestClient, headers: Dict[str, str]):
    """Test ByName scenarios matching autograder test cases"""
    print("\n=== Testing ByName Scenarios ===")

    # Create artifacts with various name formats that autograder might use
    test_cases = [
        ("simple-model", "https://huggingface.co/simple-model"),
        ("org/model-name", "https://huggingface.co/org/model-name"),  # Full path
        ("model_with_underscores", "https://huggingface.co/model_with_underscores"),
        ("Model-With-Mixed-Case", "https://huggingface.co/Model-With-Mixed-Case"),
        ("model.with.dots", "https://huggingface.co/model.with.dots"),
        ("test-model-1", "https://huggingface.co/test-model-1"),
        ("test-model-2", "https://huggingface.co/test-model-2"),
    ]

    created_artifacts = {}  # name -> id mapping
    for name, url in test_cases:
        resp = client.post("/artifact/model", json={"url": url}, headers=headers)
        if resp.status_code in (200, 201, 202):
            data = resp.json()
            art_id = data["metadata"]["id"]
            stored_name = data["metadata"]["name"]
            created_artifacts[stored_name] = art_id
            print(f"✓ Created: {name} -> stored as {stored_name!r} (id: {art_id})")
        else:
            print(f"✗ Failed to create {name}: {resp.status_code} - {resp.text[:200]}")

    # Test ByName queries - test various name formats
    # Per OpenAPI spec: URL "https://huggingface.co/google-bert/bert-base-uncased"
    # stores name as "bert-base-uncased" (last segment only)
    test_queries = [
        ("simple-model", True),  # Should find
        ("model-name", True),  # Should find (stored name from org/model-name URL)
        ("Model-With-Mixed-Case", True),  # Case-insensitive
        ("MODEL_WITH_UNDERSCORES", True),  # Case-insensitive
        ("nonexistent-model-12345", False),  # Should return 404
    ]

    for query_name, should_exist in test_queries:
        # URL-encode the name (especially for slashes)
        encoded_name = quote(query_name, safe="")
        response = client.get(f"/artifact/byName/{encoded_name}", headers=headers)
        print(f"\nByName('{query_name}'):")
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Results: {len(data)} matches")
            for item in data:
                print(f"    - {item['name']} (id: {item['id']})")
            if should_exist:
                # Check if we found the expected artifact
                found = any(item["name"].lower() == query_name.lower() for item in data)
                if found:
                    print("  ✓ Correctly found artifact")
                else:
                    print(
                        f"  ✗ Expected to find '{query_name}' but got: {[i['name'] for i in data]}"
                    )
            else:
                print("  ✗ Unexpected: found artifacts when should return 404")
        elif response.status_code == 404:
            if should_exist:
                print(f"  ✗ Failed: Expected to find '{query_name}' but got 404")
            else:
                print("  ✓ Correctly returned 404 (not found)")
        else:
            print(f"  ✗ Unexpected status: {response.text[:200]}")


def run_by_id_scenarios(client: TestClient, headers: Dict[str, str]):
    """Test ByID scenarios matching autograder test cases"""
    print("\n=== Testing ByID Scenarios ===")

    # Create artifacts of different types
    artifacts_to_create = [
        ("model", "https://huggingface.co/test/model-1"),
        ("dataset", "https://huggingface.co/test/dataset-1"),
        ("code", "https://github.com/test/code-1"),
    ]

    created = {}  # type -> (id, name)
    for art_type, url in artifacts_to_create:
        resp = client.post(f"/artifact/{art_type}", json={"url": url}, headers=headers)
        if resp.status_code in (200, 201, 202):
            data = resp.json()
            art_id = data["metadata"]["id"]
            art_name = data["metadata"]["name"]
            created[art_type] = (art_id, art_name)
            print(f"✓ Created: {art_type} -> {art_name} (id: {art_id})")
        else:
            print(f"✗ Failed to create {art_type}: {resp.status_code} - {resp.text[:200]}")

    # Test /artifacts/{type}/{id} (plural route)
    for art_type, (art_id, expected_name) in created.items():
        response = client.get(f"/artifacts/{art_type}/{art_id}", headers=headers)
        print(f"\nByID({art_type}/{art_id}) via /artifacts:")
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Found: {data['metadata']['name']} (expected: {expected_name})")
        else:
            print(f"  ✗ Failed: {response.text[:200]}")

    # Test /artifact/{type}/{id} (singular route - autograder compatibility)
    for art_type, (art_id, expected_name) in created.items():
        response = client.get(f"/artifact/{art_type}/{art_id}", headers=headers)
        print(f"\nByID({art_type}/{art_id}) via /artifact:")
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Found: {data['metadata']['name']} (expected: {expected_name})")
        else:
            print(f"  ✗ Failed: {response.text[:200]}")

    # Test /package/{id} alias (autograder uses this)
    if "model" in created:
        model_id, expected_name = created["model"]
        response = client.get(f"/package/{model_id}", headers=headers)
        print(f"\nByID via /package/{model_id}:")
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Found: {data['metadata']['name']} (expected: {expected_name})")
        else:
            print(f"  ✗ Failed: {response.text[:200]}")


def run_regex_exact_match_tests(client: TestClient, headers: Dict[str, str]):
    """Test regex exact match that's failing in autograder"""
    print("\n=== Testing Regex Exact Match ===")

    # Create test artifacts with known names
    test_names = [
        "exact-match-test",
        "prefix-exact-match-test-suffix",
        "other-model",
    ]

    created_names = []
    for name in test_names:
        resp = client.post(
            "/artifact/model", json={"url": f"https://huggingface.co/{name}"}, headers=headers
        )
        if resp.status_code in (200, 201, 202):
            data = resp.json()
            stored_name = data["metadata"]["name"]
            created_names.append(stored_name)
            print(f"✓ Created: {name} -> stored as '{stored_name}'")
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
        matched_names = [r["name"] for r in results]
        # Should only match "exact-match-test", not "prefix-exact-match-test-suffix"
        if (
            "exact-match-test" in matched_names
            and "prefix-exact-match-test-suffix" not in matched_names
        ):
            print("  ✓ Correct exact match behavior")
        else:
            print(f"  ✗ Incorrect: Expected only 'exact-match-test', got: {matched_names}")
    elif resp.status_code == 404:
        print("  ✗ No matches found - this might indicate a problem")
    else:
        print(f"  ✗ Unexpected status: {resp.text[:200]}")


def run_concurrent_rate_tests(client: TestClient, headers: Dict[str, str]):
    """Test concurrent rate requests matching autograder"""
    print("\n=== Testing Concurrent Rate Requests ===")

    # Create 10 models (autograder tests 10/11)
    model_ids = []
    for i in range(10):
        resp = client.post(
            "/artifact/model",
            json={"url": f"https://huggingface.co/test/model-{i}"},
            headers=headers,
        )
        if resp.status_code in (200, 201, 202):
            data = resp.json()
            model_id = data["metadata"]["id"]
            model_ids.append(model_id)
            print(f"✓ Created model {i}: {model_id}")
        else:
            print(f"✗ Failed to create model {i}: {resp.status_code} - {resp.text[:200]}")

    if not model_ids:
        print("  ✗ No models created, cannot test rate")
        return

    # Test /package/{id}/rate endpoint (autograder uses this)
    def rate_model(model_id: str, idx: int) -> Dict[str, Any]:
        """Rate a single model"""
        try:
            resp = client.get(f"/package/{model_id}/rate", headers=headers)
            return {
                "id": model_id,
                "idx": idx,
                "status": resp.status_code,
                "data": resp.json() if resp.status_code == 200 else resp.text[:200],
            }
        except Exception as e:
            return {
                "id": model_id,
                "idx": idx,
                "status": 0,
                "data": f"Exception: {str(e)}",
            }

    # Test concurrent requests (autograder tests concurrent)
    print(f"\nTesting concurrent rate requests for {len(model_ids)} models...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(rate_model, mid, i) for i, mid in enumerate(model_ids)]
        results = [f.result() for f in futures]

    success_count = sum(1 for r in results if r["status"] == 200)
    print(f"\nResults: {success_count}/{len(results)} successful")

    if success_count == len(results):
        print("  ✓ All concurrent rate requests succeeded!")
        for r in results:
            if r["status"] == 200:
                data = r["data"]
                net_score = data.get("net_score", "N/A")
                print(f"    ✓ model-{r['idx']}: net_score={net_score}")
    else:
        print("  ✗ Some requests failed:")
        for r in results:
            if r["status"] == 200:
                data = r["data"]
                net_score = data.get("net_score", "N/A")
                print(f"    ✓ model-{r['idx']}: net_score={net_score}")
            else:
                print(
                    f"    ✗ model-{r['idx']} ({r['id']}): status={r['status']}, error={r['data']}"
                )


def main():
    """Run all test scenarios"""
    print("=" * 60)
    print("Autograder Failure Reproduction Tests")
    print("=" * 60)

    # Disable S3 and SQLite for local testing (use in-memory)
    os.environ["USE_S3"] = "0"
    os.environ["USE_SQLITE"] = "0"
    os.environ["X_ASYNC_INGEST"] = "0"

    # Reset system first
    client = TestClient(app)
    headers = get_auth_headers()
    reset_resp = client.delete("/reset", headers=headers)
    print(f"\nReset system: {reset_resp.status_code}")

    try:
        run_by_name_tests(client, headers)
        run_by_id_scenarios(client, headers)
        run_regex_exact_match_tests(client, headers)
        run_concurrent_rate_tests(client, headers)
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Tests completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
