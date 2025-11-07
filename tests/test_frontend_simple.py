#!/usr/bin/env python3
"""
Simplified Frontend Integration Tests
Tests backend API and frontend connectivity
"""

import time
import subprocess
import sys
import os
import signal
import requests
import json
from typing import Dict, List, Tuple

# Configuration
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:3000"
DEFAULT_USERNAME = "ece30861defaultadminuser"
DEFAULT_PASSWORD = 'correcthorsebatterystaple123(!__+@**(A\'"`;DROP TABLE packages;'

class FrontendTester:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.test_results: List[Tuple[str, bool, str]] = []
        self.token = None
        
    def wait_for_service(self, url: str, name: str, timeout: int = 30) -> bool:
        """Wait for a service to be ready"""
        print(f"⏳ Waiting for {name} to start...")
        for i in range(timeout):
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    print(f"✅ {name} is ready")
                    return True
            except:
                pass
            time.sleep(1)
        print(f"❌ {name} failed to start")
        return False
    
    def start_backend(self) -> bool:
        """Start the backend server"""
        print("🚀 Starting backend server...")
        env = os.environ.copy()
        env['USE_SQLITE'] = '1'
        env['ENVIRONMENT'] = 'development'
        env['USE_S3'] = '0'
        
        try:
            self.backend_process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return self.wait_for_service(f"{BACKEND_URL}/health", "Backend", timeout=30)
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False
    
    def start_frontend(self) -> bool:
        """Start the frontend server"""
        print("🚀 Starting frontend server...")
        frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
        env = os.environ.copy()
        env['REACT_APP_API_URL'] = BACKEND_URL
        env['BROWSER'] = 'none'  # Don't open browser automatically
        
        try:
            self.frontend_process = subprocess.Popen(
                ["npm", "start"],
                cwd=frontend_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return self.wait_for_service(FRONTEND_URL, "Frontend", timeout=60)
        except Exception as e:
            print(f"❌ Failed to start frontend: {e}")
            return False
    
    def authenticate(self) -> bool:
        """Authenticate and get token"""
        print("\n📝 Testing Authentication...")
        try:
            auth_data = {
                "user": {
                    "name": DEFAULT_USERNAME,
                    "is_admin": True
                },
                "secret": {
                    "password": DEFAULT_PASSWORD
                }
            }
            response = requests.put(f"{BACKEND_URL}/authenticate", json=auth_data, timeout=10)
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            self.token = response.text.strip('"').strip("'")
            assert len(self.token) > 0, "Token is empty"
            print(f"✅ Authentication successful (token length: {len(self.token)})")
            self.test_results.append(("Authentication", True, "Successfully authenticated"))
            return True
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            self.test_results.append(("Authentication", False, str(e)))
            return False
    
    def get_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        return {"X-Authorization": self.token} if self.token else {}
    
    def test_health_endpoint(self) -> bool:
        """Test /health endpoint"""
        print("\n📝 Testing GET /health...")
        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "status" in data or "models_count" in data
            print("✅ GET /health - OK")
            self.test_results.append(("GET /health", True, "Health endpoint working"))
            return True
        except Exception as e:
            print(f"❌ GET /health failed: {e}")
            self.test_results.append(("GET /health", False, str(e)))
            return False
    
    def test_health_components(self) -> bool:
        """Test /health/components endpoint"""
        print("\n📝 Testing GET /health/components...")
        try:
            response = requests.get(
                f"{BACKEND_URL}/health/components",
                params={"windowMinutes": 60},
                timeout=5
            )
            assert response.status_code == 200
            data = response.json()
            assert "components" in data
            print("✅ GET /health/components - OK")
            self.test_results.append(("GET /health/components", True, "Health components endpoint working"))
            return True
        except Exception as e:
            print(f"❌ GET /health/components failed: {e}")
            self.test_results.append(("GET /health/components", False, str(e)))
            return False
    
    def test_list_artifacts(self) -> bool:
        """Test POST /artifacts endpoint"""
        print("\n📝 Testing POST /artifacts...")
        try:
            response = requests.post(
                f"{BACKEND_URL}/artifacts",
                json=[{"name": "*"}],
                headers=self.get_headers(),
                timeout=10
            )
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            print(f"✅ POST /artifacts - OK (found {len(data)} artifacts)")
            self.test_results.append(("POST /artifacts", True, f"Found {len(data)} artifacts"))
            return True
        except Exception as e:
            print(f"❌ POST /artifacts failed: {e}")
            self.test_results.append(("POST /artifacts", False, str(e)))
            return False
    
    def test_create_artifact(self) -> bool:
        """Test creating an artifact"""
        print("\n📝 Testing POST /artifact/{type}...")
        try:
            artifact_data = {
                "url": "https://example.com/test-artifact-v2"
            }
            response = requests.post(
                f"{BACKEND_URL}/artifact/code",
                json=artifact_data,
                headers=self.get_headers(),
                timeout=10
            )
            assert response.status_code == 201, f"Expected 201, got {response.status_code}"
            artifact = response.json()
            assert "metadata" in artifact
            assert "data" in artifact
            assert "url" in artifact["data"]
            assert "download_url" in artifact["data"], "Missing download_url field!"
            assert artifact["data"]["download_url"] is not None, "download_url should not be None"
            print(f"✅ POST /artifact/code - OK")
            print(f"   Created: {artifact['metadata']['name']} (ID: {artifact['metadata']['id']})")
            print(f"   download_url: {artifact['data']['download_url']}")
            self.test_results.append(("POST /artifact/{type}", True, f"Created artifact {artifact['metadata']['id']}"))
            return True
        except Exception as e:
            print(f"❌ POST /artifact/{type} failed: {e}")
            self.test_results.append(("POST /artifact/{type}", False, str(e)))
            return False
    
    def test_get_artifact(self) -> bool:
        """Test retrieving an artifact"""
        print("\n📝 Testing GET /artifacts/{type}/{id}...")
        try:
            # First create an artifact
            artifact_data = {"url": "https://example.com/test-retrieve"}
            create_response = requests.post(
                f"{BACKEND_URL}/artifact/model",
                json=artifact_data,
                headers=self.get_headers(),
                timeout=10
            )
            assert create_response.status_code == 201
            created_artifact = create_response.json()
            artifact_id = created_artifact["metadata"]["id"]
            
            # Now retrieve it
            response = requests.get(
                f"{BACKEND_URL}/artifacts/model/{artifact_id}",
                headers=self.get_headers(),
                timeout=10
            )
            assert response.status_code == 200
            artifact = response.json()
            assert artifact["metadata"]["id"] == artifact_id
            assert "download_url" in artifact["data"], "Missing download_url in response!"
            print(f"✅ GET /artifacts/model/{artifact_id} - OK")
            print(f"   download_url: {artifact['data']['download_url']}")
            self.test_results.append(("GET /artifacts/{type}/{id}", True, f"Retrieved artifact {artifact_id}"))
            return True
        except Exception as e:
            print(f"❌ GET /artifacts/{type}/{id} failed: {e}")
            self.test_results.append(("GET /artifacts/{type}/{id}", False, str(e)))
            return False
    
    def test_search_by_name(self) -> bool:
        """Test searching artifacts by name"""
        print("\n📝 Testing GET /artifact/byName/{name}...")
        try:
            response = requests.get(
                f"{BACKEND_URL}/artifact/byName/test",
                headers=self.get_headers(),
                timeout=10
            )
            # 200 or 404 is acceptable
            assert response.status_code in [200, 404]
            if response.status_code == 200:
                data = response.json()
                assert isinstance(data, list)
                print(f"✅ GET /artifact/byName/test - OK (found {len(data)} results)")
            else:
                print("✅ GET /artifact/byName/test - OK (no results, expected)")
            self.test_results.append(("GET /artifact/byName/{name}", True, "Search by name working"))
            return True
        except Exception as e:
            print(f"❌ GET /artifact/byName/{name} failed: {e}")
            self.test_results.append(("GET /artifact/byName/{name}", False, str(e)))
            return False
    
    def test_search_by_regex(self) -> bool:
        """Test searching artifacts by regex"""
        print("\n📝 Testing POST /artifact/byRegEx...")
        try:
            response = requests.post(
                f"{BACKEND_URL}/artifact/byRegEx",
                json={"regex": ".*"},
                headers=self.get_headers(),
                timeout=10
            )
            # 200 or 404 is acceptable
            assert response.status_code in [200, 404]
            if response.status_code == 200:
                data = response.json()
                assert isinstance(data, list)
                print(f"✅ POST /artifact/byRegEx - OK (found {len(data)} results)")
            else:
                print("✅ POST /artifact/byRegEx - OK (no results, expected)")
            self.test_results.append(("POST /artifact/byRegEx", True, "Search by regex working"))
            return True
        except Exception as e:
            print(f"❌ POST /artifact/byRegEx failed: {e}")
            self.test_results.append(("POST /artifact/byRegEx", False, str(e)))
            return False
    
    def test_tracks_endpoint(self) -> bool:
        """Test GET /tracks endpoint"""
        print("\n📝 Testing GET /tracks...")
        try:
            response = requests.get(f"{BACKEND_URL}/tracks", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "plannedTracks" in data
            print("✅ GET /tracks - OK")
            self.test_results.append(("GET /tracks", True, "Tracks endpoint working"))
            return True
        except Exception as e:
            print(f"❌ GET /tracks failed: {e}")
            self.test_results.append(("GET /tracks", False, str(e)))
            return False
    
    def test_reset_endpoint(self) -> bool:
        """Test DELETE /reset endpoint"""
        print("\n📝 Testing DELETE /reset...")
        try:
            response = requests.delete(
                f"{BACKEND_URL}/reset",
                headers=self.get_headers(),
                timeout=10
            )
            assert response.status_code == 200
            print("✅ DELETE /reset - OK")
            self.test_results.append(("DELETE /reset", True, "Reset endpoint working"))
            return True
        except Exception as e:
            print(f"❌ DELETE /reset failed: {e}")
            self.test_results.append(("DELETE /reset", False, str(e)))
            return False
    
    def test_frontend_pages(self) -> bool:
        """Test frontend pages are accessible"""
        print("\n📝 Testing Frontend Pages...")
        pages = [
            ("/", "Home"),
            ("/login", "Login"),
            ("/dashboard", "Dashboard"),
            ("/upload", "Upload"),
            ("/search", "Search"),
            ("/download", "Download"),
            ("/health", "Health"),
        ]
        
        all_ok = True
        for path, name in pages:
            try:
                response = requests.get(f"{FRONTEND_URL}{path}", timeout=5)
                assert response.status_code == 200
                assert "html" in response.text.lower() or len(response.text) > 100
                print(f"✅ {name} page - OK")
            except Exception as e:
                print(f"❌ {name} page failed: {e}")
                all_ok = False
        
        self.test_results.append(("Frontend Pages", all_ok, f"{len(pages)} pages checked"))
        return all_ok
    
    def run_all_tests(self) -> bool:
        """Run all tests"""
        print("=" * 60)
        print("🧪 Starting Frontend Integration Tests")
        print("=" * 60)
        
        # Start servers
        backend_ok = self.start_backend()
        frontend_ok = self.start_frontend()
        
        if not backend_ok:
            print("❌ Backend failed to start - aborting tests")
            return False
        
        if not frontend_ok:
            print("⚠️ Frontend failed to start - continuing with API tests only")
        
        try:
            # Run API tests
            self.test_health_endpoint()
            self.test_health_components()
            self.authenticate()
            
            if self.token:
                self.test_list_artifacts()
                self.test_create_artifact()
                self.test_get_artifact()
                self.test_search_by_name()
                self.test_search_by_regex()
                self.test_reset_endpoint()
            
            self.test_tracks_endpoint()
            
            if frontend_ok:
                self.test_frontend_pages()
            
            # Print summary
            print("\n" + "=" * 60)
            print("📊 Test Summary")
            print("=" * 60)
            
            passed = sum(1 for _, result, _ in self.test_results if result)
            total = len(self.test_results)
            
            for test_name, result, message in self.test_results:
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{status} - {test_name}: {message}")
            
            print(f"\nTotal: {passed}/{total} tests passed ({passed*100//total if total > 0 else 0}%)")
            
            return passed == total
            
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n🧹 Cleaning up...")
        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=5)
            except:
                self.backend_process.kill()
        if self.frontend_process:
            try:
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=5)
            except:
                self.frontend_process.kill()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    tester = FrontendTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

