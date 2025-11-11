"""
Locust performance tests for the Trustworthy Model Registry API.
Tests throughput and latency for critical endpoints.

To run:
    locust -f tests/locustfile.py --host=http://localhost:8000

Then open http://localhost:8089 in your browser to start the test.
"""

from locust import HttpUser, task, between
import json


class RegistryUser(HttpUser):
    """Simulates a typical user interacting with the registry"""
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Authenticate user on start"""
        # Use default admin credentials
        auth_payload = {
            "user": {
                "name": "ece30861defaultadminuser",
                "is_admin": True
            },
            "secret": {
                "password": "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE packages;"
            }
        }
        response = self.client.put("/authenticate", json=auth_payload)
        if response.status_code == 200:
            self.token = response.text
            self.headers = {
                "X-Authorization": self.token,
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
        else:
            self.token = None
            self.headers = {}

    @task(3)
    def get_health(self):
        """Health check endpoint - high frequency"""
        self.client.get("/health", name="GET /health")

    @task(2)
    def get_health_components(self):
        """Health components endpoint"""
        self.client.get("/health/components", headers=self.headers, name="GET /health/components")

    @task(2)
    def list_artifacts(self):
        """List artifacts - common operation"""
        if not self.token:
            return
        payload = [{"name": "*"}]
        self.client.post("/artifacts", json=payload, headers=self.headers, name="POST /artifacts")

    @task(1)
    def search_by_name(self):
        """Search artifacts by name"""
        if not self.token:
            return
        self.client.get("/artifact/byName/test", headers=self.headers, name="GET /artifact/byName")

    @task(1)
    def search_by_regex(self):
        """Search artifacts by regex"""
        if not self.token:
            return
        payload = {"regex": ".*"}
        self.client.post("/artifact/byRegEx", json=payload, headers=self.headers, name="POST /artifact/byRegEx")

    @task(1)
    def get_model_lineage(self):
        """Get model lineage graph"""
        if not self.token:
            return
        # Use a test ID - may return 404 but tests the endpoint
        self.client.get("/artifact/model/test_id/lineage", headers=self.headers, name="GET /artifact/model/{id}/lineage")

    @task(1)
    def get_artifact_cost(self):
        """Get artifact cost"""
        if not self.token:
            return
        self.client.get("/artifact/model/test_id/cost", headers=self.headers, name="GET /artifact/{type}/{id}/cost")

    @task(1)
    def license_check(self):
        """License check endpoint"""
        if not self.token:
            return
        payload = {"github_url": "https://github.com/test/repo"}
        self.client.post("/artifact/model/test_id/license-check", json=payload, headers=self.headers, name="POST /artifact/model/{id}/license-check")

    @task(1)
    def enumerate_models(self):
        """Enumerate models with pagination"""
        if not self.token:
            return
        self.client.get("/models?limit=10", headers=self.headers, name="GET /models")

    @task(1)
    def get_tracks(self):
        """Get tracks - public endpoint"""
        self.client.get("/tracks", name="GET /tracks")


class ReadOnlyUser(HttpUser):
    """Simulates read-only operations (no authentication required for some)"""
    wait_time = between(2, 5)
    weight = 2  # More read-only users than authenticated users

    @task(5)
    def get_health(self):
        """Health check - no auth required"""
        self.client.get("/health", name="GET /health (read-only)")

    @task(2)
    def get_tracks(self):
        """Get tracks - public"""
        self.client.get("/tracks", name="GET /tracks (read-only)")

