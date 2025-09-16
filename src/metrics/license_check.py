import time
from config-parser-nlp.readme_parser import extract_license_evidence
from config-parser-nlp.spdx import parse_spdx_expression, SPDXParseError


# Assuming that the metric framework provides a base Metric class
# and a decorator to register metrics
class LicenseMetric(Metric):
    def __init__(self, repo_analyzer, gh_client, logger):
        self.repo = repo_analyzer
        self.gh = gh_client
        self.log = logger

    def compute(self, record) -> MetricResult
        start = time.time()

        # 1. Get license evidence
        readme_text = self.gh.get_readme_text(self.repo) # change function name as needed
        license_file_text = self.repo.get_license_file_text() # change function name as needed
        source, spdx_ids, spdx_exprs, hints = extract_license_evidence(readme_text, license_file_text)

        # 2. Classify based on evidence
        score, rationale = spdx.classify_license(spdx_ids, spdx_exprs, hints)

        # 3. Wrap in metricResult
        latency_ms = (time.time() - start) * 1000
        return MetricResult(score=score, rationale=f"{src}: {rationale}", latency_ms=latency_ms)