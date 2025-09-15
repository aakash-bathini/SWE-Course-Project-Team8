import os
import sys
import asyncio

from orchestration.logging_util import setup_logging
from src.api.prep_eval_context import prepare_eval_context
from src.orchestration.metric_orchestrator import orchestrate
from src.scoring.weights import get_weights
from src.scoring.net_score import bundle_from_report

async def _run(urls: list[str]) -> None:
    # write debug logs to src/temp.log for now
    log_file = os.path.join(os.path.dirname(__file__), "temp.log")
    setup_logging(level=2, log_file=log_file, also_stderr=False)

    for url in urls:
        ctx = prepare_eval_context(url)
        report = await orchestrate(ctx, limit=4)
        bundle = bundle_from_report(report, get_weights())

        print("== Pipeline Summary ==")
        print(f"url: {ctx.url}  category: {ctx.category}")
        print(f"overall_orchestration_latency_ms: {report.total_latency_ms}")
        print(f"net_score: {bundle.net_score:.4f}  (net_score_latency_ms: {bundle.net_score_latency_ms})")
        print("subscores:")
        for k, v in bundle.subscores.items():
            print(f"  - {k}: {v:.4f}")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m src.main <URL> [URL ...]", file=sys.stderr)
        sys.exit(1)
    asyncio.run(_run(sys.argv[1:]))
