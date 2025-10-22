import asyncio
import time
import logging
from typing import List

from src.metrics.registry import get_all_metrics
from src.models.model_types import MetricItem, MetricRun, OrchestrationReport, EvalContext

logger = logging.getLogger(__name__)


async def _run_one(item: MetricItem, ctx: EvalContext) -> MetricRun:
    name, fn = item
    t0 = time.perf_counter()
    try:
        logger.debug("Starting metric %s for url=%s", name, ctx.url)
        val = await fn(ctx)
        latency = int((time.perf_counter() - t0) * 1000)

        # Special handling for size_score (dict output)
        if name == "size_score" and isinstance(val, dict):
            logger.info(
                "Metric %s succeeded in %d ms (url=%s, dict_size_score)", name, latency, ctx.url
            )
            return MetricRun(name=name, value=val, latency_ms=latency)

        # Compatibility: older size metric returned best-device string
        acceptable = {"raspberry_pi", "jetson_nano", "desktop_pc", "aws_server"}
        if isinstance(val, str) and val in acceptable:
            logger.info(
                "Metric %s succeeded in %d ms (url=%s, value=%s)", name, latency, ctx.url, val
            )
            return MetricRun(name=name, value=val, latency_ms=latency)

        # All other metrics must be floats
        fval = float(val)
        logger.info(
            "Metric %s succeeded in %d ms (url=%s, value=%.2f)", name, latency, ctx.url, fval
        )
        return MetricRun(name=name, value=fval, latency_ms=latency)

    except Exception as e:
        latency = int((time.perf_counter() - t0) * 1000)
        logger.error("Metric %s failed after %d ms (url=%s): %s", name, latency, ctx.url, e)
        return MetricRun(
            name=name, value=None, latency_ms=latency, error=f"{type(e).__name__}: {e}"
        )


async def orchestrate(ctx: EvalContext, limit: int = 4) -> OrchestrationReport:
    items: List[MetricItem] = get_all_metrics()
    logger.info(
        "Starting orchestration with %d metrics (limit=%d, url=%s)", len(items), limit, ctx.url
    )
    sem = asyncio.Semaphore(limit)

    async def run_limited(it: MetricItem) -> MetricRun:
        async with sem:
            logger.debug("Acquired slot for %s (url=%s)", it[0], ctx.url)
            result = await _run_one(it, ctx)
            logger.debug(
                "Finished %s (latency=%d ms, error=%s, url=%s)",
                it[0],
                result.latency_ms,
                bool(result.error),
                ctx.url,
            )
            return result

    t0 = time.perf_counter()
    tasks = [asyncio.create_task(run_limited(i)) for i in items]  # fan out
    logger.debug("Dispatched %d tasks (url=%s)", len(tasks), ctx.url)
    results_list = await asyncio.gather(*tasks)  # fan in
    total_latency_ms = int((time.perf_counter() - t0) * 1000)
    failures = sum(1 for r in results_list if r.error)
    logger.info(
        "Orchestration finished in %d ms (success=%d, failed=%d, url=%s)",
        total_latency_ms,
        len(results_list) - failures,
        failures,
        ctx.url,
    )

    return OrchestrationReport(
        results={r.name: r for r in results_list}, total_latency_ms=total_latency_ms
    )
