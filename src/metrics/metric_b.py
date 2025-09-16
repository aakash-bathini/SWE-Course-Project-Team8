import asyncio
from src.models.types import EvalContext

async def metric(ctx: EvalContext) -> float:
    # Simulate a fast I/O based metric
    await asyncio.sleep(0.08)
    return 0.75
