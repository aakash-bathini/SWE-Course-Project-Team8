import asyncio
from src.models.types import EvalContext

async def metric(ctx: EvalContext) -> float:
    # Simulate a slower I/O based metric
    await asyncio.sleep(0.90)
    return 1.0
