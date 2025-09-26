import asyncio
from src.models.types import EvalContext

def _cpu_work() -> int:
    # Simulate heavier CPU work
    return sum(i * i for i in range(30_000))

async def metric(ctx: EvalContext) -> float:
    _ = await asyncio.to_thread(_cpu_work)
    return 0.95