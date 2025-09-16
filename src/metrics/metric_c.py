import asyncio
from src.models.types import EvalContext

def _cpu_work() -> int:
    # Simulate CPU work
    return sum(range(50_000))

async def metric(ctx: EvalContext) -> float:
    # Run CPU work off the event loop
    _ = await asyncio.to_thread(_cpu_work)
    return 0.5