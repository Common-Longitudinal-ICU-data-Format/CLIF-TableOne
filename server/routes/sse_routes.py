"""SSE progress streaming routes."""

import json
import asyncio

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from server import session

router = APIRouter(prefix="/api", tags=["sse"])


@router.get("/sse/progress/{task_id}")
async def progress_stream(task_id: str):
    """Stream task progress via Server-Sent Events."""

    async def event_generator():
        last_msg = None
        while True:
            tasks = session.get("tasks") or {}
            task = tasks.get(task_id)

            if task is None:
                yield f"data: {json.dumps({'status': 'error', 'message': 'Task not found'})}\n\n"
                break

            msg = json.dumps(task, default=str)
            if msg != last_msg:
                yield f"data: {msg}\n\n"
                last_msg = msg

            if task.get("status") in ("complete", "error"):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
