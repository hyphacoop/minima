import asyncio
import logging

from collections import deque

logger = logging.getLogger(__name__)

class AsyncQueueDequeueInterrupted(Exception):
  
    def __init__(self, message="AsyncQueue dequeue was interrupted"):
        self.message = message
        super().__init__(self.message)

class AsyncQueue:

    def __init__(self):
        self._data = deque([])
        self._presense_of_data = asyncio.Event()

    def enqueue(self, value):
        self._data.append(value)
        msg_type = value.get("type", "?") if isinstance(value, dict) else "?"
        msg_source = value.get("source", "?") if isinstance(value, dict) else "?"
        logger.info(f"Enqueued {msg_type} from {msg_source} (queue_size={len(self._data)})")

        if len(self._data) == 1:
            self._presense_of_data.set()

    async def dequeue(self):
        await self._presense_of_data.wait()

        if len(self._data) < 1:
            raise AsyncQueueDequeueInterrupted("AsyncQueue was dequeue was interrupted")

        result = self._data.popleft()
        msg_type = result.get("type", "?") if isinstance(result, dict) else "?"
        logger.info(f"Dequeued {msg_type} (queue_size={len(self._data)})")

        if not self._data:
            self._presense_of_data.clear()

        return result

    def size(self):
        result = len(self._data)
        return result

    def shutdown(self):
        self._presense_of_data.set()