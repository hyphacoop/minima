import os
import uuid
import asyncio
import logging
from indexer import Indexer
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
executor = ThreadPoolExecutor()

# Path to crawl/watch: CONTAINER_PATH in Docker, LOCAL_FILES_PATH when running locally
FILES_PATH = os.environ.get("CONTAINER_PATH") or os.environ.get("LOCAL_FILES_PATH")
AVAILABLE_EXTENSIONS = [".pdf", ".xls", "xlsx", ".doc", ".docx", ".txt", ".md", ".csv", ".ppt", ".pptx"]


async def crawl_loop(async_queue):
    logger.info(f"Starting crawl loop with path: {FILES_PATH}")
    existing_file_paths: list[str] = []
    for root, _, files in os.walk(FILES_PATH):
        logger.info(f"Processing folder: {root}")
        for file in files:
            if not any(file.endswith(ext) for ext in AVAILABLE_EXTENSIONS):
                logger.info(f"Skipping file: {file}")
                continue
            path = os.path.join(root, file)
            message = {
                "path": path,
                "file_id": str(uuid.uuid4()),
                "last_updated_seconds": round(os.path.getmtime(path)),
                "type": "file",
                "source": "scheduled_crawl"
            }
            existing_file_paths.append(path)
            async_queue.enqueue(message)
            logger.info(f"File enqueue: {path}")
    # Enqueue aggregate message AFTER all files are discovered
    aggregate_message = {
        "existing_file_paths": existing_file_paths,
        "type": "all_files",
        "source": "scheduled_crawl"
    }
    async_queue.enqueue(aggregate_message)


async def index_loop(async_queue, indexer: Indexer):
    loop = asyncio.get_running_loop()
    logger.info("Starting index loop")
    while True:
        message = await async_queue.dequeue()
        source = message.get("source", "unknown")
        if message["type"] == "file":
            logger.info(f"Processing message from {source}: {message.get('path', 'unknown')}")
        else:
            logger.info(f"Processing message from {source}: {message}")
        try:
            if message["type"] == "file":
                await loop.run_in_executor(executor, indexer.index, message)
            elif message["type"] == "all_files":
                await loop.run_in_executor(executor, indexer.purge, message)
        except Exception as e:
            logger.error(f"Error in processing message: {e}")
            logger.error(f"Failed to process message: {message}")
        await asyncio.sleep(0)

