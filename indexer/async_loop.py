import os
import uuid
import asyncio
import logging
from indexer import Indexer
from storage import IndexingStatus
from progress import IndexingProgress, SystemPhase
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
executor = ThreadPoolExecutor()

# Path to crawl/watch: CONTAINER_PATH in Docker, LOCAL_FILES_PATH when running locally
FILES_PATH = os.environ.get("CONTAINER_PATH") or os.environ.get("LOCAL_FILES_PATH")
AVAILABLE_EXTENSIONS = [".pdf", ".xls", "xlsx", ".doc", ".docx", ".txt", ".md", ".csv", ".ppt", ".pptx"]


async def crawl_loop(async_queue, progress: IndexingProgress = None):
    logger.info(f"Starting crawl loop with path: {FILES_PATH}")
    if progress:
        progress.set_phase(SystemPhase.crawling)
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
            if progress:
                progress.record_discovered()
            logger.info(f"File enqueue: {path}")
    if progress:
        progress.set_phase(SystemPhase.initial_indexing)
    # Enqueue aggregate message AFTER all files are discovered
    aggregate_message = {
        "existing_file_paths": existing_file_paths,
        "type": "all_files",
        "source": "scheduled_crawl"
    }
    async_queue.enqueue(aggregate_message)


async def index_loop(async_queue, indexer: Indexer, progress: IndexingProgress = None):
    loop = asyncio.get_running_loop()
    logger.info("Starting index loop")
    while True:
        try:
            logger.debug("Waiting for next message...")
            message = await async_queue.dequeue()
            source = message.get("source", "unknown")
            if message["type"] == "file":
                logger.info(f"Processing message from {source}: {message.get('path', 'unknown')}")
            else:
                logger.info(f"Processing message from {source}: {message}")
            if message["type"] == "file":
                result = await loop.run_in_executor(executor, indexer.index, message)
                if progress:
                    path = message["path"]
                    if result == IndexingStatus.no_need_reindexing:
                        progress.record_skipped(path)
                    elif result == IndexingStatus.failed:
                        progress.record_failed(path)
                    else:
                        progress.record_indexed(path)
            elif message["type"] == "all_files":
                await loop.run_in_executor(executor, indexer.purge, message)
            if progress and progress.snapshot()["phase"] == SystemPhase.initial_indexing.value:
                if async_queue.size() == 0:
                    progress.set_phase(SystemPhase.watching)
        except asyncio.CancelledError:
            logger.info("index_loop cancelled, shutting down")
            raise
        except Exception as e:
            logger.error(f"Error in index_loop iteration: {e}", exc_info=True)
        await asyncio.sleep(0)
