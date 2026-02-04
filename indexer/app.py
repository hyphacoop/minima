import os
import nltk
import logging
import asyncio
from indexer import Indexer
from pydantic import BaseModel
from storage import MinimaStore
from async_queue import AsyncQueue
from fastapi import FastAPI, APIRouter
from contextlib import asynccontextmanager
from fastapi_utilities import repeat_every
from async_loop import index_loop, crawl_loop
from file_watcher import FileWatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to crawl/watch: CONTAINER_PATH in Docker, LOCAL_FILES_PATH when running locally
FILES_PATH = os.environ.get("CONTAINER_PATH") or os.environ.get("LOCAL_FILES_PATH")
ENABLE_FILE_WATCHER = os.environ.get("ENABLE_FILE_WATCHER", "true").lower() == "true"

indexer = Indexer()
router = APIRouter()
async_queue = AsyncQueue()
MinimaStore.create_db_and_tables()

def init_loader_dependencies():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger_eng')

init_loader_dependencies()

class Query(BaseModel):
    query: str


@router.post(
    "/query", 
    response_description='Query local data storage',
)
async def query(request: Query):
    logger.info(f"Received query: {request.query}")
    try:
        result = indexer.find(request.query)
        logger.info(f"Found {len(result.get('chunks', []))} chunks for query: {request.query}")
        return {"result": result}
    except Exception as e:
        logger.error(f"Error in processing query: {e}")
        return {"error": str(e)}


@router.post(
    "/embedding", 
    response_description='Get embedding for a query',
)
async def embedding(request: Query):
    logger.info(f"Received embedding request: {request}")
    try:
        result = indexer.embed(request.query)
        logger.info(f"Found {len(result)} results for query: {request.query}")
        return {"result": result}
    except Exception as e:
        logger.error(f"Error in processing embedding: {e}")
        return {"error": str(e)}    


async def watchdog_health_check(file_watcher: FileWatcher):
    """Monitor file watcher health and restart if needed"""
    retry_count = 0
    max_retries = 5

    while True:
        await asyncio.sleep(30)  # Check every 30 seconds

        if not file_watcher.is_alive():
            logger.error(f"File watcher died! Attempting restart (attempt {retry_count + 1}/{max_retries})")

            if retry_count < max_retries:
                delay = 2 ** retry_count  # Exponential backoff
                await asyncio.sleep(delay)

                try:
                    file_watcher.stop()
                    file_watcher.start()
                    retry_count = 0  # Reset on success
                    logger.info("File watcher successfully restarted")
                except Exception as e:
                    logger.error(f"Failed to restart watcher: {e}")
                    retry_count += 1
            else:
                logger.critical("File watcher failed to restart after max retries. Manual intervention required.")
                break
        else:
            retry_count = 0  # Reset counter if watcher is healthy


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not FILES_PATH:
        raise ValueError("Set CONTAINER_PATH or LOCAL_FILES_PATH in environment")

    # Run initial crawl first
    await crawl_loop(async_queue)
    logger.info("Initial crawl complete")

    # Start file watcher for incremental updates
    file_watcher = None
    if ENABLE_FILE_WATCHER:
        try:
            file_watcher = FileWatcher(async_queue, FILES_PATH)
            file_watcher.start()
            logger.info(f"File watcher started monitoring: {FILES_PATH}")
        except Exception as e:
            logger.error(f"Failed to start watcher: {e}. Using scheduled crawl only.")
    else:
        logger.info("File watcher disabled via environment variable")

    # Start existing loops + health check
    tasks = [
        asyncio.create_task(index_loop(async_queue, indexer))
    ]

    if file_watcher is not None:
        tasks.append(asyncio.create_task(watchdog_health_check(file_watcher)))

    # Start scheduled backup polling
    await schedule_reindexing()

    try:
        yield
    finally:
        # Stop file watcher first
        if file_watcher:
            logger.info("Stopping file watcher...")
            file_watcher.stop()

        # Cancel async tasks
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        openapi_url="/indexer/openapi.json",
        docs_url="/indexer/docs",
        lifespan=lifespan
    )
    app.include_router(router)
    return app

async def trigger_re_indexer():
    logger.info("Reindexing triggered")
    try:
        await asyncio.gather(
            crawl_loop(async_queue),
            index_loop(async_queue, indexer)
        )
        logger.info("reindexing finished")
    except Exception as e:
        logger.error(f"error in scheduled reindexing {e}")


@repeat_every(seconds=60*20)
async def schedule_reindexing():
    await trigger_re_indexer()

app = create_app()