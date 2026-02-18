import os
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from async_queue import AsyncQueue
from singleton import Singleton
from storage import MinimaStore

logger = logging.getLogger(__name__)

POLLING_INTERVAL = int(os.environ.get("FILE_WATCHER_POLL_INTERVAL", "5"))
AVAILABLE_EXTENSIONS = [".pdf", ".xls", ".xlsx", ".doc", ".docx", ".txt", ".md", ".csv", ".ppt", ".pptx"]
DEBOUNCE_SECONDS = 2.0
TEMPORARY_PATTERNS = [".tmp", ".swp", "~"]


class Debouncer:
    """Debounces file system events to handle rapid changes"""

    def __init__(self, delay: float = DEBOUNCE_SECONDS):
        self.delay = delay
        self.pending_tasks: Dict[str, asyncio.Task] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

    async def debounce(self, path: str, callback, *args):
        """
        Debounce a callback for a specific file path.
        Cancels any pending task for the same path and schedules a new one.
        """
        # Cancel existing task for this path
        if path in self.pending_tasks:
            self.pending_tasks[path].cancel()

        # Schedule new task
        async def delayed_callback():
            await asyncio.sleep(self.delay)
            try:
                await callback(*args)
            finally:
                # Clean up completed task
                if path in self.pending_tasks:
                    del self.pending_tasks[path]

        self.pending_tasks[path] = asyncio.create_task(delayed_callback())

    async def start_cleanup_loop(self):
        """Periodically clean up completed tasks"""
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            completed = [k for k, v in self.pending_tasks.items() if v.done()]
            for key in completed:
                del self.pending_tasks[key]
            if completed:
                logger.debug(f"Cleaned up {len(completed)} completed debounce tasks")

    def cancel_all(self):
        """Cancel all pending tasks"""
        for task in self.pending_tasks.values():
            task.cancel()
        self.pending_tasks.clear()


class FileWatcherHandler(FileSystemEventHandler):
    """Handles file system events and enqueues them for indexing"""

    def __init__(self, async_queue: AsyncQueue, debouncer: Debouncer, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.async_queue = async_queue
        self.debouncer = debouncer
        self.loop = loop

    def _should_process(self, path: str, is_directory: bool) -> bool:
        """Check if a file should be processed based on extension and patterns"""
        if is_directory:
            return False

        filename = Path(path).name

        # Skip hidden files
        if filename.startswith('.'):
            logger.debug(f"Skipping hidden file: {filename}")
            return False

        # Skip temporary files
        for pattern in TEMPORARY_PATTERNS:
            if filename.endswith(pattern):
                logger.debug(f"Skipping temporary file: {filename}")
                return False

        # Check supported extensions
        if not any(path.endswith(ext) for ext in AVAILABLE_EXTENSIONS):
            logger.debug(f"Skipping unsupported file type: {filename}")
            return False

        return True

    def _enqueue_file(self, path: str):
        """Enqueue a file for indexing"""
        if not os.path.exists(path):
            logger.warning(f"File no longer exists, skipping: {path}")
            return

        # Check queue size for backpressure
        if self.async_queue.size() > 1000:
            logger.warning(f"Queue size exceeded 1000, skipping non-critical file: {path}")
            return

        try:
            message = {
                "path": path,
                "file_id": str(uuid.uuid4()),
                "last_updated_seconds": round(os.path.getmtime(path)),
                "type": "file",
                "source": "file_watcher"
            }
            self.async_queue.enqueue(message)
            logger.info(f"File enqueued from watcher: {path}")
        except Exception as e:
            logger.error(f"Error enqueueing file {path}: {e}")

    def _enqueue_delete(self, path: str):
        """Enqueue a delete event (purge)"""
        try:
            # Get all currently indexed paths except the deleted one
            existing_paths = MinimaStore.select_all_indexed_paths()
            existing_paths = [p for p in existing_paths if p != path]

            message = {
                "existing_file_paths": existing_paths,
                "type": "all_files",
                "source": "file_watcher"
            }
            self.async_queue.enqueue(message)
            logger.info(f"Delete event enqueued for: {path}")
        except Exception as e:
            logger.error(f"Error enqueueing delete for {path}: {e}")

    def on_created(self, event: FileSystemEvent):
        """Handle file creation events"""
        if not self._should_process(event.src_path, event.is_directory):
            return

        logger.info(f"File created: {event.src_path}")

        # Use thread-safe call to enqueue with debounce
        path = event.src_path

        def schedule_debounce():
            asyncio.create_task(self.debouncer.debounce(path, self._async_enqueue_file, path))

        self.loop.call_soon_threadsafe(schedule_debounce)

    def on_modified(self, event: FileSystemEvent):
        """Handle file modification events"""
        if not self._should_process(event.src_path, event.is_directory):
            return

        logger.info(f"File modified: {event.src_path}")

        # Use thread-safe call to enqueue with debounce
        path = event.src_path

        def schedule_debounce():
            asyncio.create_task(self.debouncer.debounce(path, self._async_enqueue_file, path))

        self.loop.call_soon_threadsafe(schedule_debounce)

    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion events (no debounce)"""
        if event.is_directory:
            return

        logger.debug(f"File deleted: {event.src_path}")

        # Delete events are not debounced - immediate processing
        self.loop.call_soon_threadsafe(self._enqueue_delete, event.src_path)

    def on_moved(self, event: FileSystemEvent):
        """Handle file move/rename events as delete + create"""
        if event.is_directory:
            return

        logger.debug(f"File moved: {event.src_path} -> {event.dest_path}")

        # Treat as delete of old path
        self.loop.call_soon_threadsafe(self._enqueue_delete, event.src_path)

        # Treat as create of new path (with debounce)
        if self._should_process(event.dest_path, False):
            dest_path = event.dest_path

            def schedule_debounce():
                asyncio.create_task(self.debouncer.debounce(dest_path, self._async_enqueue_file, dest_path))

            self.loop.call_soon_threadsafe(schedule_debounce)

    async def _async_enqueue_file(self, path: str):
        """Async wrapper for enqueueing files"""
        self._enqueue_file(path)


class FileWatcher(metaclass=Singleton):
    """Main file watcher orchestrator"""

    def __init__(self, async_queue: AsyncQueue, watch_path: str):
        self.async_queue = async_queue
        self.watch_path = watch_path
        self.observer: Optional[PollingObserver] = None
        self.handler: Optional[FileWatcherHandler] = None
        self.debouncer = Debouncer(DEBOUNCE_SECONDS)
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self):
        """Start the file watcher"""
        if self.observer and self.observer.is_alive():
            logger.warning("File watcher already running")
            return

        try:
            # Get the current event loop
            self.loop = asyncio.get_event_loop()

            # Create handler and observer
            self.handler = FileWatcherHandler(self.async_queue, self.debouncer, self.loop)
            self.observer = PollingObserver(timeout=POLLING_INTERVAL)
            self.observer.schedule(self.handler, self.watch_path, recursive=True)

            # Start observer
            self.observer.start()
            logger.info(f"File watcher started monitoring: {self.watch_path} (recursive)")

            # Start cleanup loop
            asyncio.create_task(self.debouncer.start_cleanup_loop())

        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")
            raise

    def stop(self):
        """Stop the file watcher"""
        if self.observer:
            logger.info("Stopping file watcher...")
            self.observer.stop()
            self.observer.join(timeout=5)
            self.debouncer.cancel_all()
            logger.info("File watcher stopped")

    def is_alive(self) -> bool:
        """Check if the observer is running"""
        return self.observer is not None and self.observer.is_alive()
