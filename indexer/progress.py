import time
import threading
from enum import Enum


class SystemPhase(str, Enum):
    starting = "starting"
    crawling = "crawling"
    initial_indexing = "initial_indexing"
    watching = "watching"


class IndexingProgress:
    def __init__(self):
        self._lock = threading.Lock()
        self._phase = SystemPhase.starting
        self._files_discovered = 0
        self._files_indexed = 0
        self._files_skipped = 0
        self._files_failed = 0
        self._last_file_processed = None
        self._started_at = time.time()

    def set_phase(self, phase: SystemPhase):
        with self._lock:
            self._phase = phase

    def record_discovered(self):
        with self._lock:
            self._files_discovered += 1

    def record_indexed(self, path: str):
        with self._lock:
            self._files_indexed += 1
            self._last_file_processed = path

    def record_skipped(self, path: str):
        with self._lock:
            self._files_skipped += 1
            self._last_file_processed = path

    def record_failed(self, path: str):
        with self._lock:
            self._files_failed += 1
            self._last_file_processed = path

    def snapshot(self) -> dict:
        with self._lock:
            now = time.time()
            return {
                "phase": self._phase.value,
                "files_discovered": self._files_discovered,
                "files_processed": self._files_indexed + self._files_skipped + self._files_failed,
                "files_indexed": self._files_indexed,
                "files_skipped": self._files_skipped,
                "files_failed": self._files_failed,
                "last_file_processed": self._last_file_processed,
                "started_at": self._started_at,
                "uptime_seconds": round(now - self._started_at, 1),
            }
