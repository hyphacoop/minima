"""
Tests for FileWatcher: filtering, event handling, debounce.
Run: cd indexer && .venv/bin/python -m pytest test_file_watcher.py -v -s
"""
import asyncio
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from async_queue import AsyncQueue
from singleton import Singleton

# Use a fast poll interval for tests
os.environ["FILE_WATCHER_POLL_INTERVAL"] = "1"
from file_watcher import FileWatcher, FileWatcherHandler, Debouncer, AVAILABLE_EXTENSIONS
from progress import IndexingProgress


@pytest.fixture(autouse=True)
def clear_singletons():
    """Clear Singleton instances between tests to avoid stale state."""
    Singleton._instances.clear()
    yield
    Singleton._instances.clear()


@pytest.fixture
def tmp_watch_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def async_queue():
    return AsyncQueue()


# ── Filtering tests (unit tests on _should_process) ──


class TestShouldProcess:
    def setup_method(self):
        handler = FileWatcherHandler.__new__(FileWatcherHandler)
        handler.async_queue = None
        handler.debouncer = None
        handler.loop = None
        self.handler = handler

    def test_skip_directories(self):
        assert self.handler._should_process("/tmp/somedir", is_directory=True) is False

    def test_skip_hidden_files(self):
        assert self.handler._should_process("/tmp/.hidden.txt", is_directory=False) is False

    def test_skip_temporary_tmp(self):
        assert self.handler._should_process("/tmp/doc.tmp", is_directory=False) is False

    def test_skip_temporary_swp(self):
        assert self.handler._should_process("/tmp/file.swp", is_directory=False) is False

    def test_skip_temporary_tilde(self):
        assert self.handler._should_process("/tmp/notes.txt~", is_directory=False) is False

    def test_skip_unsupported_jpg(self):
        assert self.handler._should_process("/tmp/photo.jpg", is_directory=False) is False

    def test_skip_unsupported_py(self):
        assert self.handler._should_process("/tmp/script.py", is_directory=False) is False

    def test_supported_pdf(self):
        assert self.handler._should_process("/tmp/report.pdf", is_directory=False) is True

    def test_supported_txt(self):
        assert self.handler._should_process("/tmp/notes.txt", is_directory=False) is True

    def test_supported_csv(self):
        assert self.handler._should_process("/tmp/data.csv", is_directory=False) is True

    def test_supported_docx(self):
        assert self.handler._should_process("/tmp/report.docx", is_directory=False) is True


# ── Integration tests through PollingObserver ──


async def _wait_for_queue(async_queue, min_count=1, timeout=8.0):
    """Poll the queue until it has at least min_count items or timeout."""
    elapsed = 0
    while elapsed < timeout:
        if async_queue.size() >= min_count:
            break
        await asyncio.sleep(0.2)
        elapsed += 0.2
    messages = []
    while async_queue.size() > 0:
        messages.append(await async_queue.dequeue())
    return messages


@pytest.mark.asyncio
async def test_skip_hidden_files(async_queue, tmp_watch_dir):
    """Hidden files should not be enqueued."""
    watcher = FileWatcher(async_queue, tmp_watch_dir)
    watcher.start()
    try:
        await asyncio.sleep(0.5)
        with open(os.path.join(tmp_watch_dir, ".hidden.txt"), "w") as f:
            f.write("hidden")
        # Wait for poll (1s) + debounce (2s) + buffer
        await asyncio.sleep(4)
        assert async_queue.size() == 0
    finally:
        watcher.stop()


@pytest.mark.asyncio
async def test_skip_temporary_files(async_queue, tmp_watch_dir):
    """Temporary files (.tmp, .swp, ~) should not be enqueued."""
    watcher = FileWatcher(async_queue, tmp_watch_dir)
    watcher.start()
    try:
        await asyncio.sleep(0.5)
        for name in ["doc.tmp", "file.swp", "notes.txt~"]:
            with open(os.path.join(tmp_watch_dir, name), "w") as f:
                f.write("temp")
        await asyncio.sleep(4)
        assert async_queue.size() == 0
    finally:
        watcher.stop()


@pytest.mark.asyncio
async def test_skip_unsupported_extensions(async_queue, tmp_watch_dir):
    """Unsupported file types should not be enqueued."""
    watcher = FileWatcher(async_queue, tmp_watch_dir)
    watcher.start()
    try:
        await asyncio.sleep(0.5)
        for name in ["photo.jpg", "script.py"]:
            with open(os.path.join(tmp_watch_dir, name), "w") as f:
                f.write("data")
        await asyncio.sleep(4)
        assert async_queue.size() == 0
    finally:
        watcher.stop()


@pytest.mark.asyncio
async def test_supported_extensions(async_queue, tmp_watch_dir):
    """Supported files should be enqueued with correct paths and source."""
    watcher = FileWatcher(async_queue, tmp_watch_dir)
    watcher.start()
    try:
        await asyncio.sleep(0.5)
        files = ["report.pdf", "notes.txt", "data.csv"]
        for name in files:
            path = os.path.join(tmp_watch_dir, name)
            with open(path, "w") as f:
                f.write("content")

        messages = await _wait_for_queue(async_queue, min_count=3, timeout=8)
        enqueued_basenames = sorted(os.path.basename(m["path"]) for m in messages if m["type"] == "file")
        assert enqueued_basenames == sorted(files)
        for m in messages:
            if m["type"] == "file":
                assert m["source"] == "file_watcher"
                assert "file_id" in m
    finally:
        watcher.stop()


@pytest.mark.asyncio
async def test_debounce_rapid_writes(async_queue, tmp_watch_dir):
    """Rapid writes to the same file should result in a single enqueue."""
    watcher = FileWatcher(async_queue, tmp_watch_dir)
    watcher.start()
    try:
        await asyncio.sleep(0.5)
        test_file = os.path.join(tmp_watch_dir, "rapid.txt")
        for i in range(5):
            with open(test_file, "w") as f:
                f.write(f"version {i}")
            await asyncio.sleep(0.1)

        messages = await _wait_for_queue(async_queue, min_count=1, timeout=8)
        file_messages = [m for m in messages if m["type"] == "file"]
        # Debounce should collapse the rapid writes into one enqueue
        assert len(file_messages) == 1
        assert os.path.realpath(file_messages[0]["path"]) == os.path.realpath(test_file)
    finally:
        watcher.stop()


@pytest.mark.asyncio
async def test_delete_enqueues_purge(async_queue, tmp_watch_dir):
    """Deleting a file should enqueue an all_files purge message."""
    watcher = FileWatcher(async_queue, tmp_watch_dir)
    watcher.start()
    try:
        await asyncio.sleep(0.5)
        test_file = os.path.join(tmp_watch_dir, "doc.txt")
        with open(test_file, "w") as f:
            f.write("content")

        # Wait for the create event to be processed
        await _wait_for_queue(async_queue, min_count=1, timeout=8)

        # Now delete the file and mock MinimaStore.select_all_indexed_paths
        with patch("file_watcher.MinimaStore") as mock_store:
            mock_store.select_all_indexed_paths.return_value = [test_file]
            os.remove(test_file)
            # Wait for poll + processing
            await asyncio.sleep(3)

        messages = []
        while async_queue.size() > 0:
            messages.append(await async_queue.dequeue())

        purge_messages = [m for m in messages if m["type"] == "all_files"]
        assert len(purge_messages) >= 1
        assert purge_messages[0]["source"] == "file_watcher"
        # The deleted file should be excluded from existing_file_paths
        assert test_file not in purge_messages[0]["existing_file_paths"]
    finally:
        watcher.stop()


# ── Progress / debounce tests ──


@pytest.mark.asyncio
async def test_debouncer_pending_count():
    """Debouncer.pending_count reflects in-flight tasks."""
    debouncer = Debouncer(delay=1.0)
    assert debouncer.pending_count == 0

    called = asyncio.Event()

    async def callback():
        called.set()

    await debouncer.debounce("file.txt", callback)
    assert debouncer.pending_count == 1

    # Wait for debounce delay + callback to finish
    await asyncio.sleep(1.5)
    assert called.is_set()
    assert debouncer.pending_count == 0


@pytest.mark.asyncio
async def test_discovered_incremented_on_file_create(async_queue, tmp_watch_dir):
    """FileWatcher with progress should increment files_discovered on file creation."""
    prog = IndexingProgress()
    watcher = FileWatcher(async_queue, tmp_watch_dir, progress=prog)
    watcher.start()
    try:
        await asyncio.sleep(0.5)
        with open(os.path.join(tmp_watch_dir, "doc.txt"), "w") as f:
            f.write("hello")

        # Wait for poll + debounce
        await _wait_for_queue(async_queue, min_count=1, timeout=8)
        assert prog.snapshot()["files_discovered"] == 1
    finally:
        watcher.stop()


@pytest.mark.asyncio
async def test_discovered_not_incremented_on_modify(async_queue, tmp_watch_dir):
    """Modifying an existing file should NOT increment files_discovered."""
    prog = IndexingProgress()

    # Create file before watcher starts (simulates pre-existing file)
    test_file = os.path.join(tmp_watch_dir, "existing.txt")
    with open(test_file, "w") as f:
        f.write("original")

    watcher = FileWatcher(async_queue, tmp_watch_dir, progress=prog)
    watcher.start()
    try:
        await asyncio.sleep(0.5)

        # Modify the existing file
        with open(test_file, "w") as f:
            f.write("modified")

        # Wait for poll + debounce
        await _wait_for_queue(async_queue, min_count=1, timeout=8)

        # Modify should not bump discovered count
        assert prog.snapshot()["files_discovered"] == 0
    finally:
        watcher.stop()
