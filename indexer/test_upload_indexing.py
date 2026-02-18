"""
Reproduces the on-upload indexing pipeline locally.
Run: cd indexer && python -m pytest test_upload_indexing.py -v -s
"""
import asyncio
import os
import tempfile
import pytest
from unittest.mock import MagicMock
from async_queue import AsyncQueue
from async_loop import index_loop

# Use a fast poll interval for tests (PollingObserver)
os.environ.setdefault("FILE_WATCHER_POLL_INTERVAL", "1")
from file_watcher import FileWatcher


@pytest.fixture
def tmp_watch_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def async_queue():
    return AsyncQueue()


@pytest.fixture
def mock_indexer():
    indexer = MagicMock()
    indexer.index = MagicMock()   # sync method called via run_in_executor
    indexer.purge = MagicMock()
    return indexer


@pytest.mark.asyncio
async def test_uploaded_file_gets_indexed(tmp_watch_dir, async_queue, mock_indexer):
    """
    Simulates a file upload after index_loop is already running.
    Asserts the file is picked up by index_loop within 5 seconds.
    """
    # Start index_loop as a background task
    loop_task = asyncio.create_task(index_loop(async_queue, mock_indexer))

    # Start file watcher
    watcher = FileWatcher(async_queue, tmp_watch_dir)
    watcher.start()

    # Give the watcher time to initialize
    await asyncio.sleep(0.5)

    # Simulate a file upload — write a .txt file
    test_file = os.path.join(tmp_watch_dir, "upload_test.txt")
    with open(test_file, "w") as f:
        f.write("test content for indexing")

    # Wait for the pipeline: poll interval (1s) + debounce (2s) + processing
    indexed = False
    for _ in range(100):  # 10 seconds total (100 * 0.1s)
        await asyncio.sleep(0.1)
        if mock_indexer.index.called:
            indexed = True
            break

    # Clean up
    watcher.stop()
    loop_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        pass

    # Assert
    assert indexed, "index_loop never called indexer.index() — file upload was not processed"
    call_args = mock_indexer.index.call_args[0][0]
    assert os.path.realpath(call_args["path"]) == os.path.realpath(test_file)
    assert call_args["source"] == "file_watcher"
    assert call_args["type"] == "file"
