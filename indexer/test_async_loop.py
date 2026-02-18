"""
Tests for index_loop: message routing, error resilience, cancellation.
Run: cd indexer && .venv/bin/python -m pytest test_async_loop.py -v -s
"""
import asyncio
import pytest
from unittest.mock import MagicMock
from async_queue import AsyncQueue
from async_loop import index_loop
from progress import IndexingProgress


@pytest.fixture
def async_queue():
    return AsyncQueue()


@pytest.fixture
def mock_indexer():
    indexer = MagicMock()
    indexer.index = MagicMock()
    indexer.purge = MagicMock()
    return indexer


@pytest.fixture
def progress():
    return IndexingProgress()


@pytest.mark.asyncio
async def test_index_loop_routes_file_to_index(async_queue, mock_indexer, progress):
    """A 'file' message should be routed to indexer.index()."""
    message = {
        "path": "/tmp/test.txt",
        "file_id": "abc-123",
        "last_updated_seconds": 1000,
        "type": "file",
        "source": "file_watcher",
    }
    async_queue.enqueue(message)

    loop_task = asyncio.create_task(index_loop(async_queue, mock_indexer, progress))
    # Give the loop time to process the message
    for _ in range(50):
        await asyncio.sleep(0.05)
        if mock_indexer.index.called:
            break

    loop_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        pass

    mock_indexer.index.assert_called_once()
    call_arg = mock_indexer.index.call_args[0][0]
    assert call_arg["path"] == "/tmp/test.txt"
    assert call_arg["type"] == "file"


@pytest.mark.asyncio
async def test_index_loop_routes_all_files_to_purge(async_queue, mock_indexer, progress):
    """An 'all_files' message should be routed to indexer.purge()."""
    message = {
        "existing_file_paths": ["/tmp/a.txt", "/tmp/b.txt"],
        "type": "all_files",
        "source": "file_watcher",
    }
    async_queue.enqueue(message)

    loop_task = asyncio.create_task(index_loop(async_queue, mock_indexer, progress))
    for _ in range(50):
        await asyncio.sleep(0.05)
        if mock_indexer.purge.called:
            break

    loop_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        pass

    mock_indexer.purge.assert_called_once()
    call_arg = mock_indexer.purge.call_args[0][0]
    assert call_arg["type"] == "all_files"
    assert call_arg["existing_file_paths"] == ["/tmp/a.txt", "/tmp/b.txt"]


@pytest.mark.asyncio
async def test_index_loop_survives_index_error(async_queue, mock_indexer, progress):
    """If indexer.index raises, the loop should continue processing the next message."""
    # First call raises, second call succeeds
    mock_indexer.index.side_effect = [ValueError("bad file"), None]

    bad_message = {
        "path": "/tmp/bad.txt",
        "file_id": "bad-1",
        "last_updated_seconds": 1000,
        "type": "file",
        "source": "test",
    }
    good_message = {
        "path": "/tmp/good.txt",
        "file_id": "good-1",
        "last_updated_seconds": 2000,
        "type": "file",
        "source": "test",
    }
    async_queue.enqueue(bad_message)
    async_queue.enqueue(good_message)

    loop_task = asyncio.create_task(index_loop(async_queue, mock_indexer, progress))
    for _ in range(100):
        await asyncio.sleep(0.05)
        if mock_indexer.index.call_count >= 2:
            break

    loop_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        pass

    # Both messages were attempted — loop survived the first error
    assert mock_indexer.index.call_count == 2
    second_call_arg = mock_indexer.index.call_args_list[1][0][0]
    assert second_call_arg["path"] == "/tmp/good.txt"


@pytest.mark.asyncio
async def test_index_loop_cancellation(async_queue, mock_indexer, progress):
    """Cancelling the index_loop task should raise CancelledError cleanly."""
    loop_task = asyncio.create_task(index_loop(async_queue, mock_indexer, progress))
    await asyncio.sleep(0.1)  # let the loop start waiting

    loop_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await loop_task
