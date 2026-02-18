"""
Tests for IndexingProgress: counters, phase transitions, snapshot consistency, thread safety.
Run: cd indexer && python -m pytest test_progress.py -v
"""
import threading
import pytest
from progress import IndexingProgress, SystemPhase


@pytest.fixture
def prog():
    return IndexingProgress()


def test_initial_state(prog):
    snap = prog.snapshot()
    assert snap["phase"] == "starting"
    assert snap["files_discovered"] == 0
    assert snap["files_processed"] == 0
    assert snap["files_indexed"] == 0
    assert snap["files_skipped"] == 0
    assert snap["files_failed"] == 0
    assert snap["last_file_processed"] is None
    assert snap["started_at"] > 0
    assert snap["uptime_seconds"] >= 0


def test_record_discovered(prog):
    prog.record_discovered()
    prog.record_discovered()
    assert prog.snapshot()["files_discovered"] == 2


def test_record_indexed(prog):
    prog.record_indexed("/data/a.pdf")
    snap = prog.snapshot()
    assert snap["files_indexed"] == 1
    assert snap["files_processed"] == 1
    assert snap["last_file_processed"] == "/data/a.pdf"


def test_record_skipped(prog):
    prog.record_skipped("/data/b.pdf")
    snap = prog.snapshot()
    assert snap["files_skipped"] == 1
    assert snap["files_processed"] == 1
    assert snap["last_file_processed"] == "/data/b.pdf"


def test_record_failed(prog):
    prog.record_failed("/data/c.pdf")
    snap = prog.snapshot()
    assert snap["files_failed"] == 1
    assert snap["files_processed"] == 1
    assert snap["last_file_processed"] == "/data/c.pdf"


def test_files_processed_is_sum(prog):
    prog.record_indexed("/a")
    prog.record_indexed("/b")
    prog.record_skipped("/c")
    prog.record_failed("/d")
    assert prog.snapshot()["files_processed"] == 4


def test_set_phase(prog):
    prog.set_phase(SystemPhase.crawling)
    assert prog.snapshot()["phase"] == "crawling"
    prog.set_phase(SystemPhase.initial_indexing)
    assert prog.snapshot()["phase"] == "initial_indexing"
    prog.set_phase(SystemPhase.watching)
    assert prog.snapshot()["phase"] == "watching"


def test_snapshot_returns_dict_copy(prog):
    """Mutating the returned dict must not affect internal state."""
    snap = prog.snapshot()
    snap["files_indexed"] = 9999
    assert prog.snapshot()["files_indexed"] == 0


def test_thread_safety(prog):
    """10 threads x 1000 increments each = 10000 total discovered."""
    barrier = threading.Barrier(10)

    def worker():
        barrier.wait()
        for _ in range(1000):
            prog.record_discovered()

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert prog.snapshot()["files_discovered"] == 10000
