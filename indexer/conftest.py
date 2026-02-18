"""Shared fixtures and cleanup for indexer tests."""
import pytest
from singleton import Singleton
from file_watcher import FileWatcher


@pytest.fixture(autouse=True)
def _clear_file_watcher_singleton():
    """Clear FileWatcher singleton before and after each test."""
    Singleton._instances.pop(FileWatcher, None)
    yield
    Singleton._instances.pop(FileWatcher, None)
