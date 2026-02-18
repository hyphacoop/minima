"""
Tests for watchdog_health_check: restart logic, backoff, max retries.
Run: cd indexer && .venv/bin/python -m pytest test_watchdog_health.py -v -s
"""
import asyncio
import logging
import sys
import pytest
from unittest.mock import MagicMock, patch

# Mock the heavy/infra modules before importing app.py
# app.py has module-level side effects: Indexer(), MinimaStore.create_db_and_tables(), nltk downloads
_mocked = {}
for mod_name in ("indexer",):
    if mod_name not in sys.modules:
        _mocked[mod_name] = True
        sys.modules[mod_name] = MagicMock()

# Ensure app is imported fresh with mocks applied
_app_was_loaded = "app" in sys.modules
if _app_was_loaded:
    _saved_app = sys.modules.pop("app")

# Patch MinimaStore.create_db_and_tables and Indexer to prevent side effects
with patch("storage.MinimaStore.create_db_and_tables", return_value=None), \
     patch.dict(sys.modules, {"indexer": MagicMock()}):
    import app as _app_module
    watchdog_health_check = _app_module.watchdog_health_check

# Restore app module state
if _app_was_loaded:
    sys.modules["app"] = _saved_app

# Clean up any modules we temporarily injected
for mod_name in _mocked:
    if mod_name in sys.modules and isinstance(sys.modules[mod_name], MagicMock):
        del sys.modules[mod_name]


@pytest.mark.asyncio
async def test_health_check_restarts_dead_watcher():
    """When is_alive() returns False, health check should stop+start the watcher."""
    mock_watcher = MagicMock()
    # First check: dead -> restart succeeds -> alive on next checks
    mock_watcher.is_alive.side_effect = [False, True, True]
    mock_watcher.stop.return_value = None
    mock_watcher.start.return_value = None

    call_count = 0
    _real_sleep = asyncio.sleep

    async def fake_sleep(seconds):
        nonlocal call_count
        call_count += 1
        if call_count > 4:
            raise asyncio.CancelledError()
        await _real_sleep(0)

    with patch("asyncio.sleep", side_effect=fake_sleep):
        with pytest.raises(asyncio.CancelledError):
            await watchdog_health_check(mock_watcher)

    mock_watcher.stop.assert_called()
    mock_watcher.start.assert_called()


@pytest.mark.asyncio
async def test_health_check_gives_up_after_max_retries(caplog):
    """After 5 failed restarts, health check should log critical and break."""
    mock_watcher = MagicMock()
    mock_watcher.is_alive.return_value = False
    mock_watcher.stop.return_value = None
    mock_watcher.start.side_effect = RuntimeError("observer broken")

    _real_sleep = asyncio.sleep

    async def fake_sleep(seconds):
        await _real_sleep(0)

    with patch("asyncio.sleep", side_effect=fake_sleep):
        with caplog.at_level(logging.CRITICAL):
            await watchdog_health_check(mock_watcher)

    # start() should have been called 5 times (max_retries)
    assert mock_watcher.start.call_count == 5
    assert any("max retries" in r.message.lower() for r in caplog.records if r.levelno >= logging.CRITICAL)
