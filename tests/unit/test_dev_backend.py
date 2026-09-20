"""Local launcher tests: no market, account, database or provider connections."""
import fcntl
import os
import select
import signal
import subprocess
import sys
from unittest.mock import Mock

from scripts import dev_backend as dev


def test_scanner_roles_are_explicit_and_notification_queues_are_excluded():
    basic = dev.process_plan("python", False)
    full = dev.process_plan("python", True)
    assert set(full) - set(basic) == {"scanner-ingestion", "scanner-detection", "scanner-scheduler"}
    queues = {arg.removeprefix("--queues=") for command in full.values()
              for arg in command if arg.startswith("--queues=")}
    assert queues == {"default,analysis", "scanner_ingestion", "scanner"}
    assert all(command[0] == "python" for command in full.values())
    assert "-m" in full["app-worker"] and "celery" in full["app-worker"]
    assert not any("notification" in arg or "broadcast" in arg for command in full.values() for arg in command)


def test_children_disable_notification_flags_without_mutating_parent_environment():
    parent = {"SCANNER_EVENTS_ENABLED": "1", "SCANNER_PUSH_ENABLED": "1", "REDIS_URL": "test"}
    child = dev.child_environment(parent)
    assert child["REDIS_URL"] == "test"
    assert child["SCANNER_EVENTS_ENABLED"] == child["SCANNER_PUSH_ENABLED"] == child["SCANNER_WATCHES_ENABLED"] == "0"
    assert parent["SCANNER_EVENTS_ENABLED"] == "1"
    assert child["PYTHONPATH"].split(os.pathsep) == [str(dev.ROOT / "src"), str(dev.ROOT)]


def test_failed_preflight_never_starts_workers(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dev_backend.py", "start", "--scanner"])
    monkeypatch.setattr(dev, "check_configuration", lambda: False)
    start = Mock()
    monkeypatch.setattr(dev, "serve", start)
    assert dev.main() == 1
    start.assert_not_called()


def test_duplicate_launcher_is_rejected_before_spawning(tmp_path, monkeypatch):
    monkeypatch.setattr(dev, "LOGS", tmp_path)
    popen = Mock()
    monkeypatch.setattr(dev.subprocess, "Popen", popen)
    with (tmp_path / "launcher.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert dev.serve(False) == 1
    popen.assert_not_called()


class FreePort:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def bind(self, address):
        pass


def test_occupied_api_port_never_starts_duplicate_feeds(tmp_path, monkeypatch):
    class OccupiedPort(FreePort):
        def bind(self, address):
            raise OSError("occupied")

    monkeypatch.setattr(dev, "LOGS", tmp_path)
    monkeypatch.setattr(dev.socket, "socket", OccupiedPort)
    popen = Mock()
    monkeypatch.setattr(dev.subprocess, "Popen", popen)
    assert dev.serve(False) == 1
    popen.assert_not_called()


def test_child_exit_stops_all_launched_children(tmp_path, monkeypatch):
    monkeypatch.setattr(dev, "LOGS", tmp_path)
    monkeypatch.setattr(dev.socket, "socket", FreePort)
    monkeypatch.setattr(dev, "process_plan", lambda *args: {"api": ["api"], "feed": ["feed"]})
    first, second = Mock(returncode=1), Mock(returncode=None)
    first.poll.return_value = 1
    monkeypatch.setattr(dev.subprocess, "Popen", Mock(side_effect=[first, second]))
    cleanup = Mock()
    monkeypatch.setattr(dev, "stop_children", cleanup)
    assert dev.serve(False) == 1
    cleanup.assert_called_once_with({"api": first, "feed": second})


def test_shutdown_reaps_an_owned_process_that_ignores_termination():
    process = subprocess.Popen([sys.executable, "-u", "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready'); time.sleep(60)"],
        start_new_session=True, stdout=subprocess.PIPE, text=True)
    try:
        assert select.select([process.stdout], [], [], 5)[0]
        assert process.stdout.readline().strip() == "ready"
        dev.stop_children({"owned": process}, grace=0.1)
        assert process.returncode == -signal.SIGKILL
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)
        process.stdout.close()
