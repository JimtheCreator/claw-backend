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
    assert set(full) - set(basic) == {"scanner-ingestion", "scanner-backfill", "scanner-backfill-forex", "scanner-detection", "scanner-detection-forex", "scanner-scheduler"}
    queues = {arg.removeprefix("--queues=") for command in full.values()
              for arg in command if arg.startswith("--queues=")}
    assert queues == {"default,analysis", "scanner_control,scanner_ingestion",
        "scanner_backfill,scanner_backfill_15m,scanner_backfill_30m,scanner_backfill_1h,scanner_backfill_4h,scanner_backfill_1d",
        "scanner_backfill_forex,scanner_backfill_forex_15m,scanner_backfill_forex_30m,scanner_backfill_forex_1h,scanner_backfill_forex_4h,scanner_backfill_forex_1d",
        ','.join(['scanner'] + [f'scanner_{market}_{tf}' for market in ('binance_spot','massive_crypto')
                                for tf in ('15m','30m','1h','4h','1d')]),
        ','.join(f'scanner_massive_forex_{tf}' for tf in ('15m','30m','1h','4h','1d'))}
    assert '--concurrency=2' in full['scanner-backfill']
    assert '--concurrency=4' in full['scanner-backfill-forex']
    assert '--concurrency=4' in full['scanner-detection-forex']
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

    def setsockopt(self, level, option, value):
        assert (level, option, value) == (dev.socket.SOL_SOCKET, dev.socket.SO_REUSEADDR, 1)

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


def test_notifications_are_explicit_and_use_separate_database_roles():
    plan = dev.process_plan('python',True,True)
    assert plan['scanner-inbox'][-1] == 'inbox'
    assert plan['scanner-delivery'][-1] == 'delivery'
    source = {'SCANNER_API_DATABASE_URL':'api-dsn','SCANNER_WORKER_DATABASE_URL':'worker-dsn',
              'SCANNER_DATABASE_URL':'unused-admin-dsn'}
    for role, dsn in [('api','api-dsn'),('scanner-inbox','worker-dsn'),('scanner-delivery','worker-dsn'),('price-alerts','worker-dsn'),('scanner-detection',None)]:
        env = dev.child_environment(source,True,role)
        assert env.get('SCANNER_DATABASE_URL') == dsn
        assert env['SCANNER_EVENTS_ENABLED'] == '1'
        assert env['SCANNER_PUSH_ENABLED'] == ('1' if role in ('scanner-delivery','price-alerts') else '0')
        assert 'SCANNER_WORKER_DATABASE_URL' not in env
        assert 'SCANNER_API_DATABASE_URL' not in env
    assert source['SCANNER_DATABASE_URL'] == 'unused-admin-dsn'


def test_denied_group_signal_does_not_abandon_other_children(monkeypatch, capsys):
    denied, other = Mock(pid=101), Mock(pid=202)
    denied.poll.return_value = None
    def killpg(pid, sig):
        if pid == 101:
            raise PermissionError("denied")
        if sig == 0:
            raise ProcessLookupError()
    signals = Mock(side_effect=killpg)
    monkeypatch.setattr(dev.os, "killpg", signals)
    dev.stop_children({"denied": denied, "other": other}, grace=0)
    denied.send_signal.assert_any_call(signal.SIGTERM)
    denied.send_signal.assert_any_call(signal.SIGKILL)
    signals.assert_any_call(202, signal.SIGTERM)
    signals.assert_any_call(202, signal.SIGKILL)
    other.wait.assert_called()
    assert all("timeout" in call.kwargs for call in denied.wait.call_args_list)
    assert "macOS denied" in capsys.readouterr().out


def test_scanner_tasks_do_not_accumulate_unused_celery_results():
    from src.core.services import scanner_tasks, scanner_ingestion_tasks
    from src.core.services.workers.celery_worker import celery_app
    for task in [scanner_tasks.scan_market_universe, scanner_tasks.scan_scheduled_universe,
                 scanner_tasks.scan_market_instrument, scanner_tasks.finalize_scanner_batch,
                 scanner_ingestion_tasks.prepare_scanner_scan,
                 scanner_ingestion_tasks.prepare_scanner_instrument,
                 scanner_ingestion_tasks.persist_scanner_candle]:
        assert task.ignore_result
        assert not task.store_errors_even_if_ignored
    assert celery_app.conf.result_expires == 3600


def test_worker_failure_restarts_only_its_role_and_keeps_api_running(tmp_path, monkeypatch):
    monkeypatch.setattr(dev, 'LOGS', tmp_path)
    monkeypatch.setattr(dev.socket, 'socket', FreePort)
    monkeypatch.setattr(dev, 'process_plan', lambda *args: {'api':['api'], 'feed':['feed']})
    api, failed, replacement = Mock(returncode=None), Mock(returncode=1), Mock(returncode=None)
    api.poll.return_value = replacement.poll.return_value = None
    failed.poll.return_value = 1
    factory = Mock(side_effect=[api, failed, replacement])
    monkeypatch.setattr(dev.subprocess, 'Popen', factory)
    clock = [0]
    monkeypatch.setattr(dev.time, 'monotonic', lambda: clock[0])
    def advance(seconds):
        clock[0] += seconds
        if clock[0] >= 4: raise KeyboardInterrupt()
    monkeypatch.setattr(dev.time, 'sleep', advance)
    cleanup = Mock()
    monkeypatch.setattr(dev, 'stop_children', cleanup)
    assert dev.serve(False) == 0
    assert [c.args[0] for c in factory.call_args_list] == [['api'], ['feed'], ['feed']]
    assert cleanup.call_args_list[0].args == ({'feed': failed},)
    assert cleanup.call_args_list[-1].args == ({'api':api, 'feed':replacement},)
