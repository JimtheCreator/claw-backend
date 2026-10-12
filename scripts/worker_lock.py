"""Prevent competing copies of a notification worker in one checkout."""
from contextlib import contextmanager
from pathlib import Path
import fcntl


@contextmanager
def worker_lock(name, directory=None):
    directory = Path(directory) if directory else Path(__file__).resolve().parents[1] / "logs" / "dev-backend"
    directory.mkdir(parents=True, exist_ok=True)
    # Keep the file: unlinking a locked file allows another inode to be locked.
    with (directory / f"{name}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"{name} is already running in this checkout") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
