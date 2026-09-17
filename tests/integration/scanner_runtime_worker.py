"""Actual production task functions on a test-only, restricted Celery worker."""
import os
from tests.integration.scanner_runtime_support import install_local_guard

install_local_guard()
if os.getenv("SCANNER_RUNTIME_RECOVERY") == "1":
    from tests.integration.scanner_runtime_support import install_failure_hooks
    install_failure_hooks()

from src.core.services.workers.celery_worker import celery_app

celery_app.conf.update(include=[
    "src.core.services.scanner_tasks",
    "src.core.services.scanner_ingestion_tasks",
], worker_enable_remote_control=False)


@celery_app.task(name="scanner_runtime_probe", queue="scanner")
def probe():
    return {"pid": os.getpid(), "isolated": os.getenv("SCANNER_RUNTIME_TEST") == "1"}
