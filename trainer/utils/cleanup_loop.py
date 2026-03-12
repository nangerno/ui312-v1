"""Periodic cleanup loop — runs as an asyncio background task in the main event loop.

Design notes:
- Runs entirely inside the main FastAPI event loop so it can safely acquire
  _task_lock from trainer.tasks without cross-event-loop issues.
- All blocking Docker SDK calls are offloaded to a thread pool via
  asyncio.to_thread so the event loop is never blocked.
- start_cleanup_loop_in_thread is removed; the caller (asgi.py) creates an
  asyncio.Task directly via asyncio.create_task(periodically_cleanup_tasks_and_cache()).
"""

import asyncio
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path

import docker
from dateutil.parser import isoparse

from core.models.utility_models import TaskStatus
from trainer import constants as cst
from trainer.tasks import _task_lock
from trainer.tasks import save_task_history
from trainer.tasks import task_history
from trainer.utils.logging_two import get_all_context_tags
from trainer.utils.logging_two import get_logger
from trainer.utils.logging_two import stream_container_logs


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Sync helpers — each runs inside asyncio.to_thread
# ---------------------------------------------------------------------------

def _run_cache_cleanup_container_sync(abs_task_path: Path) -> None:
    """Start the cache-cleaner Docker container and stream its logs (blocking)."""
    client = docker.from_env()
    try:
        container = client.containers.run(
            image=cst.CACHE_CLEANER_DOCKER_IMAGE,
            volumes={
                cst.VOLUME_NAMES[0]: {"bind": "/checkpoints", "mode": "rw"},
                cst.VOLUME_NAMES[1]: {"bind": "/cache", "mode": "rw"},
                str(abs_task_path): {"bind": "/app/trainer/task_history.json", "mode": "ro"},
            },
            remove=True,
            detach=True,
        )
        stream_container_logs(container, get_all_context_tags())
    except Exception as e:
        logger.error(f"Cache cleanup container failed: {e}")


def _cleanup_stopped_containers_sync() -> None:
    """Remove stopped/created containers older than 1 hour and prune unused volumes (blocking)."""
    client = docker.from_env()

    try:
        all_containers = client.containers.list(all=True)
        containers_to_remove = []

        for container in all_containers:
            if container.status not in ("created", "exited"):
                continue
            try:
                created_str = container.attrs.get("Created", "")
                if created_str:
                    created_dt = isoparse(created_str)
                    age_hours = (datetime.now(timezone.utc) - created_dt).total_seconds() / 3600
                    if age_hours > 1:
                        containers_to_remove.append(container)
                else:
                    containers_to_remove.append(container)
            except Exception as e:
                logger.warning(f"Could not parse creation time for {container.name}: {e}")
                containers_to_remove.append(container)

        if containers_to_remove:
            logger.info(f"Cleaning up {len(containers_to_remove)} stopped/created containers...")
            for container in containers_to_remove:
                try:
                    container.remove(force=True, v=True)
                    logger.debug(f"Removed container: {container.name or container.id[:12]}")
                except Exception as e:
                    logger.warning(f"Failed to remove container {container.name or container.id[:12]}: {e}")

        try:
            prune_result = client.volumes.prune()
            if prune_result.get("SpaceReclaimed", 0) > 0:
                logger.info(f"Pruned volumes: {prune_result}")
        except Exception as e:
            logger.warning(f"Failed to prune volumes: {e}")

        logger.info("Container and volume cleanup completed.")

    except Exception as e:
        logger.error(f"Error during container/volume cleanup: {e}")


# ---------------------------------------------------------------------------
# Main async loop
# ---------------------------------------------------------------------------

async def periodically_cleanup_tasks_and_cache(poll_interval_seconds: int = 600) -> None:
    """Periodically mark stale tasks as failed and clean up Docker resources.

    Must be run as an asyncio.Task inside the main event loop so that
    _task_lock (an asyncio.Lock) can be safely awaited.
    """
    while True:
        try:
            # ----------------------------------------------------------------
            # Step 1: Mark timed-out TRAINING tasks as FAILED
            # ----------------------------------------------------------------
            if task_history:
                async with _task_lock:
                    now = datetime.utcnow()
                    changed = False
                    for task in task_history:
                        if task.status != TaskStatus.TRAINING or not task.started_at:
                            continue
                        timeout = (
                            timedelta(hours=task.training_data.hours_to_complete)
                            + timedelta(minutes=cst.STALE_TASK_GRACE_MINUTES)
                        )
                        deadline = task.started_at + timeout
                        if now > deadline:
                            task.status = TaskStatus.FAILURE
                            task.finished_at = now
                            task.logs.append(
                                f"[{now.isoformat()}] Task marked as FAILED due to timeout."
                            )
                            changed = True
                            logger.warning(f"Task {task.training_data.task_id} marked FAILED due to timeout.")
                    if changed:
                        await save_task_history()

                # ----------------------------------------------------------------
                # Step 2: Run the cache-cleaner container
                # ----------------------------------------------------------------
                abs_task_path = Path(cst.TASKS_FILE_PATH).resolve()
                if abs_task_path.exists():
                    logger.info("Starting cache cleanup container...")
                    try:
                        await asyncio.to_thread(_run_cache_cleanup_container_sync, abs_task_path)
                        logger.info("Cache cleanup container finished.")
                    except Exception as e:
                        logger.error(f"Failed to run cache cleanup container: {e}")

            # ----------------------------------------------------------------
            # Step 3: Remove stopped containers and prune volumes
            # ----------------------------------------------------------------
            try:
                await asyncio.to_thread(_cleanup_stopped_containers_sync)
            except Exception as e:
                logger.error(f"Error during container/volume cleanup: {e}")

        except asyncio.CancelledError:
            logger.info("Cleanup loop cancelled.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in cleanup loop: {e}")

        await asyncio.sleep(poll_interval_seconds)
