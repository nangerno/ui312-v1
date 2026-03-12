import asyncio
import os
import shutil
import threading
from urllib.parse import urlparse

import pynvml
from git import GitCommandError
from git import Repo

import trainer.constants as cst
from core.models.utility_models import GPUInfo
from core.models.utility_models import GPUType
from trainer.tasks import get_running_tasks


# ---------------------------------------------------------------------------
# Repository cloning
# ---------------------------------------------------------------------------

# Per-repo locks so two concurrent requests for the same repo don't clobber
# each other while running inside asyncio.to_thread.
_repo_clone_locks: dict[str, threading.Lock] = {}
_repo_locks_registry = threading.Lock()


def _get_repo_lock(repo_url: str) -> threading.Lock:
    with _repo_locks_registry:
        if repo_url not in _repo_clone_locks:
            _repo_clone_locks[repo_url] = threading.Lock()
        return _repo_clone_locks[repo_url]


def clone_repo(repo_url: str, parent_dir: str, branch: str = None, commit_hash: str = None) -> str:
    # Validate URL scheme to prevent SSRF via file://, git://, etc.
    parsed = urlparse(repo_url)
    if parsed.scheme not in ("https", "http") or not parsed.netloc:
        raise ValueError(
            f"Untrusted repository URL scheme '{parsed.scheme}'. Only HTTPS/HTTP URLs are allowed."
        )

    repo_name = os.path.basename(parsed.path)
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    repo_dir = os.path.join(parent_dir, repo_name)

    with _get_repo_lock(repo_url):
        if os.path.exists(repo_dir):
            try:
                repo = Repo(repo_dir)
                current_commit = repo.head.commit.hexsha

                if commit_hash and current_commit.startswith(commit_hash):
                    return repo_dir
                elif branch and repo.active_branch.name == branch:
                    return repo_dir
                shutil.rmtree(repo_dir)
            except Exception:
                shutil.rmtree(repo_dir)

        try:
            repo = Repo.clone_from(repo_url, repo_dir, branch=branch) if branch else Repo.clone_from(repo_url, repo_dir)

            if commit_hash:
                repo.git.fetch("--all")
                try:
                    repo.git.checkout(commit_hash)
                except GitCommandError as checkout_error:
                    if "pathspec" in str(checkout_error) and "did not match any file(s) known to git" in str(checkout_error):
                        raise RuntimeError(f"Invalid commit hash '{commit_hash}' - commit not found in repository")
                    else:
                        raise

            return repo_dir

        except GitCommandError as e:
            raise RuntimeError(f"Error in cloning: {str(e)}")

        except Exception as e:
            raise RuntimeError(f"Unexpected error while cloning: {str(e)}")


# ---------------------------------------------------------------------------
# GPU info — blocking pynvml calls isolated to a sync helper so the async
# wrapper can offload them to a thread without blocking the event loop.
# ---------------------------------------------------------------------------

def _get_gpu_info_sync() -> list[GPUInfo]:
    pynvml.nvmlInit()
    try:
        device_count = pynvml.nvmlDeviceGetCount()

        index_to_type: dict[int, GPUType] = {}
        index_to_vram: dict[int, int] = {}

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode("utf-8").upper()
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_vram_gb = int(mem_info.total / 1024 / 1024 / 1024)

            for gpu_type in GPUType:
                if gpu_type.value in name:
                    index_to_type[i] = gpu_type
                    index_to_vram[i] = total_vram_gb
                    break

        busy_gpu_ids: set[int] = set()
        running_tasks = get_running_tasks()
        for task in running_tasks:
            for gpu_id in task.gpu_ids:
                busy_gpu_ids.add(gpu_id)

        gpu_infos: list[GPUInfo] = []
        for gpu_id in range(device_count):
            if gpu_id not in index_to_type:
                continue
            gpu_infos.append(
                GPUInfo(
                    gpu_id=gpu_id,
                    gpu_type=index_to_type[gpu_id],
                    vram_gb=index_to_vram[gpu_id],
                    available=gpu_id not in busy_gpu_ids,
                )
            )
        return gpu_infos
    finally:
        pynvml.nvmlShutdown()


async def get_gpu_info() -> list[GPUInfo]:
    """Async wrapper — delegates blocking pynvml calls to a thread pool."""
    return await asyncio.to_thread(_get_gpu_info_sync)


# ---------------------------------------------------------------------------
# WandB environment helpers
# ---------------------------------------------------------------------------

def build_wandb_env(task_id: str, hotkey: str) -> dict:
    wandb_path = f"{cst.WANDB_LOGS_DIR}/{task_id}_{hotkey}"
    return {
        "WANDB_MODE": "offline",
        **{key: wandb_path for key in cst.WANDB_DIRECTORIES},
    }


# ---------------------------------------------------------------------------
# Container error extraction
# ---------------------------------------------------------------------------

def extract_container_error(logs: str) -> str | None:
    """Return the most relevant error line from container logs.

    Scans from the bottom up for lines matching common error patterns.
    Falls back to the last non-empty line if no error keyword is found.
    """
    lines = logs.strip().splitlines()
    if not lines:
        return None

    error_keywords = (
        "error",
        "exception",
        "traceback",
        "fatal",
        "killed",
        "oom",
        "cuda error",
        "assertion",
        "segfault",
        "sigkill",
    )

    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if any(kw in stripped.lower() for kw in error_keywords):
            return stripped

    # Fallback: last non-empty line
    for line in reversed(lines):
        if line.strip():
            return line.strip()

    return None


# ---------------------------------------------------------------------------
# GPU availability check
# ---------------------------------------------------------------------------

def are_gpus_available(requested_gpu_ids: list[int]) -> bool:
    """Return True if *all* requested GPU IDs are free (not used by a running task)."""
    running_tasks = get_running_tasks()
    for task in running_tasks:
        for gpu_id in requested_gpu_ids:
            if gpu_id in task.gpu_ids:
                return False
    return True
