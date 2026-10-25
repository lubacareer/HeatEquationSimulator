"""Single-directory job publication, serialized with GUI cancellation requests."""
from __future__ import annotations

from contextlib import contextmanager
import os
import json
from pathlib import Path
import shutil
import tempfile
from typing import Callable

from .models import SimulationCancelled


@contextmanager
def _publication_lock(request_path: Path):
    lock_path = request_path.with_suffix(".publication.lock")
    with lock_path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def request_cancel(request_path: Path) -> bool:
    """Return false when publication already won the completion/cancel race."""
    with _publication_lock(request_path):
        if request_path.with_suffix(".published").exists():
            return False
        request_path.with_suffix(".cancel").touch()
        return True


class JobPublication:
    def __init__(self, store, request_path: Path, cancelled: Callable[[], bool]):
        from .store import RunStore
        self.request_path = request_path
        self.cancelled = cancelled
        self.pending_root = (store.root / ".pending").resolve()
        self.pending_root.mkdir(exist_ok=True)
        self.stage = Path(tempfile.mkdtemp(prefix="job-", dir=self.pending_root))
        self.store = RunStore(self.stage, target_root=store.root.resolve(), publication_id=request_path.stem)

    def commit(self) -> None:
        # Preserve replayable paths in auxiliary request/metadata JSON files.
        for path in self.stage.rglob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            path.write_text(json.dumps(self.store._metadata(data), indent=2), encoding="utf-8")
        # Serialization, checkpoints and copied aliases remain invisible until
        # one atomic directory move. The same lock guards GUI cancellation.
        with _publication_lock(self.request_path):
            if self.cancelled() or self.request_path.with_suffix(".cancel").exists():
                raise SimulationCancelled("Job cancelled before publishing its results.")
            runs = list((self.stage / "runs").iterdir())
            if len(runs) != 1:
                raise RuntimeError("A job must publish exactly one source run or analysis bundle.")
            source = runs[0]
            destination = self.store._destinations[source.name]
            if destination.exists():
                raise ValueError("The publication destination already exists.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            try:
                self.request_path.with_suffix(".published").write_text("published", encoding="ascii")
            except OSError:
                os.replace(destination, source)
                raise

    def close(self) -> None:
        target = self.stage.resolve()
        if target != self.pending_root and target.is_relative_to(self.pending_root) and self.stage.is_dir():
            shutil.rmtree(self.stage)
