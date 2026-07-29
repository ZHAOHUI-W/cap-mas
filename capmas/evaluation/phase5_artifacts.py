"""Immutable, secret-free artifact directories for Phase 5 experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import os
import tempfile
import time
from collections.abc import Mapping


_SUBDIRECTORIES = ("logs", "results", "traces", "evidence", "artifacts")
_SENSITIVE_KEYS = {"api_key", "api-key", "authorization", "proxy-authorization"}


@dataclass(frozen=True)
class Phase5RunDirectory:
    path: Path

    @classmethod
    def create(
        cls,
        root: str | Path,
        experiment_name: str,
        run_id: str,
    ) -> "Phase5RunDirectory":
        if not experiment_name or not run_id:
            raise ValueError("experiment name and run id must not be empty")
        base = Path(root) / experiment_name
        base.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        for suffix in range(1, 10_000):
            label = f"{stamp}_{run_id}" if suffix == 1 else f"{stamp}_{run_id}_{suffix}"
            path = base / label
            try:
                path.mkdir()
            except FileExistsError:
                continue
            for name in _SUBDIRECTORIES:
                (path / name).mkdir()
            return cls(path)
        raise RuntimeError("could not allocate a unique Phase 5 run directory")

    def write_json(self, name: str, payload: object) -> Path:
        path = self._safe_path(name)
        encoded = json.dumps(_redact(payload), indent=2, sort_keys=True, default=str) + "\n"
        self._atomic_write(path, encoded.encode("utf-8"))
        return path

    def write_text(self, name: str, content: str) -> Path:
        """Atomically publish a redacted text artifact."""
        path = self._safe_path(name)
        self._atomic_write(path, content.encode("utf-8"))
        return path

    def finalize_manifest(self) -> Path:
        entries = []
        for path in sorted(item for item in self.path.rglob("*") if item.is_file()):
            if path.name == "manifest.json":
                continue
            relative = path.relative_to(self.path).as_posix()
            entries.append(
                {
                    "path": relative,
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        return self.write_json(
            "manifest.json",
            {"run_dir": self.path.name, "files": entries},
        )

    def log_path(self, name: str = "runner.log") -> Path:
        if Path(name).name != name or not name:
            raise ValueError("log name must be a plain file name")
        path = self.path / "logs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _safe_path(self, name: str) -> Path:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact name must stay within the run directory")
        path = self.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise


def _redact(value: object, *, key: str | None = None) -> object:
    if key is not None and (
        key.lower() in _SENSITIVE_KEYS
        or key.lower() in {"headers", "provider_headers"}
        or "provider_header" in key.lower()
    ):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(item_key): _redact(item_value, key=str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
