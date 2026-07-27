from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import re
import tempfile
import time
from typing import BinaryIO, Protocol

from capmas.contracts.core import ArtifactRef


class SharedArtifactStore(Protocol):
    def put(self, value: bytes, media_type: str) -> ArtifactRef: ...

    def open(self, reference: ArtifactRef) -> BinaryIO: ...

    def exists(self, reference: ArtifactRef) -> bool: ...

    def pin(self, reference: ArtifactRef, ttl_ms: int) -> None: ...

    def release(self, reference: ArtifactRef) -> None: ...


class FileArtifactStore:
    """Content-addressed artifact store suitable for thread/process transport."""

    _URI = re.compile(r"^artifact://sha256/([0-9a-f]{64})$")

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        checksum: str = "sha256",
        max_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        if checksum != "sha256":
            raise ValueError("only sha256 artifacts are supported")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes

    def put(self, value: bytes, media_type: str) -> ArtifactRef:
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError("FileArtifactStore accepts bytes only")
        payload = bytes(value)
        if len(payload) > self.max_bytes:
            raise ValueError(f"artifact exceeds max_bytes={self.max_bytes}")
        digest = sha256(payload).hexdigest()
        reference = ArtifactRef(
            uri=f"artifact://sha256/{digest}",
            media_type=media_type,
            sha256=digest,
            byte_size=len(payload),
        )
        destination = self._path(reference)
        if not destination.exists():
            with tempfile.NamedTemporaryFile(dir=self.root, prefix=".artifact-", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        metadata = destination.with_suffix(".meta")
        if not metadata.exists():
            metadata.write_text(media_type, encoding="utf-8")
        return reference

    def open(self, reference: ArtifactRef) -> BinaryIO:
        return self._path(reference).open("rb")

    def exists(self, reference: ArtifactRef) -> bool:
        try:
            path = self._path(reference)
        except ValueError:
            return False
        return path.is_file()

    def pin(self, reference: ArtifactRef, ttl_ms: int) -> None:
        if ttl_ms < 0:
            raise ValueError("ttl_ms must be non-negative")
        path = self._path(reference)
        if not path.is_file():
            raise FileNotFoundError(reference.uri)
        path.with_suffix(".pin").write_text(
            str(time.time_ns() + ttl_ms * 1_000_000),
            encoding="ascii",
        )

    def release(self, reference: ArtifactRef) -> None:
        self._path(reference).with_suffix(".pin").unlink(missing_ok=True)

    def cleanup_expired(self, now_ns: int | None = None) -> int:
        now = time.time_ns() if now_ns is None else now_ns
        removed = 0
        for pin in self.root.glob("*.pin"):
            try:
                expires = int(pin.read_text(encoding="ascii"))
            except (OSError, ValueError):
                expires = 0
            if expires > now:
                continue
            artifact = pin.with_suffix("")
            pin.unlink(missing_ok=True)
            if artifact.is_file():
                artifact.unlink()
                artifact.with_suffix(".meta").unlink(missing_ok=True)
                removed += 1
        return removed

    def _path(self, reference: ArtifactRef) -> Path:
        match = self._URI.fullmatch(reference.uri)
        if match is None:
            raise ValueError(f"unsupported artifact URI: {reference.uri}")
        digest = match.group(1)
        if reference.sha256 is not None and reference.sha256 != digest:
            raise ValueError("artifact checksum does not match URI")
        return self.root / digest
