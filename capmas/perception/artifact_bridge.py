from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from hashlib import sha256
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, BinaryIO, Protocol

from capmas.contracts.core import ArtifactRef


class SharedArtifactStore(Protocol):
    def put(self, value: bytes, media_type: str) -> ArtifactRef: ...

    def open(self, reference: ArtifactRef) -> BinaryIO: ...

    def exists(self, reference: ArtifactRef) -> bool: ...

    def pin(self, reference: ArtifactRef, ttl_ms: int) -> None: ...

    def release(self, reference: ArtifactRef) -> None: ...


class ArtifactSink(Protocol):
    """Minimal capture-side seam for values that become artifact references."""

    def put(self, value: object, media_type: str) -> ArtifactRef: ...


class ArtifactCodec(Protocol):
    """Encode runtime values into self-describing shared artifact bytes."""

    media_type_suffix: str

    def encode(self, value: object) -> bytes: ...

    def decode(self, payload: bytes) -> object: ...

    def encoded_media_type(self, media_type: str) -> str: ...


@dataclass(frozen=True)
class ArtifactIOMetrics:
    put_count: int
    bytes_written: int
    total_put_latency_ms: float
    last_put_latency_ms: float


class NumpyArtifactCodec:
    """Serialize numeric NumPy arrays as non-pickle `.npy` payloads."""

    media_type_suffix = "+npy"

    def encode(self, value: object) -> bytes:
        np = self._numpy()
        if not isinstance(value, np.ndarray):
            raise TypeError("NumpyArtifactCodec accepts numpy.ndarray values only")
        if value.dtype.hasobject:
            raise TypeError("NumpyArtifactCodec rejects object dtype arrays")
        output = BytesIO()
        try:
            np.save(output, value, allow_pickle=False)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"failed to encode NumPy artifact: {exc}") from exc
        return output.getvalue()

    def decode(self, payload: bytes) -> object:
        np = self._numpy()
        try:
            value = np.load(BytesIO(payload), allow_pickle=False)
        except Exception as exc:
            raise ValueError(f"failed to decode NumPy artifact: {exc}") from exc
        if not isinstance(value, np.ndarray):
            raise ValueError("failed to decode NumPy artifact: payload is not an array")
        if value.dtype.hasobject:
            raise ValueError("failed to decode NumPy artifact: object dtype is forbidden")
        return np.array(value, copy=True)

    def encoded_media_type(self, media_type: str) -> str:
        if not isinstance(media_type, str) or not media_type:
            raise ValueError("media_type must be a non-empty string")
        return media_type if media_type.endswith(self.media_type_suffix) else media_type + self.media_type_suffix

    @staticmethod
    def _numpy() -> Any:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - environment dependency guard
            raise RuntimeError("NumpyArtifactCodec requires numpy") from exc
        return np


class EncodedArtifactStore:
    """Codec-aware facade over a bytes-only shared artifact store.

    NumPy arrays are encoded before publication and decoded on ``get``. Raw
    bytes remain supported for map or JSON artifacts produced by the worker.
    ``open`` intentionally exposes bytes because it is the cross-process
    transport seam; callers that need a runtime value use ``get``.
    """

    def __init__(self, store: SharedArtifactStore, codec: ArtifactCodec) -> None:
        self.store = store
        self.codec = codec
        self._put_count = 0
        self._bytes_written = 0
        self._total_put_latency_ms = 0.0
        self._last_put_latency_ms = 0.0

    def put(self, value: object, media_type: str) -> ArtifactRef:
        started = time.perf_counter()
        if isinstance(value, (bytes, bytearray, memoryview)):
            reference = self.store.put(bytes(value), media_type)
        else:
            payload = self.codec.encode(value)
            reference = self.store.put(payload, self.codec.encoded_media_type(media_type))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._put_count += 1
        self._bytes_written += reference.byte_size or 0
        self._total_put_latency_ms += elapsed_ms
        self._last_put_latency_ms = elapsed_ms
        return reference

    def metrics(self) -> ArtifactIOMetrics:
        return ArtifactIOMetrics(
            put_count=self._put_count,
            bytes_written=self._bytes_written,
            total_put_latency_ms=self._total_put_latency_ms,
            last_put_latency_ms=self._last_put_latency_ms,
        )

    def get(self, reference: ArtifactRef) -> object:
        payload = self._read_verified(reference)
        if reference.media_type.endswith(self.codec.media_type_suffix):
            return self.codec.decode(payload)
        return payload

    def open(self, reference: ArtifactRef) -> BinaryIO:
        return self.store.open(reference)

    def exists(self, reference: ArtifactRef) -> bool:
        return self.store.exists(reference)

    def pin(self, reference: ArtifactRef, ttl_ms: int) -> None:
        self.store.pin(reference, ttl_ms)

    def release(self, reference: ArtifactRef) -> None:
        self.store.release(reference)

    def _read_verified(self, reference: ArtifactRef) -> bytes:
        try:
            with self.store.open(reference) as handle:
                payload = handle.read()
        except (OSError, ValueError) as exc:
            raise ValueError(f"unable to read artifact {reference.uri}: {exc}") from exc
        if reference.byte_size is not None and len(payload) != reference.byte_size:
            raise ValueError(
                f"artifact byte size mismatch for {reference.uri}: "
                f"expected={reference.byte_size} actual={len(payload)}"
            )
        if reference.sha256 is not None:
            digest = sha256(payload).hexdigest()
            if digest != reference.sha256:
                raise ValueError(
                    f"artifact checksum mismatch for {reference.uri}: "
                    f"expected={reference.sha256} actual={digest}"
                )
        return payload


class FileArtifactStore:
    """Content-addressed artifact store suitable for thread/process transport."""

    _URI = re.compile(r"^artifact://sha256/([0-9a-f]{64})$")

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        checksum: str = "sha256",
        max_bytes: int = 512 * 1024 * 1024,
        fsync: bool = True,
    ) -> None:
        if checksum != "sha256":
            raise ValueError("only sha256 artifacts are supported")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.fsync = fsync

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
                if self.fsync:
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
