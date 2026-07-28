from io import BytesIO

import numpy as np
import pytest

from capmas.contracts.core import ArtifactRef
from capmas.perception.artifact_bridge import (
    EncodedArtifactStore,
    FileArtifactStore,
    NumpyArtifactCodec,
)


def test_file_artifact_round_trip_uses_content_addressed_uri(tmp_path) -> None:
    store = FileArtifactStore(tmp_path, checksum="sha256")

    reference = store.put(b"rgb-bytes", "image/rgb")

    assert reference.uri.startswith("artifact://sha256/")
    assert reference.sha256 is not None
    assert reference.byte_size == len(b"rgb-bytes")
    assert store.exists(reference)
    assert store.open(reference).read() == b"rgb-bytes"


def test_release_does_not_delete_pinned_content_before_ttl(tmp_path) -> None:
    store = FileArtifactStore(tmp_path, checksum="sha256")
    reference = store.put(b"map", "application/octet-stream")

    store.pin(reference, ttl_ms=1000)
    store.release(reference)

    assert store.exists(reference)


def test_file_artifact_store_rejects_unknown_uri_scheme(tmp_path) -> None:
    store = FileArtifactStore(tmp_path)
    reference = ArtifactRef("artifact://memory/not-local", "image/rgb")

    assert not store.exists(reference)


def test_file_artifact_store_can_skip_fsync_while_preserving_atomic_publish(tmp_path) -> None:
    store = FileArtifactStore(tmp_path, fsync=False)

    reference = store.put(b"capture", "application/octet-stream")

    assert store.exists(reference)
    assert store.open(reference).read() == b"capture"


def test_numpy_artifact_round_trip_preserves_dtype_shape_and_values(tmp_path) -> None:
    store = EncodedArtifactStore(FileArtifactStore(tmp_path), NumpyArtifactCodec())
    value = np.array([[0, 255, 17], [4, 8, 16]], dtype=np.uint16)

    reference = store.put(value, "image/depth")
    restored = store.get(reference)

    assert reference.media_type == "image/depth+npy"
    assert reference.byte_size is not None and reference.byte_size > 0
    assert reference.sha256 is not None
    assert isinstance(restored, np.ndarray)
    assert restored.dtype == value.dtype
    assert restored.shape == value.shape
    np.testing.assert_array_equal(restored, value)


def test_numpy_codec_rejects_object_arrays_without_pickle(tmp_path) -> None:
    codec = NumpyArtifactCodec()

    with pytest.raises(TypeError, match="object dtype"):
        codec.encode(np.array([{"not": "pickle"}], dtype=object))


def test_encoded_store_keeps_raw_bytes_for_non_codec_artifacts(tmp_path) -> None:
    store = EncodedArtifactStore(FileArtifactStore(tmp_path), NumpyArtifactCodec())

    reference = store.put(b"map-json", "application/json")

    assert reference.media_type == "application/json"
    assert store.get(reference) == b"map-json"


def test_encoded_store_reports_capture_write_metrics(tmp_path) -> None:
    store = EncodedArtifactStore(FileArtifactStore(tmp_path), NumpyArtifactCodec())

    store.put(np.arange(4, dtype=np.float32), "array/ee-pose")
    metrics = store.metrics()

    assert metrics.put_count == 1
    assert metrics.bytes_written > 0
    assert metrics.total_put_latency_ms >= metrics.last_put_latency_ms >= 0.0


def test_encoded_store_detects_checksum_corruption_before_decode(tmp_path) -> None:
    store = EncodedArtifactStore(FileArtifactStore(tmp_path), NumpyArtifactCodec())
    reference = store.put(np.arange(4, dtype=np.float32), "array/ee-pose")
    artifact_path = next(tmp_path.glob("[0-9a-f]" * 64))
    artifact_path.write_bytes(b"corrupt".ljust(reference.byte_size or 0, b"!"))

    with pytest.raises(ValueError, match="checksum"):
        store.get(reference)


def test_file_publication_leaves_only_complete_content_addressed_artifact(tmp_path) -> None:
    store = EncodedArtifactStore(FileArtifactStore(tmp_path), NumpyArtifactCodec())
    reference = store.put(np.arange(16, dtype=np.uint8), "image/rgb")

    assert store.exists(reference)
    assert not list(tmp_path.glob(".artifact-*"))
    with store.open(reference) as handle:
        assert handle.read(6) == b"\x93NUMPY"


def test_numpy_codec_rejects_truncated_payload() -> None:
    with pytest.raises(ValueError, match="decode"):
        NumpyArtifactCodec().decode(BytesIO(b"\x93NUMPY"))
