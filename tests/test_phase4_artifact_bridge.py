from capmas.contracts.core import ArtifactRef
from capmas.perception.artifact_bridge import FileArtifactStore


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
