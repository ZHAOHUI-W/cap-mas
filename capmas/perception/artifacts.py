from __future__ import annotations

from typing import Any
from uuid import uuid4

from capmas.contracts.core import ArtifactRef


class InMemoryArtifactStore:
    """Test and prototype artifact sink; production can replace it with a CAS."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    def put(self, value: Any, media_type: str) -> ArtifactRef:
        uri = f"artifact://memory/{uuid4()}"
        self._values[uri] = value
        return ArtifactRef(uri=uri, media_type=media_type)

    def get(self, reference: ArtifactRef) -> Any:
        return self._values[reference.uri]
