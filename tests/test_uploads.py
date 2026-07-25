from __future__ import annotations

import hashlib

from pareto.cleaning.uploads import uploaded_file_identity


class _UploadedFile:
    def __init__(
        self,
        *,
        file_id: str | None = None,
        name: str = "input.csv",
        size: int = 10,
        content: bytes = b"test-value",
    ) -> None:
        self.file_id = file_id
        self.name = name
        self.size = size
        self._content = content
        self.getvalue_calls = 0

    def getvalue(self) -> bytes:
        self.getvalue_calls += 1
        return self._content


def test_uploaded_file_identity_prefers_streamlit_file_id() -> None:
    uploaded = _UploadedFile(file_id="streamlit-file-id")

    assert uploaded_file_identity(uploaded) == "streamlit-file-id"
    assert uploaded.getvalue_calls == 0


def test_uploaded_file_identity_falls_back_to_name_size_and_content_hash() -> None:
    uploaded = _UploadedFile(file_id=None, name="input.csv", size=10, content=b"test-value")

    assert uploaded_file_identity(uploaded) == (
        "input.csv",
        10,
        hashlib.sha256(b"test-value").hexdigest(),
    )


def test_content_hash_fallback_distinguishes_same_name_and_size() -> None:
    first = _UploadedFile(name="input.csv", size=10, content=b"first-data")
    second = _UploadedFile(name="input.csv", size=10, content=b"other-data")

    assert uploaded_file_identity(first) != uploaded_file_identity(second)


def test_content_hash_fallback_is_idempotent() -> None:
    first = _UploadedFile(name="input.csv", size=10, content=b"same-data")
    second = _UploadedFile(name="input.csv", size=10, content=b"same-data")

    assert uploaded_file_identity(first) == uploaded_file_identity(second)
