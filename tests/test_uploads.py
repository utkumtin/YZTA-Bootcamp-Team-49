from __future__ import annotations

from pareto.cleaning.uploads import uploaded_file_identity


class _UploadedFile:
    def __init__(self, file_id: str = "streamlit-file-id") -> None:
        self.file_id = file_id


def test_uploaded_file_identity_uses_streamlit_file_id() -> None:
    uploaded = _UploadedFile(file_id="abc123")

    assert uploaded_file_identity(uploaded) == "id:abc123"


def test_uploaded_file_identity_distinguishes_different_file_ids() -> None:
    first = _UploadedFile(file_id="file-a")
    second = _UploadedFile(file_id="file-b")

    assert uploaded_file_identity(first) != uploaded_file_identity(second)


def test_uploaded_file_identity_is_idempotent_for_same_file_id() -> None:
    first = _UploadedFile(file_id="same-id")
    second = _UploadedFile(file_id="same-id")

    assert uploaded_file_identity(first) == uploaded_file_identity(second)
