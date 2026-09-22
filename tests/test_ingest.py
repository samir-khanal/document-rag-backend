from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_ingest_rejects_unsupported_extension():
    """
    A file with a disallowed extension (e.g. .csv) should be rejected
    with 400 before any extraction or storage happens.
    """
    response = client.post(
        "/documents/ingest",
        files={"file": ("data.csv", b"a,b,c\n1,2,3", "text/csv")},
        data={"chunking_strategy": "recursive"},
    )

    assert response.status_code == 400
    detail = response.json()["detail"].lower()
    assert "pdf" in detail and "txt" in detail


def test_ingest_rejects_empty_document():
    """
    An empty TXT file should be rejected. The endpoint checks content
    length before embedding or storing, so no side effects occur.
    """
    response = client.post(
        "/documents/ingest",
        files={"file": ("empty.txt", b"", "text/plain")},
        data={"chunking_strategy": "recursive"},
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_ingest_rejects_whitespace_only_txt():
    """
    A TXT file with only whitespace should be rejected — the chunker
    would otherwise produce zero usable chunks and the endpoint raises
    'Document does not contain usable text.'
    """
    response = client.post(
        "/documents/ingest",
        files={"file": ("blank.txt", b"   \n\n   \t  ", "text/plain")},
        data={"chunking_strategy": "recursive"},
    )

    assert response.status_code == 400