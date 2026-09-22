from app.services.retrieval_service import retrieve_relevant_chunks


def test_retrieve_relevant_chunks():
    """Test retrieval using a query related to an ingested document."""

    results = retrieve_relevant_chunks(
        query="What information is in the document?",
        limit=3,
    )

    assert isinstance(results, list)
    assert len(results) <= 3

    if results:
        assert "text" in results[0]
        assert "document_id" in results[0]
        assert "filename" in results[0]
        assert "score" in results[0]