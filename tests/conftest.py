import os
import sys
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from reranker_service import RerankerService


@pytest.fixture(autouse=True)
def auto_mock_reranker_for_offline_tests(request):
    """
    Ensures unit tests running in environments without the physical 396MB GGUF
    reranker file on disk succeed hermetically.
    Tests verifying the mandatory error (with 'mandatory_raises_when_missing' in name)
    are skipped from this auto-mocking so their error assertions remain authentic.
    """
    # Do not mock when running reranker service or note rag pool test suites
    nodeid = request.node.nodeid
    if (
        "test_reranker_service" in nodeid
        or "test_note_rag_pool" in nodeid
        or "mandatory_raises_when_missing" in nodeid
    ):
        yield
        return

    with patch.object(RerankerService, "is_model_available", return_value=True), \
         patch.object(
             RerankerService,
             "score_candidates",
             side_effect=lambda query, candidates, instruction=None: [
                 (0.99 - 0.01 * i, c) for i, c in enumerate(candidates)
             ]
         ):
        yield
