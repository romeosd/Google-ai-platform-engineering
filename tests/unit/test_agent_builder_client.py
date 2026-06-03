"""
Unit tests for AgentBuilderClient.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.rag.agent_builder_client import AgentBuilderClient, RAGAnswer, SearchHit


class TestAgentBuilderClient:

    @patch("src.rag.agent_builder_client.get_config")
    @patch("src.rag.agent_builder_client.load_config")
    @patch("src.rag.agent_builder_client.discoveryengine.SearchServiceClient")
    @patch("src.rag.agent_builder_client.discoveryengine.ConversationalSearchServiceClient")
    @patch("src.rag.agent_builder_client.discoveryengine.DocumentServiceClient")
    def _make_client(
        self, mock_doc_cls, mock_conv_cls, mock_search_cls, mock_load, mock_get,
        project="test-project", location="global", ds_id="ds-123", app_id="app-456"
    ) -> AgentBuilderClient:
        cfg = MagicMock()
        cfg.project_id = project
        mock_get.return_value = cfg
        mock_load.return_value = {
            "agent_builder": {
                "location": location,
                "data_stores": {"documents": ds_id},
                "apps": {"search_app_id": app_id, "chat_app_id": "chat-789"},
                "retrieval": {"max_results": 5},
            }
        }
        return AgentBuilderClient(
            project_id=project,
            location=location,
            data_store_id=ds_id,
            search_app_id=app_id,
        )

    def test_rag_answer_source_uris(self) -> None:
        answer = RAGAnswer(
            answer="The policy requires 7 years retention.",
            citations=[
                {"uri": "gs://bucket/policy.pdf", "title": "Data Policy"},
                {"uri": "gs://bucket/guide.pdf", "title": "Compliance Guide"},
            ],
        )
        assert len(answer.source_uris) == 2
        assert "gs://bucket/policy.pdf" in answer.source_uris

    def test_rag_answer_format_with_citations(self) -> None:
        answer = RAGAnswer(
            answer="Retention is 7 years.",
            citations=[{"uri": "gs://bucket/policy.pdf", "title": "Policy"}],
        )
        formatted = answer.format_with_citations()
        assert "Retention is 7 years." in formatted
        assert "gs://bucket/policy.pdf" in formatted
        assert "Sources:" in formatted

    def test_rag_answer_no_citations_returns_plain(self) -> None:
        answer = RAGAnswer(answer="Plain answer.", citations=[])
        assert answer.format_with_citations() == "Plain answer."

    def test_search_hit_defaults(self) -> None:
        hit = SearchHit(id="doc-1", content="Some content")
        assert hit.score == 0.0
        assert hit.title == ""
        assert hit.uri == ""
        assert hit.metadata == {}
