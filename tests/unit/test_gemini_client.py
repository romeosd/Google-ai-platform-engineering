"""
Unit tests for GeminiClient.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.models.gemini_client import EmbeddingResult, GeminiClient, GeminiResult


class TestGeminiClient:

    def _mock_config(self) -> MagicMock:
        cfg = MagicMock()
        cfg.project_id = "test-project"
        cfg.vertex_ai.location = "australia-southeast1"
        cfg.vertex_ai.staging_bucket = "gs://test-bucket/staging"
        cfg.vertex_ai.inference = {
            "max_output_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
        }
        cfg.get_model.side_effect = lambda key: {
            "gemini_2_flash": "gemini-2.0-flash-001",
            "gemini_2_pro": "gemini-2.0-pro-001",
            "text_embed_004": "text-embedding-004",
        }.get(key, "gemini-2.0-flash-001")
        return cfg

    @patch("src.models.gemini_client.get_config")
    @patch("src.models.gemini_client.load_config")
    @patch("src.models.gemini_client.vertexai")
    @patch("src.models.gemini_client.GenerativeModel")
    def test_generate_returns_gemini_result(
        self, mock_model_cls: MagicMock, mock_vertexai: MagicMock,
        mock_load: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_get.return_value = self._mock_config()
        mock_load.return_value = {}

        mock_part = MagicMock()
        mock_part.text = "Vertex AI is a unified ML platform."
        mock_part.function_call = None

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_candidate.finish_reason = "STOP"
        mock_candidate.grounding_metadata = None

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 15
        mock_usage.candidates_token_count = 10

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = mock_usage

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_model_cls.return_value = mock_model

        client = GeminiClient()
        result = client.generate("What is Vertex AI?")

        assert isinstance(result, GeminiResult)
        assert result.text == "Vertex AI is a unified ML platform."
        assert result.input_tokens == 15
        assert result.output_tokens == 10
        assert result.total_tokens == 25
        assert result.estimated_cost_usd >= 0

    @patch("src.models.gemini_client.get_config")
    @patch("src.models.gemini_client.load_config")
    @patch("src.models.gemini_client.vertexai")
    @patch("src.models.gemini_client.GenerativeModel")
    def test_generate_stream_yields_chunks(
        self, mock_model_cls: MagicMock, mock_vertexai: MagicMock,
        mock_load: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_get.return_value = self._mock_config()
        mock_load.return_value = {}

        chunks = []
        for text in ["Hello", " Gemini", "!"]:
            p = MagicMock()
            p.text = text
            c = MagicMock()
            c.content.parts = [p]
            chunk = MagicMock()
            chunk.candidates = [c]
            chunks.append(chunk)

        empty_chunk = MagicMock()
        empty_chunk.candidates = []
        chunks.append(empty_chunk)

        mock_model = MagicMock()
        mock_model.generate_content.return_value = iter(chunks)
        mock_model_cls.return_value = mock_model

        client = GeminiClient()
        result = "".join(client.generate_stream("Say hello"))
        assert result == "Hello Gemini!"

    def test_gemini_result_total_tokens(self) -> None:
        result = GeminiResult(text="test", model="gemini-2.0-flash-001", input_tokens=50, output_tokens=25)
        assert result.total_tokens == 75

    def test_gemini_result_is_grounded_false_when_no_chunks(self) -> None:
        result = GeminiResult(text="test", model="gemini-2.0-flash-001", grounding_chunks=[])
        assert result.is_grounded is False

    def test_gemini_result_is_grounded_true_with_chunks(self) -> None:
        result = GeminiResult(
            text="test", model="gemini-2.0-flash-001",
            grounding_chunks=[{"uri": "https://example.com", "title": "Source"}]
        )
        assert result.is_grounded is True

    def test_embedding_result_dimensions(self) -> None:
        result = EmbeddingResult(embedding=[0.1] * 768, model="text-embedding-004")
        assert result.dimensions == 768
