"""
Vertex AI Gemini — foundation model client.

Provides:
- Text generation (Gemini 2.0 Flash, Gemini 2.0 Pro, Gemini 1.5 Pro/Flash)
- Streaming generation
- Multi-modal: image, video, audio, PDF + text
- Embeddings (text-embedding-004)
- Grounding with Google Search
- Function calling / tool use
- Structured JSON output
- Token counting and cost estimation
- Automatic retry with exponential backoff
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import vertexai
from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    GroundingChunk,
    HarmBlockThreshold,
    HarmCategory,
    Image,
    Part,
    SafetySetting,
    Tool,
    ToolConfig,
    grounding,
)
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from src.utils.config import get_config, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Cost per 1M tokens (USD) — australia-southeast1 pricing
_TOKEN_COSTS: dict[str, dict[str, float]] = {
    "gemini-2.0-flash-001":     {"input": 0.075, "output": 0.30},
    "gemini-2.0-pro-001":       {"input": 1.25,  "output": 5.00},
    "gemini-1.5-pro-002":       {"input": 1.25,  "output": 5.00},
    "gemini-1.5-flash-002":     {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-8b-001":  {"input": 0.0375, "output": 0.15},
}

_DEFAULT_SAFETY = [
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
]


@dataclass
class GeminiResult:
    """Structured result from a Gemini model invocation."""

    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = ""
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    grounding_chunks: list[dict[str, Any]] = field(default_factory=list)
    function_calls: list[dict[str, Any]] = field(default_factory=list)
    raw_response: Any = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def is_grounded(self) -> bool:
        return len(self.grounding_chunks) > 0


@dataclass
class EmbeddingResult:
    """Structured result from a Vertex AI embedding call."""

    embedding: list[float]
    model: str
    token_count: int = 0

    @property
    def dimensions(self) -> int:
        return len(self.embedding)


class GeminiClient:
    """
    Production-grade Vertex AI Gemini client.

    Supports all Gemini modalities: text, image, video, audio, PDF.
    Includes grounding with Google Search, function calling,
    structured output, and cost tracking.

    Example:
        client = GeminiClient(model_key="gemini_2_flash")

        # Text generation
        result = client.generate("Explain Vertex AI Agent Builder in 3 sentences.")
        print(result.text)

        # Multi-modal with image
        result = client.generate_with_media(
            "Describe the architecture shown in this diagram.",
            media_paths=[Path("architecture.png")],
        )

        # Grounded generation with Google Search
        result = client.generate_grounded(
            "What are the latest Vertex AI announcements?",
            use_google_search=True,
        )
        print(result.grounding_chunks)

        # Embeddings
        emb = client.embed("Vertex AI provides a unified ML platform")
        print(f"Dimensions: {emb.dimensions}")
    """

    def __init__(
        self,
        model_key: str = "gemini_2_flash",
        location: str | None = None,
        project_id: str | None = None,
    ) -> None:
        cfg = get_config()
        raw = load_config()

        self._project_id = project_id or cfg.project_id
        self._location = location or cfg.vertex_ai.location
        self._model_name = cfg.get_model(model_key)
        self._inference = cfg.vertex_ai.inference

        vertexai.init(project=self._project_id, location=self._location)

        self._model = GenerativeModel(
            model_name=self._model_name,
            safety_settings=_DEFAULT_SAFETY,
        )

        logger.info(
            "GeminiClient initialised",
            model=self._model_name,
            project=self._project_id,
            location=self._location,
        )

    @retry(
        retry=retry_if_exception_type((GoogleAPICallError, ResourceExhausted)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        tools: list[Tool] | None = None,
        response_mime_type: str | None = None,
        response_schema: Any = None,
    ) -> GeminiResult:
        """
        Generate text from a Gemini model.

        Args:
            prompt: The user prompt.
            system: Optional system instruction.
            max_tokens: Override config max_output_tokens.
            temperature: Override config temperature.
            top_p: Override config top_p.
            tools: List of Vertex AI Tool objects for function calling.
            response_mime_type: "application/json" for structured output.
            response_schema: Pydantic model or dict schema for JSON output.

        Returns:
            GeminiResult with text, token counts, and cost estimate.
        """
        model = self._model
        if system:
            model = GenerativeModel(
                model_name=self._model_name,
                system_instruction=system,
                safety_settings=_DEFAULT_SAFETY,
            )

        gen_config = GenerationConfig(
            max_output_tokens=max_tokens or self._inference.get("max_output_tokens", 4096),
            temperature=temperature if temperature is not None else self._inference.get("temperature", 0.1),
            top_p=top_p or self._inference.get("top_p", 0.95),
            top_k=self._inference.get("top_k", 40),
        )

        if response_mime_type:
            gen_config = GenerationConfig(
                max_output_tokens=max_tokens or self._inference.get("max_output_tokens", 4096),
                temperature=0.0,
                response_mime_type=response_mime_type,
                response_schema=response_schema,
            )

        kwargs: dict[str, Any] = {
            "contents": [Content(role="user", parts=[Part.from_text(prompt)])],
            "generation_config": gen_config,
        }
        if tools:
            kwargs["tools"] = tools

        start = time.perf_counter()
        try:
            response = model.generate_content(**kwargs)
        except (GoogleAPICallError, ResourceExhausted) as exc:
            logger.error("Gemini generation failed", error=str(exc), model=self._model_name)
            raise
        latency_ms = (time.perf_counter() - start) * 1000

        return self._parse_response(response, latency_ms)

    def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Generator[str, None, None]:
        """
        Stream tokens from a Gemini model.

        Yields text chunks as they arrive from the model.
        """
        model = self._model
        if system:
            model = GenerativeModel(
                model_name=self._model_name,
                system_instruction=system,
                safety_settings=_DEFAULT_SAFETY,
            )

        gen_config = GenerationConfig(
            max_output_tokens=max_tokens or self._inference.get("max_output_tokens", 4096),
            temperature=temperature if temperature is not None else self._inference.get("temperature", 0.1),
        )

        stream = model.generate_content(
            [Content(role="user", parts=[Part.from_text(prompt)])],
            generation_config=gen_config,
            stream=True,
        )

        for chunk in stream:
            if chunk.candidates and chunk.candidates[0].content.parts:
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        yield part.text

    def generate_with_media(
        self,
        prompt: str,
        media_paths: list[Path] | None = None,
        gcs_uris: list[str] | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> GeminiResult:
        """
        Multi-modal generation with images, PDFs, video, or audio.

        Supports local files (auto-detected MIME type) and GCS URIs.

        Args:
            prompt: The text prompt about the media.
            media_paths: List of local file paths (images, PDFs).
            gcs_uris: List of GCS URIs (gs://bucket/file.mp4 for video).
            system: Optional system instruction.
            max_tokens: Override default max tokens.

        Returns:
            GeminiResult with text response.
        """
        parts: list[Part] = []

        for path in (media_paths or []):
            suffix = path.suffix.lower()
            if suffix in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"):
                parts.append(Part.from_image(Image.load_from_file(str(path))))
            elif suffix == ".pdf":
                pdf_bytes = path.read_bytes()
                parts.append(Part.from_data(data=pdf_bytes, mime_type="application/pdf"))
            elif suffix in (".mp4", ".mov", ".avi", ".mkv"):
                parts.append(Part.from_data(data=path.read_bytes(), mime_type="video/mp4"))
            elif suffix in (".mp3", ".wav", ".flac", ".m4a"):
                parts.append(Part.from_data(data=path.read_bytes(), mime_type="audio/mpeg"))

        for uri in (gcs_uris or []):
            ext = uri.split(".")[-1].lower()
            mime_map = {
                "mp4": "video/mp4", "mov": "video/mov",
                "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "pdf": "application/pdf",
                "mp3": "audio/mpeg", "wav": "audio/wav",
            }
            mime = mime_map.get(ext, "application/octet-stream")
            parts.append(Part.from_uri(uri=uri, mime_type=mime))

        parts.append(Part.from_text(prompt))

        model = self._model
        if system:
            model = GenerativeModel(
                model_name=self._model_name,
                system_instruction=system,
                safety_settings=_DEFAULT_SAFETY,
            )

        gen_config = GenerationConfig(
            max_output_tokens=max_tokens or self._inference.get("max_output_tokens", 4096),
            temperature=self._inference.get("temperature", 0.1),
        )

        start = time.perf_counter()
        response = model.generate_content(
            [Content(role="user", parts=parts)],
            generation_config=gen_config,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        return self._parse_response(response, latency_ms)

    def generate_grounded(
        self,
        prompt: str,
        use_google_search: bool = True,
        vertex_search_datastore: str | None = None,
        system: str | None = None,
    ) -> GeminiResult:
        """
        Generate a response grounded in Google Search or Vertex AI Search.

        Grounded responses include source citations and attribution metadata.

        Args:
            prompt: The user prompt.
            use_google_search: Ground with live Google Search results.
            vertex_search_datastore: Vertex AI Search datastore resource name
                (overrides Google Search when provided).
            system: Optional system instruction.

        Returns:
            GeminiResult with text and grounding_chunks containing source URLs.
        """
        if vertex_search_datastore:
            grounding_tool = Tool.from_retrieval(
                retrieval=grounding.Retrieval(
                    source=grounding.VertexAISearch(datastore=vertex_search_datastore)
                )
            )
        else:
            grounding_tool = Tool.from_google_search_retrieval(
                google_search_retrieval=grounding.GoogleSearchRetrieval()
            )

        model = self._model
        if system:
            model = GenerativeModel(
                model_name=self._model_name,
                system_instruction=system,
                safety_settings=_DEFAULT_SAFETY,
            )

        gen_config = GenerationConfig(
            max_output_tokens=self._inference.get("max_output_tokens", 4096),
            temperature=self._inference.get("temperature", 0.1),
        )

        start = time.perf_counter()
        response = model.generate_content(
            [Content(role="user", parts=[Part.from_text(prompt)])],
            generation_config=gen_config,
            tools=[grounding_tool],
        )
        latency_ms = (time.perf_counter() - start) * 1000

        result = self._parse_response(response, latency_ms)

        # Extract grounding metadata
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
                gm = candidate.grounding_metadata
                for chunk in (gm.grounding_chunks or []):
                    if hasattr(chunk, "web") and chunk.web:
                        result.grounding_chunks.append({
                            "uri": chunk.web.uri,
                            "title": chunk.web.title,
                        })

        logger.info(
            "Grounded generation complete",
            model=self._model_name,
            grounding_chunks=len(result.grounding_chunks),
            use_google_search=use_google_search,
        )

        return result

    def generate_with_tools(
        self,
        prompt: str,
        function_declarations: list[dict[str, Any]],
        system: str | None = None,
        auto_invoke: bool = False,
    ) -> GeminiResult:
        """
        Generate with function calling (tool use).

        Args:
            prompt: The user prompt.
            function_declarations: List of function schema dicts with
                name, description, and parameters keys.
            system: Optional system instruction.
            auto_invoke: Not implemented — return function call spec for caller to execute.

        Returns:
            GeminiResult with function_calls populated if the model chose to call a function.
        """
        tools = [
            Tool(function_declarations=[
                FunctionDeclaration(
                    name=fn["name"],
                    description=fn["description"],
                    parameters=fn.get("parameters", {}),
                )
                for fn in function_declarations
            ])
        ]

        return self.generate(prompt=prompt, system=system, tools=tools)

    def embed(
        self,
        text: str,
        model_key: str = "text_embed_004",
        task_type: str = "RETRIEVAL_DOCUMENT",
        output_dimensionality: int | None = None,
    ) -> EmbeddingResult:
        """
        Generate a vector embedding using Vertex AI text embedding models.

        Args:
            text: Text to embed.
            model_key: Config key for the embedding model.
            task_type: RETRIEVAL_DOCUMENT | RETRIEVAL_QUERY | SEMANTIC_SIMILARITY |
                       CLASSIFICATION | CLUSTERING | QUESTION_ANSWERING | FACT_VERIFICATION
            output_dimensionality: Reduced dimensions (text-embedding-004: max 3072,
                                   textembedding-gecko: 768).

        Returns:
            EmbeddingResult with the embedding vector.
        """
        cfg = get_config()
        model_name = cfg.get_model(model_key)

        embed_model = TextEmbeddingModel.from_pretrained(model_name)

        inputs = [TextEmbeddingInput(text=text, task_type=task_type)]
        kwargs: dict[str, Any] = {}
        if output_dimensionality:
            kwargs["output_dimensionality"] = output_dimensionality

        try:
            embeddings = embed_model.get_embeddings(inputs, **kwargs)
        except GoogleAPICallError as exc:
            logger.error("Embedding failed", error=str(exc))
            raise

        emb = embeddings[0]
        token_count = getattr(emb, "statistics", {})
        if hasattr(token_count, "token_count"):
            token_count = token_count.token_count
        else:
            token_count = 0

        return EmbeddingResult(
            embedding=emb.values,
            model=model_name,
            token_count=token_count,
        )

    def embed_batch(
        self,
        texts: list[str],
        model_key: str = "text_embed_004",
        task_type: str = "RETRIEVAL_DOCUMENT",
        batch_size: int = 250,
    ) -> list[EmbeddingResult]:
        """Embed a list of texts in batches (API limit: 250 per call)."""
        cfg = get_config()
        model_name = cfg.get_model(model_key)
        embed_model = TextEmbeddingModel.from_pretrained(model_name)

        results: list[EmbeddingResult] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = [TextEmbeddingInput(text=t, task_type=task_type) for t in batch]
            embeddings = embed_model.get_embeddings(inputs)
            for emb in embeddings:
                results.append(EmbeddingResult(embedding=emb.values, model=model_name))
            logger.debug("Embedding batch", batch=i // batch_size + 1, size=len(batch))

        return results

    def count_tokens(self, prompt: str) -> int:
        """Count tokens in a prompt without generating a response."""
        response = self._model.count_tokens([Content(role="user", parts=[Part.from_text(prompt)])])
        return response.total_tokens

    def json_output(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        system: str | None = None,
    ) -> dict[str, Any]:
        """
        Request structured JSON output using Gemini's controlled generation.

        Args:
            prompt: The user prompt.
            response_schema: JSON schema dict describing the expected output.
            system: Optional system instruction.

        Returns:
            Parsed JSON dict matching the schema.
        """
        import json

        result = self.generate(
            prompt=prompt,
            system=system,
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        try:
            return json.loads(result.text)
        except Exception as exc:
            logger.error("JSON output parse failed", error=str(exc))
            raise ValueError(f"Model returned invalid JSON: {result.text[:200]}") from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_response(self, response: Any, latency_ms: float) -> GeminiResult:
        """Parse raw Vertex AI response into a GeminiResult."""
        text = ""
        function_calls: list[dict[str, Any]] = []

        if response.candidates:
            candidate = response.candidates[0]
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
                if hasattr(part, "function_call") and part.function_call:
                    function_calls.append({
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args),
                    })

        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

        costs = _TOKEN_COSTS.get(self._model_name, {"input": 0.0, "output": 0.0})
        cost = (input_tokens / 1_000_000 * costs["input"]) + (output_tokens / 1_000_000 * costs["output"])

        finish_reason = ""
        if response.candidates:
            finish_reason = str(response.candidates[0].finish_reason)

        result = GeminiResult(
            text=text,
            model=self._model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
            function_calls=function_calls,
            raw_response=response,
        )

        logger.info(
            "Gemini generation complete",
            model=self._model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=f"{latency_ms:.0f}",
            cost_usd=f"{cost:.6f}",
        )

        return result
