"""
Vertex AI Agent Builder — RAG and search client.

Provides:
- Vertex AI Search (enterprise document search)
- Agent Builder RAG Engine (managed RAG)
- Data store management and document ingestion
- Grounded answer generation with citations
- Hybrid (semantic + keyword) retrieval
- Conversation management
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from google.cloud import discoveryengine_v1 as discoveryengine
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPICallError

from src.utils.config import get_config, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchHit:
    """A single search result from Vertex AI Search."""

    id: str
    content: str
    score: float = 0.0
    title: str = ""
    uri: str = ""
    snippet: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGAnswer:
    """A grounded answer from Vertex AI Agent Builder."""

    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    search_hits: list[SearchHit] = field(default_factory=list)
    session_id: str = ""

    @property
    def source_uris(self) -> list[str]:
        return [c.get("uri", "") for c in self.citations if c.get("uri")]

    def format_with_citations(self) -> str:
        if not self.citations:
            return self.answer
        sources = "\n".join(f"[{i+1}] {c.get('uri', c.get('title', 'unknown'))}" for i, c in enumerate(self.citations))
        return f"{self.answer}\n\n**Sources:**\n{sources}"


class AgentBuilderClient:
    """
    Production Vertex AI Agent Builder client.

    Provides enterprise search and grounded RAG over unstructured documents,
    websites, and BigQuery data via Vertex AI Search data stores.

    Example:
        ab = AgentBuilderClient()

        # Search documents
        hits = ab.search("What is the data residency policy for PII?", top_k=10)
        for hit in hits:
            print(f"[{hit.score:.3f}] {hit.title}: {hit.snippet}")

        # Full grounded RAG answer with citations
        answer = ab.answer_query("Summarise our APRA compliance obligations")
        print(answer.format_with_citations())
    """

    def __init__(
        self,
        project_id: str | None = None,
        location: str | None = None,
        data_store_id: str | None = None,
        search_app_id: str | None = None,
    ) -> None:
        cfg = get_config()
        raw = load_config()
        ab_cfg = raw.get("agent_builder", {})

        self._project_id = project_id or cfg.project_id
        self._location = location or ab_cfg.get("location", "global")
        self._data_store_id = data_store_id or ab_cfg.get("data_stores", {}).get("documents", "")
        self._search_app_id = search_app_id or ab_cfg.get("apps", {}).get("search_app_id", "")
        self._retrieval = ab_cfg.get("retrieval", {})

        api_endpoint = f"{self._location}-discoveryengine.googleapis.com" if self._location != "global" else None
        client_options = ClientOptions(api_endpoint=api_endpoint) if api_endpoint else None

        self._search_client = discoveryengine.SearchServiceClient(client_options=client_options)
        self._conv_client = discoveryengine.ConversationalSearchServiceClient(client_options=client_options)
        self._doc_client = discoveryengine.DocumentServiceClient(client_options=client_options)

        logger.info(
            "AgentBuilderClient initialised",
            project=self._project_id,
            location=self._location,
            data_store=self._data_store_id,
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filter_expr: str | None = None,
        search_type: str = "CONTENT_SEARCH",
    ) -> list[SearchHit]:
        """
        Search documents in a Vertex AI Search data store.

        Args:
            query: The search query.
            top_k: Number of results to return.
            filter_expr: OData filter expression.
            search_type: "CONTENT_SEARCH" | "WEBSITE_SEARCH"

        Returns:
            List of SearchHit sorted by relevance score.
        """
        k = top_k or self._retrieval.get("max_results", 10)
        serving_config = (
            f"projects/{self._project_id}/locations/{self._location}/"
            f"collections/default_collection/engines/{self._search_app_id}/"
            f"servingConfigs/default_serving_config"
        )

        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=query,
            page_size=k,
            content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
                snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                    return_snippet=True,
                    max_snippet_count=3,
                ),
                summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                    summary_result_count=k,
                    include_citations=True,
                    ignore_adversarial_query=True,
                    ignore_non_summary_seeking_query=False,
                ),
                extractive_content_spec=discoveryengine.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                    max_extractive_answers_count=3,
                    max_extractive_segments_count=5,
                ),
            ),
            filter=filter_expr or "",
        )

        try:
            response = self._search_client.search(request)
        except GoogleAPICallError as exc:
            logger.error("Vertex AI Search failed", error=str(exc), query=query[:80])
            raise

        hits: list[SearchHit] = []
        for result in response.results:
            doc = result.document
            derived = doc.derived_struct_data if doc.derived_struct_data else {}

            snippets = derived.get("snippets", [])
            snippet_text = snippets[0].get("snippet", "") if snippets else ""

            hits.append(SearchHit(
                id=doc.id,
                content=snippet_text,
                score=result.relevance_score if hasattr(result, "relevance_score") else 0.0,
                title=derived.get("title", ""),
                uri=derived.get("link", ""),
                snippet=snippet_text,
                metadata=dict(derived),
            ))

        logger.info(
            "Agent Builder search complete",
            query_snippet=query[:60],
            results=len(hits),
        )

        return hits

    def answer_query(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int | None = None,
    ) -> RAGAnswer:
        """
        Get a grounded, cited answer from Vertex AI Agent Builder.

        Uses the Answer Generation API which retrieves relevant chunks
        and generates a grounded response with inline citations.

        Args:
            query: The user's question.
            session_id: Session ID for multi-turn conversation context.
            top_k: Number of documents to retrieve for grounding.

        Returns:
            RAGAnswer with the generated answer and source citations.
        """
        k = top_k or self._retrieval.get("max_results", 10)
        serving_config = (
            f"projects/{self._project_id}/locations/{self._location}/"
            f"collections/default_collection/engines/{self._search_app_id}/"
            f"servingConfigs/default_serving_config"
        )

        query_obj = discoveryengine.Query(text=query)

        session_spec = None
        if session_id:
            session_spec = discoveryengine.AnswerQueryRequest.SessionSpec(
                session=session_id
            )

        request = discoveryengine.AnswerQueryRequest(
            serving_config=serving_config,
            query=query_obj,
            session=session_id,
            answer_generation_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec(
                model_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec.ModelSpec(
                    model_version="gemini-1.5-flash-001/answer_gen/v2",
                ),
                prompt_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec.PromptSpec(
                    preamble="You are a helpful enterprise assistant. Answer concisely and cite sources."
                ),
                include_citations=True,
                answer_language_code="en",
            ),
            search_spec=discoveryengine.AnswerQueryRequest.SearchSpec(
                search_params=discoveryengine.AnswerQueryRequest.SearchSpec.SearchParams(
                    max_return_results=k,
                    filter=self._retrieval.get("filter", ""),
                )
            ),
            query_understanding_spec=discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec(
                query_rephraser_spec=discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec.QueryRephraserSpec(
                    disable=False,
                    max_rephrase_steps=2,
                )
            ),
        )

        try:
            response = self._conv_client.answer_query(request)
        except GoogleAPICallError as exc:
            logger.error("Answer query failed", error=str(exc))
            raise

        answer_text = ""
        citations: list[dict[str, Any]] = []
        search_hits: list[SearchHit] = []
        new_session_id = ""

        if response.answer:
            answer_text = response.answer.answer_text

            for citation in response.answer.citations:
                sources = []
                for ref in citation.sources:
                    ref_idx = ref.reference_id
                    if ref_idx and response.answer.references:
                        try:
                            ref_obj = response.answer.references[int(ref_idx)]
                            chunk = ref_obj.chunk_info
                            doc_meta = ref_obj.unstructured_document_info
                            sources.append({
                                "uri": doc_meta.uri if doc_meta else "",
                                "title": doc_meta.title if doc_meta else "",
                                "content": chunk.content if chunk else "",
                            })
                        except (IndexError, ValueError):
                            pass
                citations.extend(sources)

        if response.session:
            new_session_id = response.session.name

        logger.info(
            "Answer query complete",
            query_snippet=query[:60],
            answer_length=len(answer_text),
            citations=len(citations),
        )

        return RAGAnswer(
            answer=answer_text,
            citations=citations,
            search_hits=search_hits,
            session_id=new_session_id,
        )

    def ingest_documents_gcs(
        self,
        gcs_uri: str,
        data_store_id: str | None = None,
    ) -> str:
        """
        Trigger GCS import into a Vertex AI Search data store.

        Args:
            gcs_uri: GCS URI of documents (gs://bucket/prefix/ or gs://bucket/file.jsonl).
            data_store_id: Target data store ID (overrides config).

        Returns:
            Operation name for tracking the import job.
        """
        ds_id = data_store_id or self._data_store_id
        parent = (
            f"projects/{self._project_id}/locations/{self._location}/"
            f"collections/default_collection/dataStores/{ds_id}/branches/default_branch"
        )

        request = discoveryengine.ImportDocumentsRequest(
            parent=parent,
            gcs_source=discoveryengine.GcsSource(
                input_uris=[gcs_uri],
                data_schema="content",
            ),
            reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL,
        )

        operation = self._doc_client.import_documents(request=request)
        logger.info("GCS document import started", gcs_uri=gcs_uri, operation=operation.operation.name)
        return operation.operation.name
