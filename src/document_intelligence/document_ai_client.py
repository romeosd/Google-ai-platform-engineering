"""
Google Cloud Document AI — intelligent document processing.

Provides:
- Layout parser (text, tables, form fields, entities)
- Form parser (key-value extraction)
- Invoice parser (structured billing data)
- Custom document extractor (trained on domain docs)
- Batch processing via GCS
- Async processing for large documents
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPICallError
from google.cloud import documentai

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentTable:
    """A table extracted from a document."""

    page: int
    rows: list[list[str]] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_count(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    def to_markdown(self) -> str:
        if not self.rows:
            return ""
        header = "| " + " | ".join(self.rows[0]) + " |"
        sep = "| " + " | ".join(["---"] * len(self.rows[0])) + " |"
        body = "\n".join("| " + " | ".join(r) + " |" for r in self.rows[1:])
        return "\n".join([header, sep, body]) if body else "\n".join([header, sep])


@dataclass
class FormField:
    """A key-value field from a form document."""

    name: str
    value: str
    confidence: float = 0.0

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.8


@dataclass
class DocumentAIResult:
    """Aggregated result from Document AI processing."""

    processor_id: str
    page_count: int = 0
    raw_text: str = ""
    tables: list[DocumentTable] = field(default_factory=list)
    form_fields: list[FormField] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    raw_document: Any = None

    @property
    def table_count(self) -> int:
        return len(self.tables)

    def get_field(self, name: str, default: str = "") -> str:
        for f in self.form_fields:
            if f.name.lower() == name.lower():
                return f.value
        return default

    def get_entity(self, entity_type: str) -> list[dict[str, Any]]:
        return [e for e in self.entities if e.get("type") == entity_type]

    def tables_as_markdown(self) -> str:
        return "\n\n".join(
            f"### Table {i+1} (Page {t.page})\n{t.to_markdown()}"
            for i, t in enumerate(self.tables)
        )


class DocumentAIClient:
    """
    Production Google Cloud Document AI client.

    Handles layout, form, invoice, and custom model processing
    for both synchronous (single docs) and batch (GCS) workloads.

    Example:
        doc_ai = DocumentAIClient()

        # Process a local PDF
        result = doc_ai.process_document(
            Path("invoice.pdf"),
            processor_key="invoice_parser",
        )
        print(f"Invoice total: {result.get_entity('total_amount')}")
        print(result.tables_as_markdown())

        # Batch process from GCS
        operation = doc_ai.batch_process_gcs(
            gcs_input_uri="gs://bucket/documents/",
            gcs_output_uri="gs://bucket/output/",
            processor_key="layout_parser",
        )
    """

    def __init__(self) -> None:
        raw = load_config()
        di_cfg = raw.get("document_ai", {})

        self._project_id = raw.get("gcp", {}).get("project_id", "")
        self._location = di_cfg.get("location", "us")
        self._processors = di_cfg.get("processors", {})

        # Document AI only available in us and eu
        api_endpoint = f"{self._location}-documentai.googleapis.com"
        client_options = ClientOptions(api_endpoint=api_endpoint)

        self._client = documentai.DocumentProcessorServiceClient(client_options=client_options)

        logger.info(
            "DocumentAIClient initialised",
            project=self._project_id,
            location=self._location,
        )

    def process_document(
        self,
        file_path: Path,
        processor_key: str = "layout_parser",
        processor_id: str | None = None,
    ) -> DocumentAIResult:
        """
        Process a single document with the specified Document AI processor.

        Args:
            file_path: Path to the document (PDF, JPEG, PNG, TIFF, GIF, BMP, WebP).
            processor_key: Config key for the processor (layout_parser, form_parser, etc.)
            processor_id: Direct processor ID (overrides processor_key).

        Returns:
            DocumentAIResult with extracted text, tables, fields, and entities.
        """
        pid = processor_id or self._processors.get(processor_key, "")
        if not pid:
            raise ValueError(f"Processor '{processor_key}' not configured. Add its ID to gcp_config.yaml.")

        processor_name = (
            f"projects/{self._project_id}/locations/{self._location}/processors/{pid}"
        )

        file_bytes = file_path.read_bytes()
        suffix = file_path.suffix.lower()
        mime_map = {
            ".pdf": "application/pdf",
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".tiff": "image/tiff", ".tif": "image/tiff",
            ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
        }
        mime_type = mime_map.get(suffix, "application/pdf")

        raw_document = documentai.RawDocument(content=file_bytes, mime_type=mime_type)
        request = documentai.ProcessRequest(name=processor_name, raw_document=raw_document)

        try:
            result = self._client.process_document(request=request)
        except GoogleAPICallError as exc:
            logger.error("Document AI processing failed", error=str(exc), processor=pid)
            raise

        doc = result.document
        parsed = self._parse_document(doc, pid)

        logger.info(
            "Document AI processing complete",
            processor=processor_key,
            pages=parsed.page_count,
            tables=parsed.table_count,
            fields=len(parsed.form_fields),
            entities=len(parsed.entities),
        )

        return parsed

    def batch_process_gcs(
        self,
        gcs_input_uri: str,
        gcs_output_uri: str,
        processor_key: str = "layout_parser",
        processor_id: str | None = None,
    ) -> str:
        """
        Batch process documents from GCS using Document AI.

        Processes all documents under the input GCS prefix asynchronously.
        Results are written to the output GCS prefix as JSON.

        Args:
            gcs_input_uri: GCS URI prefix containing input documents.
            gcs_output_uri: GCS URI prefix for output results.
            processor_key: Config key for the processor.
            processor_id: Direct processor ID.

        Returns:
            Long-running operation name for tracking.
        """
        pid = processor_id or self._processors.get(processor_key, "")
        processor_name = (
            f"projects/{self._project_id}/locations/{self._location}/processors/{pid}"
        )

        gcs_documents = documentai.GcsDocuments(
            documents=[documentai.GcsDocument(gcs_uri=gcs_input_uri, mime_type="application/pdf")]
        )

        input_config = documentai.BatchDocumentsInputConfig(gcs_documents=gcs_documents)
        output_config = documentai.DocumentOutputConfig(
            gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
                gcs_uri=gcs_output_uri
            )
        )

        request = documentai.BatchProcessRequest(
            name=processor_name,
            input_documents=input_config,
            document_output_config=output_config,
        )

        try:
            operation = self._client.batch_process_documents(request=request)
        except GoogleAPICallError as exc:
            logger.error("Batch Document AI failed", error=str(exc))
            raise

        op_name = operation.operation.name
        logger.info("Batch Document AI job started", operation=op_name, input=gcs_input_uri)
        return op_name

    def wait_for_batch(self, operation_name: str, timeout_seconds: int = 1800) -> bool:
        """
        Wait for a batch Document AI operation to complete.

        Args:
            operation_name: The operation name from batch_process_gcs.
            timeout_seconds: Max wait time (default 30 min).

        Returns:
            True if completed successfully.
        """
        from google.longrunning import operations_pb2
        from google.protobuf import empty_pb2

        elapsed = 0
        while elapsed < timeout_seconds:
            operation = self._client._transport.operations_client.get_operation(
                name=operation_name
            )
            if operation.done:
                if operation.HasField("error"):
                    raise RuntimeError(f"Batch operation failed: {operation.error.message}")
                logger.info("Batch Document AI complete", operation=operation_name)
                return True
            time.sleep(15)
            elapsed += 15

        raise TimeoutError(f"Batch operation {operation_name} timed out after {timeout_seconds}s")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_document(self, document: documentai.Document, processor_id: str) -> DocumentAIResult:
        """Convert a raw Document AI Document into a DocumentAIResult."""
        raw_text = document.text or ""
        page_count = len(document.pages)

        def get_text(layout: Any) -> str:
            if not layout or not layout.text_anchor.text_segments:
                return ""
            text = ""
            for seg in layout.text_anchor.text_segments:
                start = int(seg.start_index) if seg.start_index else 0
                end = int(seg.end_index) if seg.end_index else 0
                text += raw_text[start:end]
            return text.strip()

        # Tables
        tables: list[DocumentTable] = []
        for page in document.pages:
            for table in page.tables:
                rows: list[list[str]] = []
                for row in list(table.header_rows) + list(table.body_rows):
                    rows.append([get_text(cell.layout) for cell in row.cells])
                tables.append(DocumentTable(page=page.page_number, rows=rows))

        # Form fields
        form_fields: list[FormField] = []
        for page in document.pages:
            for field in page.form_fields:
                name = get_text(field.field_name)
                value = get_text(field.field_value)
                confidence = field.field_value.confidence if field.field_value else 0.0
                if name:
                    form_fields.append(FormField(name=name, value=value, confidence=confidence))

        # Entities (from specialized processors)
        entities: list[dict[str, Any]] = []
        for entity in document.entities:
            entities.append({
                "type": entity.type_,
                "mention_text": entity.mention_text,
                "normalized_value": str(entity.normalized_value.text) if entity.normalized_value else "",
                "confidence": entity.confidence,
                "properties": [
                    {"type": p.type_, "mention_text": p.mention_text}
                    for p in entity.properties
                ],
            })

        # Paragraphs
        paragraphs: list[str] = []
        for page in document.pages:
            for para in page.paragraphs:
                text = get_text(para.layout)
                if text:
                    paragraphs.append(text)

        return DocumentAIResult(
            processor_id=processor_id,
            page_count=page_count,
            raw_text=raw_text,
            tables=tables,
            form_fields=form_fields,
            entities=entities,
            paragraphs=paragraphs,
            raw_document=document,
        )
