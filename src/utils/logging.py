"""
Structured logging for Google AI Platform Engineering.
Integrates with Google Cloud Logging and Cloud Trace.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(
    level: str = "INFO",
    json_output: bool = False,
    service_name: str = "google-ai-platform",
    gcp_project_id: str | None = None,
) -> None:
    """
    Configure structlog with optional Cloud Logging export.

    Args:
        level: Log level string.
        json_output: JSON lines for Cloud Logging. False = coloured console.
        service_name: Stamped on every log record.
        gcp_project_id: If provided, configure Cloud Logging handler.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.CallsiteParameterAdder([
            structlog.processors.CallsiteParameter.FILENAME,
            structlog.processors.CallsiteParameter.FUNC_NAME,
            structlog.processors.CallsiteParameter.LINENO,
        ]),
    ]

    renderer = structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(getattr(logging, level.upper()))

    for noisy in ("google", "urllib3", "grpc", "proto"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if gcp_project_id:
        _configure_cloud_logging(gcp_project_id)


def _configure_cloud_logging(project_id: str) -> None:
    """Attach Google Cloud Logging handler."""
    try:
        import google.cloud.logging
        client = google.cloud.logging.Client(project=project_id)
        client.setup_logging()
    except ImportError:
        logging.getLogger(__name__).warning("google-cloud-logging not installed")


def get_logger(name: str, **context: Any) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name).bind(**context)


configure_logging()
