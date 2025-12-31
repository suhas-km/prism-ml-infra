"""
Structured logging configuration using structlog.
Outputs JSON logs with timestamp, service, job_id, and level fields.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any, Optional

import structlog
from structlog.types import Processor


# Context variables for request-scoped logging
current_job_id: ContextVar[Optional[str]] = ContextVar("current_job_id", default=None)
current_service: ContextVar[str] = ContextVar("current_service", default="platform")


def add_job_context(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add job_id and service from context variables."""
    job_id = current_job_id.get()
    if job_id is not None:
        event_dict["job_id"] = job_id
    
    event_dict["service"] = current_service.get()
    return event_dict


def configure_logging(
    service_name: str = "platform",
    level: str = "INFO",
    json_output: bool = True,
) -> None:
    """
    Configure structlog for production JSON logging.
    
    Args:
        service_name: Default service name for log entries
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: If True, output JSON; otherwise human-readable
    """
    # Set the default service context
    current_service.set(service_name)
    
    # Shared processors for all log entries
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        add_job_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if json_output:
        # JSON output for production
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        # Human-readable output for development
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure stdlib logging to use structlog
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Quiet noisy loggers
    logging.getLogger("ray").setLevel(logging.WARNING)
    logging.getLogger("vllm").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Bound structlog logger
    """
    return structlog.get_logger(name)


def set_job_context(job_id: str) -> None:
    """Set the current job ID for logging context."""
    current_job_id.set(job_id)


def clear_job_context() -> None:
    """Clear the current job ID from logging context."""
    current_job_id.set(None)


def set_service_context(service: str) -> None:
    """Set the current service name for logging context."""
    current_service.set(service)


class LogContext:
    """Context manager for scoped job logging."""
    
    def __init__(self, job_id: str, service: Optional[str] = None) -> None:
        self.job_id = job_id
        self.service = service
        self._prev_job_id: Optional[str] = None
        self._prev_service: Optional[str] = None
    
    def __enter__(self) -> "LogContext":
        self._prev_job_id = current_job_id.get()
        current_job_id.set(self.job_id)
        
        if self.service is not None:
            self._prev_service = current_service.get()
            current_service.set(self.service)
        
        return self
    
    def __exit__(self, *args: Any) -> None:
        current_job_id.set(self._prev_job_id)
        
        if self._prev_service is not None:
            current_service.set(self._prev_service)
