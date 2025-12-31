"""
Distributed LLM Platform - Ray-orchestrated inference and training.
"""

__version__ = "1.0.0"

from ml_platform.config import ResourceConfig, get_config
from ml_platform.logger import configure_logging, get_logger

__all__ = [
    "ResourceConfig",
    "get_config",
    "configure_logging",
    "get_logger",
]
