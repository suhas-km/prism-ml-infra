"""
Centralized configuration with dynamic hardware detection.
Uses Ray cluster resources to determine available GPUs and allocate them
between inference and training workloads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import ray


@dataclass
class ResourceConfig:
    """Dynamic resource configuration based on Ray cluster state."""
    
    # Model configuration
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-hf"))
    max_model_len: int = field(default_factory=lambda: int(os.getenv("MAX_MODEL_LEN", "4096")))
    
    # Resource allocation fractions
    inference_gpu_fraction: float = field(default_factory=lambda: float(os.getenv("INFERENCE_GPU_FRACTION", "0.6")))
    training_gpu_fraction: float = field(default_factory=lambda: float(os.getenv("TRAINING_GPU_FRACTION", "0.4")))
    
    # Derived values (computed post-init)
    available_gpus: int = field(init=False, default=0)
    available_cpus: int = field(init=False, default=0)
    inference_gpus: int = field(init=False, default=0)
    training_gpus: int = field(init=False, default=0)
    tensor_parallel_size: int = field(init=False, default=1)
    
    # Service configuration
    serve_host: str = field(default_factory=lambda: os.getenv("SERVE_HOST", "0.0.0.0"))
    serve_port: int = field(default_factory=lambda: int(os.getenv("SERVE_PORT", "8000")))
    
    # Training defaults
    default_lora_r: int = field(default_factory=lambda: int(os.getenv("LORA_R", "16")))
    default_lora_alpha: int = field(default_factory=lambda: int(os.getenv("LORA_ALPHA", "32")))
    default_lora_dropout: float = field(default_factory=lambda: float(os.getenv("LORA_DROPOUT", "0.05")))
    
    def __post_init__(self) -> None:
        """Detect hardware resources from Ray cluster."""
        self._detect_resources()
    
    def _detect_resources(self) -> None:
        """Query Ray cluster for available resources and compute allocations."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        resources = ray.cluster_resources()
        
        self.available_gpus = int(resources.get("GPU", 0))
        self.available_cpus = int(resources.get("CPU", os.cpu_count() or 4))
        
        # Allocate GPUs between inference and training
        if self.available_gpus > 0:
            self.inference_gpus = max(1, int(self.available_gpus * self.inference_gpu_fraction))
            self.training_gpus = max(1, self.available_gpus - self.inference_gpus)
            self.tensor_parallel_size = self.inference_gpus
        else:
            # CPU-only mode
            self.inference_gpus = 0
            self.training_gpus = 0
            self.tensor_parallel_size = 1
    
    def refresh(self) -> None:
        """Re-detect resources (useful after cluster scaling)."""
        self._detect_resources()
    
    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU resources are available."""
        return self.available_gpus > 0
    
    @property
    def device(self) -> str:
        """Return device string for PyTorch."""
        return "cuda" if self.is_gpu_available else "cpu"
    
    def get_training_compute_config(self) -> dict:
        """Return compute configuration for training actors."""
        if self.is_gpu_available:
            return {"num_gpus": self.training_gpus}
        return {"num_cpus": max(1, self.available_cpus // 2)}
    
    def get_inference_compute_config(self) -> dict:
        """Return compute configuration for inference deployment."""
        if self.is_gpu_available:
            return {"num_gpus": self.inference_gpus}
        return {"num_cpus": max(1, self.available_cpus // 2)}


# Singleton instance
_config: Optional[ResourceConfig] = None


def get_config() -> ResourceConfig:
    """Get or create the singleton ResourceConfig instance."""
    global _config
    if _config is None:
        _config = ResourceConfig()
    return _config


def reset_config() -> None:
    """Reset the singleton config (useful for testing)."""
    global _config
    _config = None
