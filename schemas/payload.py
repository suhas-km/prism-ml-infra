"""
Pydantic models for API request and response payloads.
Defines the contract between clients and the ML platform.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request payload for chat/inference endpoint."""
    
    prompt: str = Field(..., description="Input prompt for generation")
    max_tokens: int = Field(default=256, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Nucleus sampling probability")
    adapter_path: Optional[str] = Field(default=None, description="Path to LoRA adapter for inference")
    stop_sequences: Optional[List[str]] = Field(default=None, description="Stop sequences for generation")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Explain machine learning in simple terms:",
                    "max_tokens": 256,
                    "temperature": 0.7,
                    "adapter_path": None,
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """Response payload for chat/inference endpoint."""
    
    request_id: str = Field(..., description="Unique request identifier")
    prompt: str = Field(..., description="Original input prompt")
    generated_text: str = Field(..., description="Generated text output")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    finish_reason: str = Field(..., description="Reason for generation completion")
    latency_ms: float = Field(..., description="Generation latency in milliseconds")


class Hyperparameters(BaseModel):
    """Training hyperparameters for fine-tuning."""
    
    # LoRA configuration
    lora_r: int = Field(default=16, ge=1, le=256, description="LoRA rank")
    lora_alpha: int = Field(default=32, ge=1, le=512, description="LoRA alpha scaling factor")
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5, description="LoRA dropout rate")
    target_modules: Optional[List[str]] = Field(
        default=None,
        description="Target modules for LoRA. If None, uses model defaults.",
    )
    
    # Training configuration
    learning_rate: float = Field(default=2e-4, ge=1e-7, le=1e-1, description="Learning rate")
    num_epochs: int = Field(default=3, ge=1, le=100, description="Number of training epochs")
    batch_size: int = Field(default=4, ge=1, le=128, description="Training batch size")
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=64, description="Gradient accumulation steps")
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5, description="Warmup ratio")
    weight_decay: float = Field(default=0.001, ge=0.0, le=1.0, description="Weight decay")
    max_seq_length: int = Field(default=512, ge=32, le=8192, description="Maximum sequence length")
    
    # Optimization
    fp16: bool = Field(default=True, description="Use FP16 mixed precision")
    gradient_checkpointing: bool = Field(default=True, description="Enable gradient checkpointing")


class FineTuneRequest(BaseModel):
    """Request payload for fine-tuning endpoint."""
    
    dataset_path: str = Field(..., description="Path to training dataset (JSONL or HuggingFace dataset)")
    output_path: str = Field(..., description="Path to save the trained adapter")
    base_model: Optional[str] = Field(default=None, description="Base model override. If None, uses platform default.")
    hyperparameters: Hyperparameters = Field(default_factory=Hyperparameters, description="Training hyperparameters")
    validation_split: float = Field(default=0.1, ge=0.0, le=0.5, description="Validation split ratio")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "dataset_path": "/data/training/my_dataset.jsonl",
                    "output_path": "/models/adapters/my_adapter",
                    "hyperparameters": {
                        "lora_r": 16,
                        "learning_rate": 2e-4,
                        "num_epochs": 3,
                    },
                }
            ]
        }
    }


class JobStatus(str, Enum):
    """Status of a fine-tuning job."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FineTuneResponse(BaseModel):
    """Response payload for fine-tuning endpoint."""
    
    job_id: str = Field(..., description="Unique job identifier for tracking")
    status: JobStatus = Field(..., description="Current job status")
    message: str = Field(..., description="Status message")
    output_path: Optional[str] = Field(default=None, description="Path where adapter will be saved")


class JobStatusResponse(BaseModel):
    """Response for job status query."""
    
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Progress percentage (0-1)")
    current_epoch: Optional[int] = Field(default=None, description="Current training epoch")
    total_epochs: Optional[int] = Field(default=None, description="Total training epochs")
    loss: Optional[float] = Field(default=None, description="Current training loss")
    output_path: Optional[str] = Field(default=None, description="Output path for completed job")
    error_message: Optional[str] = Field(default=None, description="Error message if job failed")


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    
    status: str = Field(..., description="Service health status")
    available_gpus: int = Field(..., description="Number of available GPUs")
    available_cpus: int = Field(..., description="Number of available CPUs")
    inference_ready: bool = Field(..., description="Whether inference engine is ready")
    active_training_jobs: int = Field(..., description="Number of active training jobs")
