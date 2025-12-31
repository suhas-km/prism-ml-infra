"""
FastAPI router defining API endpoints for inference and training.
"""

from __future__ import annotations

from typing import Dict

from fastapi import APIRouter, HTTPException, status

from ml_platform.schemas.payload import (
    ChatRequest,
    ChatResponse,
    FineTuneRequest,
    FineTuneResponse,
    HealthResponse,
    JobStatus,
    JobStatusResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    Returns system status including GPU availability and active jobs.
    
    Note: This is a placeholder. The actual implementation is in main.py
    where the inference engine and job registry are available.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Health check must be accessed via the main deployment",
    )


@router.post("/chat", response_model=ChatResponse, tags=["Inference"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Generate text completion for the given prompt.
    Supports dynamic LoRA adapter loading via adapter_path.
    
    Note: This is a placeholder. The actual implementation is in main.py
    where the inference engine handle is available.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Chat endpoint must be accessed via the main deployment",
    )


@router.post("/fine-tune", response_model=FineTuneResponse, tags=["Training"])
async def fine_tune(request: FineTuneRequest) -> FineTuneResponse:
    """
    Start a fine-tuning job with the specified parameters.
    Returns immediately with a job_id for tracking.
    
    Note: This is a placeholder. The actual implementation is in main.py
    where the training actor can be spawned.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Fine-tune endpoint must be accessed via the main deployment",
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["Training"])
async def get_job_status(job_id: str) -> JobStatusResponse:
    """
    Get the status of a fine-tuning job.
    
    Note: This is a placeholder. The actual implementation is in main.py
    where the job registry is available.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Job status endpoint must be accessed via the main deployment",
    )


@router.delete("/jobs/{job_id}", response_model=JobStatusResponse, tags=["Training"])
async def cancel_job(job_id: str) -> JobStatusResponse:
    """
    Cancel a running fine-tuning job.
    
    Note: This is a placeholder. The actual implementation is in main.py
    where the job registry is available.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Cancel job endpoint must be accessed via the main deployment",
    )
