"""
Ray Serve entrypoint and resource orchestrator.
Manages inference engine deployment and training job scheduling.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import ray
from fastapi import FastAPI, HTTPException, status
from ray import serve

from ml_platform.config import ResourceConfig, get_config
from ml_platform.core.inference import InferenceEngine
from ml_platform.core.trainer import TrainingActor
from ml_platform.logger import configure_logging, get_logger
from ml_platform.schemas.payload import (
    ChatRequest,
    ChatResponse,
    FineTuneRequest,
    FineTuneResponse,
    HealthResponse,
    JobStatus,
    JobStatusResponse,
)

logger = get_logger(__name__)


class JobRegistry:
    """Registry for tracking active and completed training jobs."""
    
    def __init__(self) -> None:
        self._jobs: Dict[str, ray.actor.ActorHandle] = {}
    
    def register(self, job_id: str, actor: ray.actor.ActorHandle) -> None:
        """Register a training job actor."""
        self._jobs[job_id] = actor
    
    def get(self, job_id: str) -> Optional[ray.actor.ActorHandle]:
        """Get actor handle by job ID."""
        return self._jobs.get(job_id)
    
    def remove(self, job_id: str) -> None:
        """Remove job from registry."""
        self._jobs.pop(job_id, None)
    
    def list_jobs(self) -> Dict[str, ray.actor.ActorHandle]:
        """List all registered jobs."""
        return dict(self._jobs)
    
    @property
    def active_count(self) -> int:
        """Count of active jobs."""
        return len(self._jobs)


@serve.deployment(
    name="ml_platform",
    ray_actor_options={"num_cpus": 1},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_ongoing_requests": 10,
    },
)
@serve.ingress(FastAPI(
    title="Distributed LLM Platform",
    description="High-throughput inference and distributed fine-tuning platform",
    version="1.0.0",
))
class MLPlatformDeployment:
    """
    Ray Serve deployment combining inference and training orchestration.
    """
    
    def __init__(self) -> None:
        """Initialize the ML platform deployment."""
        configure_logging(service_name="platform", json_output=True)
        
        self.config: ResourceConfig = get_config()
        self.inference_engine: InferenceEngine = InferenceEngine(config=self.config)
        self.job_registry: JobRegistry = JobRegistry()
        self._initialized: bool = False
        
        logger.info(
            "ml_platform_deployment_init",
            available_gpus=self.config.available_gpus,
            inference_gpus=self.config.inference_gpus,
            training_gpus=self.config.training_gpus,
        )
    
    async def _ensure_initialized(self) -> None:
        """Lazily initialize the inference engine."""
        if not self._initialized:
            await self.inference_engine.start()
            self._initialized = True
    
    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.1)
    async def _batch_predict(
        self,
        prompts: list[str],
        adapter_paths: list[Optional[str]],
        max_tokens_list: list[int],
        temperatures: list[float],
        top_ps: list[float],
        stop_sequences_list: list[Optional[list[str]]],
    ) -> list[dict]:
        """Batch inference for improved throughput."""
        results = []
        for prompt, adapter_path, max_tokens, temp, top_p, stop_seqs in zip(
            prompts, adapter_paths, max_tokens_list, temperatures, top_ps, stop_sequences_list
        ):
            result = await self.inference_engine.predict(
                prompt=prompt,
                adapter_path=adapter_path,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                stop_sequences=stop_seqs,
            )
            results.append(result)
        return results
    
    # ========== API Endpoints ==========
    
    @serve.app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check(self) -> HealthResponse:
        """Health check endpoint with system status."""
        self.config.refresh()
        
        return HealthResponse(
            status="healthy",
            available_gpus=self.config.available_gpus,
            available_cpus=self.config.available_cpus,
            inference_ready=self.inference_engine.is_ready,
            active_training_jobs=self.job_registry.active_count,
        )
    
    @serve.app.post("/chat", response_model=ChatResponse, tags=["Inference"])
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Generate text completion for the given prompt."""
        await self._ensure_initialized()
        
        logger.info(
            "chat_request_received",
            prompt_length=len(request.prompt),
            adapter_path=request.adapter_path,
        )
        
        try:
            result = await self.inference_engine.predict(
                prompt=request.prompt,
                adapter_path=request.adapter_path,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop_sequences=request.stop_sequences,
            )
            
            return ChatResponse(
                request_id=result["request_id"],
                prompt=result["prompt"],
                generated_text=result["generated_text"],
                tokens_generated=result["tokens_generated"],
                finish_reason=result["finish_reason"],
                latency_ms=result["latency_ms"],
            )
            
        except Exception as e:
            logger.error("chat_request_failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference failed: {str(e)}",
            )
    
    @serve.app.post("/fine-tune", response_model=FineTuneResponse, tags=["Training"])
    async def fine_tune(self, request: FineTuneRequest) -> FineTuneResponse:
        """
        Start a fine-tuning job.
        Spawns a detached Ray actor for training and returns immediately.
        """
        job_id = str(uuid.uuid4())
        
        logger.info(
            "fine_tune_request_received",
            job_id=job_id,
            dataset_path=request.dataset_path,
            output_path=request.output_path,
        )
        
        try:
            # Get compute config for training
            compute_config = self.config.get_training_compute_config()
            
            # Spawn detached training actor
            actor_options = {
                "name": f"trainer_{job_id}",
                "lifetime": "detached",
                **compute_config,
            }
            
            training_actor = TrainingActor.options(**actor_options).remote(
                job_id=job_id,
            )
            
            # Register the job
            self.job_registry.register(job_id, training_actor)
            
            # Start training in background (non-blocking)
            training_actor.train.remote(
                dataset_path=request.dataset_path,
                output_path=request.output_path,
                hyperparams=request.hyperparameters.model_dump(),
                base_model=request.base_model,
                validation_split=request.validation_split,
            )
            
            logger.info("fine_tune_job_started", job_id=job_id)
            
            return FineTuneResponse(
                job_id=job_id,
                status=JobStatus.PENDING,
                message="Training job started successfully",
                output_path=request.output_path,
            )
            
        except Exception as e:
            logger.error("fine_tune_request_failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to start training job: {str(e)}",
            )
    
    @serve.app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["Training"])
    async def get_job_status(self, job_id: str) -> JobStatusResponse:
        """Get the status of a fine-tuning job."""
        actor = self.job_registry.get(job_id)
        
        if actor is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found",
            )
        
        try:
            status_dict = await actor.get_status.remote()
            
            return JobStatusResponse(
                job_id=status_dict["job_id"],
                status=JobStatus(status_dict["status"]),
                progress=status_dict.get("progress"),
                current_epoch=status_dict.get("current_epoch"),
                total_epochs=status_dict.get("total_epochs"),
                loss=status_dict.get("loss"),
                output_path=status_dict.get("output_path"),
                error_message=status_dict.get("error_message"),
            )
            
        except ray.exceptions.RayActorError:
            # Actor has terminated
            self.job_registry.remove(job_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} actor has terminated",
            )
    
    @serve.app.delete("/jobs/{job_id}", response_model=JobStatusResponse, tags=["Training"])
    async def cancel_job(self, job_id: str) -> JobStatusResponse:
        """Cancel a running fine-tuning job."""
        actor = self.job_registry.get(job_id)
        
        if actor is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found",
            )
        
        try:
            status_dict = await actor.cancel.remote()
            
            # Cleanup actor
            ray.kill(actor)
            self.job_registry.remove(job_id)
            
            logger.info("job_cancelled", job_id=job_id)
            
            return JobStatusResponse(
                job_id=status_dict["job_id"],
                status=JobStatus.CANCELLED,
                progress=status_dict.get("progress"),
                current_epoch=status_dict.get("current_epoch"),
                total_epochs=status_dict.get("total_epochs"),
            )
            
        except ray.exceptions.RayActorError:
            self.job_registry.remove(job_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} actor has terminated",
            )
    
    @serve.app.get("/jobs", tags=["Training"])
    async def list_jobs(self) -> Dict[str, Any]:
        """List all registered training jobs."""
        jobs = []
        for job_id, actor in self.job_registry.list_jobs().items():
            try:
                status_dict = await actor.get_status.remote()
                jobs.append(status_dict)
            except ray.exceptions.RayActorError:
                self.job_registry.remove(job_id)
        
        return {"jobs": jobs, "total": len(jobs)}


def create_app() -> serve.Application:
    """Create the Ray Serve application."""
    return MLPlatformDeployment.bind()


def main() -> None:
    """
    Main entrypoint for the ML platform.
    Initializes Ray and deploys the Serve application.
    """
    # Configure logging
    configure_logging(service_name="platform", json_output=True)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    config = get_config()
    
    logger.info(
        "starting_ml_platform",
        available_gpus=config.available_gpus,
        available_cpus=config.available_cpus,
        inference_gpus=config.inference_gpus,
        training_gpus=config.training_gpus,
        host=config.serve_host,
        port=config.serve_port,
    )
    
    # Deploy the application
    serve.run(
        create_app(),
        host=config.serve_host,
        port=config.serve_port,
    )
    
    logger.info(
        "ml_platform_running",
        url=f"http://{config.serve_host}:{config.serve_port}",
    )


if __name__ == "__main__":
    main()
