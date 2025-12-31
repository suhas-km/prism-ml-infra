"""
vLLM Inference Engine wrapper with dynamic LoRA adapter loading.
Hardware-agnostic design with tensor parallelism support.
"""

from __future__ import annotations

import time
import uuid
from typing import List, Optional

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.lora.request import LoRARequest

from ml_platform.config import ResourceConfig, get_config
from ml_platform.logger import LogContext, get_logger

logger = get_logger(__name__)


class InferenceEngine:
    """
    High-throughput inference engine using vLLM's AsyncLLMEngine.
    Supports dynamic LoRA adapter loading at inference time.
    """
    
    def __init__(
        self,
        config: Optional[ResourceConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the inference engine.
        
        Args:
            config: Resource configuration. If None, uses singleton.
            model_name: Override model name. If None, uses config default.
        """
        self.config = config or get_config()
        self.model_name = model_name or self.config.model_name
        self._engine: Optional[AsyncLLMEngine] = None
        self._lora_counter: int = 0
        
        logger.info(
            "inference_engine_init",
            model=self.model_name,
            tensor_parallel_size=self.config.tensor_parallel_size,
            device=self.config.device,
        )
    
    async def start(self) -> None:
        """Start the vLLM engine asynchronously."""
        logger.info("inference_engine_starting", model=self.model_name)
        
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            tensor_parallel_size=self.config.tensor_parallel_size,
            max_model_len=self.config.max_model_len,
            enable_lora=True,
            max_loras=4,
            max_lora_rank=64,
            trust_remote_code=True,
            dtype="auto",
            gpu_memory_utilization=0.85 if self.config.is_gpu_available else None,
        )
        
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        logger.info("inference_engine_started", model=self.model_name)
    
    async def stop(self) -> None:
        """Stop the vLLM engine and release resources."""
        logger.info("inference_engine_stopping")
        if self._engine is not None:
            # vLLM handles cleanup internally
            self._engine = None
        logger.info("inference_engine_stopped")
    
    @property
    def is_ready(self) -> bool:
        """Check if the engine is ready to serve requests."""
        return self._engine is not None
    
    async def predict(
        self,
        prompt: str,
        adapter_path: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop_sequences: Optional[List[str]] = None,
    ) -> dict:
        """
        Generate text completion for the given prompt.
        
        Args:
            prompt: Input text prompt
            adapter_path: Optional path to LoRA adapter
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            stop_sequences: Optional stop sequences
        
        Returns:
            Dictionary containing generated text and metadata
        
        Raises:
            RuntimeError: If engine is not initialized
        """
        if self._engine is None:
            raise RuntimeError("Inference engine not started. Call start() first.")
        
        request_id = str(uuid.uuid4())
        
        with LogContext(job_id=request_id, service="inference"):
            logger.info(
                "predict_start",
                prompt_length=len(prompt),
                adapter_path=adapter_path,
                max_tokens=max_tokens,
            )
            
            start_time = time.perf_counter()
            
            # Configure sampling parameters
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_sequences,
            )
            
            # Configure LoRA request if adapter path provided
            lora_request: Optional[LoRARequest] = None
            if adapter_path is not None:
                self._lora_counter += 1
                lora_request = LoRARequest(
                    lora_name=f"adapter_{self._lora_counter}",
                    lora_int_id=self._lora_counter,
                    lora_path=adapter_path,
                )
                logger.info("lora_adapter_loading", adapter_path=adapter_path)
            
            # Generate response
            generated_text = ""
            tokens_generated = 0
            finish_reason = "unknown"
            
            async for output in self._engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
            ):
                if output.finished:
                    if output.outputs:
                        generated_text = output.outputs[0].text
                        tokens_generated = len(output.outputs[0].token_ids)
                        finish_reason = output.outputs[0].finish_reason or "stop"
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            logger.info(
                "predict_complete",
                tokens_generated=tokens_generated,
                latency_ms=round(elapsed_ms, 2),
                finish_reason=finish_reason,
            )
            
            return {
                "request_id": request_id,
                "prompt": prompt,
                "generated_text": generated_text,
                "tokens_generated": tokens_generated,
                "finish_reason": finish_reason,
                "latency_ms": round(elapsed_ms, 2),
            }
    
    async def health_check(self) -> dict:
        """Check engine health status."""
        return {
            "ready": self.is_ready,
            "model": self.model_name,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "device": self.config.device,
        }
