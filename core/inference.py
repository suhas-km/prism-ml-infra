"""
HuggingFace Transformers Inference Engine with LoRA adapter support.
Hardware-agnostic design compatible with ARM64/GH200.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from ml_platform.config import ResourceConfig, get_config
from ml_platform.logger import LogContext, get_logger

logger = get_logger(__name__)


class InferenceEngine:
    """
    Inference engine using HuggingFace Transformers.
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
        self._model = None
        self._tokenizer = None
        self._current_adapter: Optional[str] = None
        
        logger.info(
            "inference_engine_init",
            model=self.model_name,
            device=self.config.device,
        )
    
    async def start(self) -> None:
        """Load the model and tokenizer asynchronously."""
        logger.info("inference_engine_starting", model=self.model_name)
        
        # Run model loading in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)
        
        logger.info("inference_engine_started", model=self.model_name)
    
    def _load_model(self) -> None:
        """Load model and tokenizer (blocking)."""
        device = self.config.device
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Configure model loading
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }
        
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
        
        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        
        if device == "cpu":
            self._model = self._model.to(device)
        
        self._model.eval()
    
    async def stop(self) -> None:
        """Unload model and release resources."""
        logger.info("inference_engine_stopping")
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("inference_engine_stopped")
    
    @property
    def is_ready(self) -> bool:
        """Check if the engine is ready to serve requests."""
        return self._model is not None and self._tokenizer is not None
    
    def _load_adapter(self, adapter_path: str) -> None:
        """Load a LoRA adapter onto the base model."""
        if self._current_adapter == adapter_path:
            return
        
        # Unload current adapter if any
        if self._current_adapter is not None and hasattr(self._model, 'unload'):
            self._model = self._model.unload()
        
        # Load new adapter
        self._model = PeftModel.from_pretrained(
            self._model,
            adapter_path,
            is_trainable=False,
        )
        self._current_adapter = adapter_path
        logger.info("lora_adapter_loaded", adapter_path=adapter_path)
    
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
        if not self.is_ready:
            raise RuntimeError("Inference engine not started. Call start() first.")
        
        request_id = str(uuid.uuid4())
        
        with LogContext(job_id=request_id, service="inference"):
            logger.info(
                "predict_start",
                prompt_length=len(prompt),
                adapter_path=adapter_path,
                max_tokens=max_tokens,
            )
            
            # Load adapter if specified
            if adapter_path is not None:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_adapter, adapter_path)
            
            start_time = time.perf_counter()
            
            # Run generation in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._generate_sync,
                prompt,
                max_tokens,
                temperature,
                top_p,
                stop_sequences,
            )
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            logger.info(
                "predict_complete",
                tokens_generated=result["tokens_generated"],
                latency_ms=round(elapsed_ms, 2),
                finish_reason=result["finish_reason"],
            )
            
            return {
                "request_id": request_id,
                "prompt": prompt,
                "generated_text": result["generated_text"],
                "tokens_generated": result["tokens_generated"],
                "finish_reason": result["finish_reason"],
                "latency_ms": round(elapsed_ms, 2),
            }
    
    def _generate_sync(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: Optional[List[str]],
    ) -> dict:
        """Synchronous generation (runs in thread pool)."""
        # Tokenize input
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_model_len - max_tokens,
        )
        
        # Move to device
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        input_length = inputs["input_ids"].shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        
        # Decode output (only new tokens)
        generated_ids = outputs[0][input_length:]
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        tokens_generated = len(generated_ids)
        
        # Determine finish reason
        finish_reason = "length"
        if tokens_generated < max_tokens:
            if generated_ids[-1].item() == self._tokenizer.eos_token_id:
                finish_reason = "stop"
        
        # Check for stop sequences
        if stop_sequences:
            for seq in stop_sequences:
                if seq in generated_text:
                    generated_text = generated_text.split(seq)[0]
                    finish_reason = "stop"
                    break
        
        return {
            "generated_text": generated_text,
            "tokens_generated": tokens_generated,
            "finish_reason": finish_reason,
        }
    
    async def health_check(self) -> dict:
        """Check engine health status."""
        return {
            "ready": self.is_ready,
            "model": self.model_name,
            "device": self.config.device,
        }
