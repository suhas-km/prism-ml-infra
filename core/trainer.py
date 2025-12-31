"""
Distributed training actor for Supervised Fine-Tuning (SFT) with LoRA.
Uses Ray actors for distributed execution and TRL for training.
"""

from __future__ import annotations

import gc
import os
from typing import Any, Dict, Optional

import ray
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

from ml_platform.config import ResourceConfig, get_config
from ml_platform.logger import LogContext, get_logger, set_service_context
from ml_platform.schemas.payload import Hyperparameters, JobStatus

logger = get_logger(__name__)


@ray.remote
class TrainingActor:
    """
    Ray actor for distributed SFT training with LoRA adapters.
    Each actor manages its own GPU memory and training state.
    """
    
    def __init__(
        self,
        job_id: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the training actor.
        
        Args:
            job_id: Unique identifier for this training job
            config: Optional configuration override as dict
        """
        set_service_context("trainer")
        self.job_id = job_id
        self._config = ResourceConfig(**config) if config else get_config()
        
        self._status: JobStatus = JobStatus.PENDING
        self._progress: float = 0.0
        self._current_epoch: int = 0
        self._total_epochs: int = 0
        self._current_loss: Optional[float] = None
        self._error_message: Optional[str] = None
        self._output_path: Optional[str] = None
        
        logger.info(
            "training_actor_init",
            job_id=job_id,
            device=self._config.device,
            training_gpus=self._config.training_gpus,
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Return current training status."""
        return {
            "job_id": self.job_id,
            "status": self._status.value,
            "progress": self._progress,
            "current_epoch": self._current_epoch,
            "total_epochs": self._total_epochs,
            "loss": self._current_loss,
            "output_path": self._output_path,
            "error_message": self._error_message,
        }
    
    def train(
        self,
        dataset_path: str,
        output_path: str,
        hyperparams: Dict[str, Any],
        base_model: Optional[str] = None,
        validation_split: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Execute supervised fine-tuning with LoRA.
        
        Args:
            dataset_path: Path to training dataset (JSONL or HF dataset)
            output_path: Path to save trained adapter
            hyperparams: Training hyperparameters dict
            base_model: Base model name. If None, uses config default.
            validation_split: Validation split ratio
        
        Returns:
            Dictionary with training results
        """
        with LogContext(job_id=self.job_id, service="trainer"):
            logger.info(
                "train_start",
                dataset_path=dataset_path,
                output_path=output_path,
                base_model=base_model,
            )
            
            self._status = JobStatus.RUNNING
            self._output_path = output_path
            
            try:
                # Parse hyperparameters
                hp = Hyperparameters(**hyperparams)
                self._total_epochs = hp.num_epochs
                
                # Determine model
                model_name = base_model or self._config.model_name
                
                # Load tokenizer
                logger.info("loading_tokenizer", model=model_name)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Configure quantization for memory efficiency
                quantization_config = None
                if self._config.is_gpu_available:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                
                # Load base model
                logger.info("loading_base_model", model=model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto" if self._config.is_gpu_available else None,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if hp.fp16 else torch.float32,
                )
                
                if hp.gradient_checkpointing:
                    model.gradient_checkpointing_enable()
                
                # Configure LoRA
                logger.info(
                    "configuring_lora",
                    r=hp.lora_r,
                    alpha=hp.lora_alpha,
                    dropout=hp.lora_dropout,
                )
                
                target_modules = hp.target_modules
                if target_modules is None:
                    # Default target modules for common architectures
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=hp.lora_r,
                    lora_alpha=hp.lora_alpha,
                    lora_dropout=hp.lora_dropout,
                    target_modules=target_modules,
                    bias="none",
                )
                
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
                
                # Load dataset
                logger.info("loading_dataset", path=dataset_path)
                if dataset_path.endswith(".jsonl") or dataset_path.endswith(".json"):
                    dataset = load_dataset("json", data_files=dataset_path, split="train")
                else:
                    dataset = load_dataset(dataset_path, split="train")
                
                # Split dataset
                if validation_split > 0:
                    split_dataset = dataset.train_test_split(test_size=validation_split)
                    train_dataset = split_dataset["train"]
                    eval_dataset = split_dataset["test"]
                else:
                    train_dataset = dataset
                    eval_dataset = None
                
                logger.info(
                    "dataset_loaded",
                    train_size=len(train_dataset),
                    eval_size=len(eval_dataset) if eval_dataset else 0,
                )
                
                # Configure training
                os.makedirs(output_path, exist_ok=True)
                
                sft_config = SFTConfig(
                    output_dir=output_path,
                    num_train_epochs=hp.num_epochs,
                    per_device_train_batch_size=hp.batch_size,
                    gradient_accumulation_steps=hp.gradient_accumulation_steps,
                    learning_rate=hp.learning_rate,
                    warmup_ratio=hp.warmup_ratio,
                    weight_decay=hp.weight_decay,
                    max_seq_length=hp.max_seq_length,
                    fp16=hp.fp16 and self._config.is_gpu_available,
                    logging_steps=10,
                    save_strategy="epoch",
                    evaluation_strategy="epoch" if eval_dataset else "no",
                    save_total_limit=2,
                    load_best_model_at_end=True if eval_dataset else False,
                    report_to="none",
                    remove_unused_columns=False,
                )
                
                # Initialize trainer
                logger.info("initializing_trainer")
                trainer = SFTTrainer(
                    model=model,
                    args=sft_config,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    processing_class=tokenizer,
                    peft_config=peft_config,
                )
                
                # Train
                logger.info("training_starting", epochs=hp.num_epochs)
                train_result = trainer.train()
                
                # Save adapter
                logger.info("saving_adapter", path=output_path)
                trainer.save_model(output_path)
                tokenizer.save_pretrained(output_path)
                
                self._status = JobStatus.COMPLETED
                self._progress = 1.0
                self._current_epoch = hp.num_epochs
                
                final_loss = train_result.training_loss
                self._current_loss = final_loss
                
                logger.info(
                    "train_complete",
                    final_loss=final_loss,
                    output_path=output_path,
                )
                
                return {
                    "job_id": self.job_id,
                    "status": self._status.value,
                    "final_loss": final_loss,
                    "output_path": output_path,
                }
                
            except Exception as e:
                self._status = JobStatus.FAILED
                self._error_message = str(e)
                logger.error("train_failed", error=str(e), exc_info=True)
                raise
                
            finally:
                # Cleanup to release VRAM
                self._cleanup_resources()
    
    def _cleanup_resources(self) -> None:
        """Release GPU memory after training."""
        logger.info("cleanup_resources_start")
        
        # Clear Python references
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info(
                "cuda_cache_cleared",
                memory_allocated=torch.cuda.memory_allocated(),
                memory_reserved=torch.cuda.memory_reserved(),
            )
        
        logger.info("cleanup_resources_complete")
    
    def cancel(self) -> Dict[str, Any]:
        """Cancel the training job."""
        logger.info("training_cancelled", job_id=self.job_id)
        self._status = JobStatus.CANCELLED
        self._cleanup_resources()
        return self.get_status()
