# Distributed LLM Platform

A production-grade distributed LLM platform orchestrated by **Ray**, supporting inference via **HuggingFace Transformers** and distributed fine-tuning with **LoRA/PEFT**. Compatible with ARM64/GH200 architectures.

## Architecture

```
/ml_platform
  ├── config.py           # Dynamic resource detection and configuration
  ├── logger.py           # Structured JSON logging with structlog
  ├── /schemas
  │   └── payload.py      # Pydantic models for API contracts
  ├── /core
  │   ├── inference.py    # HuggingFace Transformers inference engine
  │   └── trainer.py      # Ray Actor for SFT with LoRA
  ├── /api
  │   └── routes.py       # FastAPI router definitions
  └── main.py             # Ray Serve deployment & orchestration
```

## Features

- **Dynamic Resource Allocation**: Automatically detects available GPUs and allocates between inference/training
- **HuggingFace Transformers Inference**: Compatible with ARM64/GH200 and standard x86 architectures
- **Dynamic LoRA Loading**: Load adapters at inference time without restarting
- **Distributed Training**: Ray actors for parallel fine-tuning jobs
- **Structured Logging**: JSON logs with job context for observability

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables (Optional)

```bash
export MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Default model (non-gated)
export MAX_MODEL_LEN=4096
export SERVE_HOST="0.0.0.0"
export SERVE_PORT=8000
export INFERENCE_GPU_FRACTION=0.6
export TRAINING_GPU_FRACTION=0.4
```

### 3. Start the Platform

```bash
python -m ml_platform.main
```

The platform will automatically detect available GPUs and configure the inference engine accordingly.

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

### Chat / Inference

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain machine learning:",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

With LoRA adapter:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain machine learning:",
    "max_tokens": 256,
    "adapter_path": "/models/adapters/my_adapter"
  }'
```

### Fine-Tuning

Start a training job:

```bash
curl -X POST http://localhost:8000/fine-tune \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "/data/training/my_dataset.jsonl",
    "output_path": "/models/adapters/my_adapter",
    "hyperparameters": {
      "lora_r": 16,
      "lora_alpha": 32,
      "learning_rate": 2e-4,
      "num_epochs": 3
    }
  }'
```

Check job status:

```bash
curl http://localhost:8000/jobs/{job_id}
```

Cancel a job:

```bash
curl -X DELETE http://localhost:8000/jobs/{job_id}
```

List all jobs:

```bash
curl http://localhost:8000/jobs
```

## Configuration

### ResourceConfig

The platform automatically detects hardware via `ray.cluster_resources()`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Base model for inference (non-gated) |
| `MAX_MODEL_LEN` | `4096` | Maximum sequence length |
| `INFERENCE_GPU_FRACTION` | `0.6` | Fraction of GPUs for inference |
| `TRAINING_GPU_FRACTION` | `0.4` | Fraction of GPUs for training |
| `SERVE_HOST` | `0.0.0.0` | Server bind address |
| `SERVE_PORT` | `8000` | Server port |

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | `16` | LoRA rank |
| `lora_alpha` | `32` | LoRA alpha scaling |
| `lora_dropout` | `0.05` | LoRA dropout rate |
| `learning_rate` | `2e-4` | Learning rate |
| `num_epochs` | `3` | Training epochs |
| `batch_size` | `4` | Batch size per device |
| `gradient_accumulation_steps` | `4` | Gradient accumulation |
| `max_seq_length` | `512` | Max sequence length |
| `fp16` | `true` | Use FP16 mixed precision |
| `gradient_checkpointing` | `true` | Enable gradient checkpointing |

## Dataset Format

Training datasets should be in JSONL format with a `text` field:

```jsonl
{"text": "### Instruction: ...\n### Response: ..."}
{"text": "### Instruction: ...\n### Response: ..."}
```

Or use HuggingFace dataset names directly.

## Logging

Logs are output as structured JSON:

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "info",
  "service": "trainer",
  "job_id": "abc-123",
  "event": "train_start",
  "dataset_path": "/data/training/my_dataset.jsonl"
}
```

## Scaling

### Multi-GPU Inference

The inference engine automatically uses `device_map="auto"` for multi-GPU distribution when CUDA is available.

### Multi-Node Training

For multi-node setups, start Ray workers:

```bash
# Head node
ray start --head --port=6379

# Worker nodes
ray start --address=<head-ip>:6379
```

## License

MIT
