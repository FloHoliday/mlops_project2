# MLOps Project 2: Containerization

This project demonstrates containerized machine learning workflows by fine-tuning a DistilBERT model on the MRPC (paraphrase detection) dataset. The implementation uses Docker for reproducible training across different environments.

## Project Overview

- **Model**: DistilBERT-base-uncased
- **Task**: Paraphrase detection (MRPC from GLUE benchmark)
- **Framework**: PyTorch Lightning
- **Package Manager**: uv (fast Python package manager)
- **Experiment Tracking**: Weights & Biases (W&B)
- **Container**: Docker with multi-stage build

## Project Structure

```
.
├── Dockerfile           # Multi-stage Docker build configuration
├── pyproject.toml       # Project dependencies and uv configuration
├── uv.lock             # Locked dependency versions
├── train.py            # Main training script
├── data.py             # Dataset handling (GLUEDataModule)
├── model.py            # Model definition (GLUETransformer)
├── .env                # Environment variables (create this, not in repo)
├── .gitignore          # Git ignore file (includes .env)
└── README.md           # This file
```

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed on your system
- [Weights & Biases](https://wandb.ai/) account (free tier works)
- W&B API key (get it from https://wandb.ai/authorize)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/FloHoliday/MLOPS_Project2
cd mlops-project2
```

### 2. Build the Docker Image

```bash
docker build -t mlops-project2 .
```

Build time: ~3-4 minutes on first build (uses layer caching for faster subsequent builds)

### 3. Run Training with Default Hyperparameters

**Option 1: Pass API key directly**
```bash
docker run --rm -e WANDB_API_KEY=your_api_key_here mlops-project2
```

**Option 2: Use .env file (recommended for local development)**

Create a `.env` file in the project root:
```bash
echo "WANDB_API_KEY=your_api_key_here" > .env
```

Then run with the env file:
```bash
docker run --rm --env-file .env mlops-project2
```

**Important**: Add `.env` to your `.gitignore` to avoid committing secrets:
```bash
echo ".env" >> .gitignore
```

**Default hyperparameters** (optimized from Project 1):
- Learning rate: 5e-5
- Batch size: 32
- Max sequence length: 256
- Optimizer: AdamW
- Weight decay: 0.0
- Warmup steps: 0
- Epochs: 3
- Seed: 42

### 4. Run Training with Custom Hyperparameters

**With .env file:**
```bash
docker run --rm \
  --env-file .env \
  mlops-project2 \
  python train.py \
  --learning_rate 3e-5 \
  --train_batch_size 16 \
  --max_seq_length 128 \
  --epochs 5
```

**Without .env file:**
```bash
docker run --rm \
  -e WANDB_API_KEY=your_api_key_here \
  mlops-project2 \
  python train.py \
  --learning_rate 3e-5 \
  --train_batch_size 16 \
  --max_seq_length 128 \
  --epochs 5
```

## Available Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--learning_rate` | float | 5e-5 | Learning rate for optimizer |
| `--weight_decay` | float | 0.0 | L2 regularization weight decay |
| `--warmup_steps` | int | 0 | Number of warmup steps |
| `--max_seq_length` | int | 256 | Maximum sequence length for tokenization |
| `--train_batch_size` | int | 32 | Training batch size |
| `--eval_batch_size` | int | 32 | Evaluation batch size |
| `--epochs` | int | 3 | Number of training epochs |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--optimizer_type` | str | AdamW | Optimizer (AdamW, Adam, or SGD) |
| `--model_name` | str | distilbert-base-uncased | HuggingFace model name |
| `--dataset` | str | mrpc | GLUE dataset task name |

### Weights & Biases Run Names

By default, W&B runs are automatically named with the format:
```
docker-lr{learning_rate}-bs{batch_size}-wd{weight_decay}-warmup{warmup_steps}-maxlen{max_seq_length}-opt{optimizer}-seed{seed}
```

**Example**: `docker-lr5e-05-bs32-wd0.0-warmup0-maxlen256-optAdamW-seed42`

To customize the run name prefix, edit `train.py` and modify the `run_name` variable (around line 26):
```python
run_name = (
    f"my-experiment-"  # Change "docker-" to your preferred prefix
    f"lr{args.learning_rate}-"
    f"bs{args.train_batch_size}-"
    # ... rest of the name
)
```

## Expected Results

### Performance Metrics
- **Validation Accuracy**: ~0.85-0.86
- **Validation F1 Score**: ~0.89-0.90
- **Validation Loss**: ~0.43-0.44

*Note: Results may vary slightly across different hardware architectures (ARM vs x86_64) due to floating-point arithmetic differences.*

## Docker Image Details

### Multi-Stage Build
The Dockerfile uses a two-stage build:
1. **Builder stage**: Installs dependencies using `uv`
2. **Runtime stage**: Contains only the virtual environment and application code

### Image Size Optimization
- **Base image**: `python:3.12-slim`
- **PyTorch**: CPU-only version (reduces image size by ~2GB)
- **Final image size**: Significantly smaller than CUDA-enabled alternatives

### Environment Variables
- `PYTHONUNBUFFERED=1`: Real-time logging output
- `UV_COMPILE_BYTECODE=1`: Precompile Python files for faster startup
- `WANDB_API_KEY`: Your W&B API key (set at runtime)

## Running Without Docker (Local Development)

### 1. Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Activate Virtual Environment

```bash
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```

### 4. Run Training

**Option 1: Using .env file (recommended)**
```bash
# Create .env file
echo "WANDB_API_KEY=your_api_key_here" > .env

# Run training
python train.py --learning_rate 5e-5 --train_batch_size 32
```

**Option 2: Using environment variable**
```bash
export WANDB_API_KEY=your_api_key_here  # macOS/Linux
# or
set WANDB_API_KEY=your_api_key_here     # Windows

python train.py --learning_rate 5e-5 --train_batch_size 32
```

The training script automatically loads the API key from the `.env` file if it exists.

## Deployment on GitHub Codespaces

1. Fork this repository to your GitHub account
2. Open in Codespaces (click "Code" → "Codespaces" → "Create codespace")
3. Wait for environment setup (~2 minutes)
4. Build and run:

```bash
docker build -t mlops-project2 .
docker run --rm -e WANDB_API_KEY=$WANDB_API_KEY mlops-project2
```

**Recommended configuration**: 4-core, 16GB RAM for optimal performance

## Troubleshooting

### Issue: "Out of memory" during training
**Solution**: Reduce batch size using `--train_batch_size 16` or `--max_seq_length 128`


### Issue: Docker build fails on "No space left on device"
**Solution**: Clean up Docker images and containers:
```bash
docker system prune -a
```

### Issue: Training is very slow
**Solution**: This is expected on CPU. Training takes 10-15x longer than GPU. Consider:
- Using a cloud instance with GPU support
- Reducing epochs: `--epochs 1`
- Using a smaller sequence length: `--max_seq_length 128`

## Project Background

This project is part of the MLOps course at HSLU (HS25). It demonstrates:
- Converting Jupyter notebooks to production-ready Python scripts
- Containerizing ML workflows with Docker
- Reproducible training across different environments
- Hyperparameter configuration via command-line arguments
- Experiment tracking with W&B

## Key Findings

1. **Perfect reproducibility** requires identical hardware, not just identical containers
2. **Multi-stage builds** significantly reduce Docker image size
3. **CPU-only PyTorch** is suitable for development/testing but 10x slower than GPU
4. **Environment variables** provide secure secret management in containers

## License

This project is for educational purposes as part of the MLOps course.

## Acknowledgments

- PyTorch Lightning team for the excellent framework
- Weights & Biases for experiment tracking
- Astral for the uv package manager
- HuggingFace for pre-trained models and datasets

## Contact

For questions or issues, please open a GitHub issue.