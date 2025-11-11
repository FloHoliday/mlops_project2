# Stage 1: Builder
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Create venv and install dependencies
RUN uv sync --frozen --no-install-project --no-dev

# Stage 2: Runtime
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

WORKDIR /app

# Copy only the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY train.py data.py model.py ./

# Set Weights & Biases API key
ENV WANDB_API_KEY=""

# Default command
CMD ["python", "train.py", \
     "--learning_rate", "5e-5", \
     "--weight_decay", "0.0", \
     "--warmup_steps", "0", \
     "--max_seq_length", "256", \
     "--train_batch_size", "32", \
     "--epochs", "3", \
     "--seed", "42", \
     "--optimizer_type", "AdamW"]