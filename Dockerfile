FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y \
    git \
    g++ \
    ffmpeg \
    graphviz \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./pyproject.toml
COPY uv.lock ./uv.lock
# hatchling build requirements
COPY README.md ./README.md
COPY core/ ./core/

ENV UV_LINK_MODE=copy
RUN \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --python 3.10

VOLUME /app/.venv

# hare build -t $USER/turbozero .
# hare run --rm -it -v .:/app -e WANDB_API_KEY=$WANDB_API_KEY --gpus device=0 $USER/turbozero uv run src/train.py