FROM diagonalge/ai-toolkit:latest

# Install redis-server for state management
RUN apt-get update && apt-get install -y redis-server && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow \
    aiohttp \
    requests \
    fastapi \
    uvicorn \
    httpx \
    loguru \
    scipy \
    numpy \
    pandas==2.2.3 \
    tiktoken==0.8.0 \
    Pillow==11.1.0 \
    redis==7.0.1 \
    "fiber @ git+https://github.com/rayonlabs/fiber.git@2.4.0"

RUN mkdir -p /dataset/configs \
    /dataset/outputs \
    /dataset/images \
    /workspace/scripts \
    /workspace/core

COPY scripts/core /workspace/core
# COPY miner /workspace/miner
COPY trainer /workspace/trainer
COPY scripts /workspace/scripts

RUN chmod +x /workspace/scripts/run_image_trainer.sh

ENTRYPOINT ["/workspace/scripts/run_image_trainer.sh"]