# CPU-only repro for export/parity checks (no 1.6GB models copied in).
FROM python:3.13-slim
ENV USE_TF=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
WORKDIR /app
COPY requirements-benchmark.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements-benchmark.txt && pip install pytest
COPY tools/export/ tools/export/
COPY packages/test-vectors/ packages/test-vectors/
COPY benchmark.py index.py ./
CMD ["python", "-m", "pytest", "tests/", "-q"]
