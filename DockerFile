# syntax=docker/dockerfile:1
# VisaWise serving image: FastAPI + LangGraph RAG over an embedded LanceDB index.
# No GPU — generation is the Gemini API; the only local models are the small
# bge embedder + cross-encoder reranker (CPU). Everything the app needs at
# runtime is baked in, so the container boots with no network except Gemini.

# ---- base ---------------------------------------------------------------
# Match the project's pin (requires-python >=3.12,<3.13). "slim" = Debian
# without the extras, small but has apt for the one system lib we need.
FROM python:3.12-slim

# Pull the uv binary straight from Astral's published image (no pip install of
# uv needed). uv is our package manager / lockfile tool.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ---- environment --------------------------------------------------------
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    # write .pyc at build time so first request isn't slowed by compilation
    UV_COMPILE_BYTECODE=1 \
    # copy files into the venv instead of hardlinking (safer across layers)
    UV_LINK_MODE=copy \
    # one fixed spot for the Hugging Face model cache so we can bake it below
    HF_HOME=/opt/hf-cache \
    # don't buffer stdout/stderr -> logs show up immediately (matters on Modal
    # and for the structured logging we add in Part 2)
    PYTHONUNBUFFERED=1 \
    # silence the tokenizers fork warning
    TOKENIZERS_PARALLELISM=false \
    # put the venv's executables (incl. the `visawise-app` script) on PATH
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# ---- system deps --------------------------------------------------------
# libgomp1 = the OpenMP runtime that torch / sentence-transformers link against.
# Without it, importing torch fails at runtime. Clean apt lists to stay small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- python deps (cached layer) ----------------------------------------
# Copy ONLY the lockfiles first, then install deps WITHOUT the project itself.
# This layer is cached and only rebuilds when pyproject/uv.lock change — so
# editing app code later doesn't re-download torch every build.
COPY pyproject.toml uv.lock ./
# --mount=type=cache keeps uv's downloaded wheels in a BuildKit cache (fast
# rebuilds) instead of baking them into the image layer. With UV_LINK_MODE=copy
# the venv is self-contained, so the cache can be dropped from the final image.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# ---- app code + install the project ------------------------------------
# Now bring in the source and install the `visawise` package itself (editable).
# Editable matters: REPO_ROOT is derived from the config.py file location, so
# the package must live at /app/src/visawise with data/ and evals/ beside it.
COPY src ./src
# hatchling reads readme = "README.md" from pyproject at build time. Copy it
# here (not in the cached deps layer) so README edits don't bust the deps cache.
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---- bake the HF models (network ON here) ------------------------------
# Import the two model loaders so their weights download into HF_HOME now, at
# build time. HF_HUB_OFFLINE is still unset here, so this RUN is allowed to hit
# huggingface.co. After this, no model download ever happens at boot.
RUN uv run python -c "from visawise.embedding import get_embedder; get_embedder(); from visawise.rag.rerank import load_local_model; load_local_model()"

# ---- bake runtime data --------------------------------------------------
# The LanceDB index (dense vectors + BM25) is gitignored but required to serve;
# copy it in. latest_summary.json powers the eval dashboard's /api/eval/runs.
COPY data/lancedb ./data/lancedb
COPY evals/runs/latest_summary.json ./evals/runs/latest_summary.json

# ---- lock down runtime --------------------------------------------------
# From here on the container must NEVER reach out to Hugging Face: models are
# already baked. This keeps cold starts fast and offline-safe (Gemini is the
# only outbound call at request time).
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000

# The console script (pyproject [project.scripts]) binds uvicorn to 0.0.0.0:8000.
# Used for local `docker run`; on Modal the ASGI app is served directly and this
# CMD is ignored.
CMD ["visawise-app"]
