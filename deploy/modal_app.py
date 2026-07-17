"""Modal deployment for VisaWise.

Wraps the repo Dockerfile (CPU-torch pin, baked LanceDB index, pre-baked bge
models) and serves the existing FastAPI app as one ASGI web endpoint. Modal
builds the image on its own x86_64 builders and serves the ASGI app directly,
so the Dockerfile CMD is ignored but the app's lifespan still runs (warming
models + opening LanceDB on each cold container).

Requires a Modal secret named "visawise-secrets" containing:
    GOOGLE_API_KEY        (Gemini generation)
    CORS_ALLOWED_ORIGINS  (comma-separated; must include the Vercel origin)
Changing the secret does NOT reach already-running containers — redeploy to
pick it up.

Deploy (from the repo root, so the relative Dockerfile/context resolve):
    modal serve  deploy/modal_app.py   # ephemeral hot-reload URL (iterate)
    modal deploy deploy/modal_app.py   # persistent URL

Public URL is derived from app + function name:
    https://<workspace>--visawise-fastapi-app.modal.run
"""

import modal

app = modal.App("visawise")

image = modal.Image.from_dockerfile("Dockerfile", context_dir=".")

secret = modal.Secret.from_name("visawise-secrets")


# Decorator order matters: app.function outermost, asgi_app innermost.
@app.function(
    image=image,
    secrets=[secret],
    min_containers=0,  # scale to zero -> $0 idle; ~5-15s cold start
    scaledown_window=300,  # keep a container warm 5 min after last request
    memory=4096,  # MB: torch + 2 bge models + LanceDB + headroom
    cpu=2.0,  # cores: the cross-encoder rerank is CPU-bound
    timeout=600,  # s: generous for cold-start model warmup
)
@modal.concurrent(max_inputs=8)  # concurrent SSE streams per container
@modal.asgi_app()
def fastapi_app():
    # Import inside the function: it must run in the container (where the
    # package is installed), not at deploy time on the laptop.
    from visawise.app.main import app as web_app

    return web_app
