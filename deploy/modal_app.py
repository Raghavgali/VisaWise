"""Modal deployment for VisaWise (Phase 5.4).

============================ CONTRACT =============================
This module, when complete, must provide:

1. A `modal.App` named "visawise".

2. An `image` built FROM THE REPO Dockerfile via `modal.Image.from_dockerfile`
   — so the CPU-torch pin, the baked LanceDB index, and the pre-baked bge
   models all carry over unchanged. Modal builds this on its own x86_64
   builders, which means (unlike the local build) there is NO QEMU emulation,
   NO local disk limit, and NO space-in-path BuildKit bug — so `from_dockerfile`
   reads the Dockerfile directly; the `-f - < Dockerfile` stdin workaround is
   NOT needed here. `.dockerignore` is honored automatically.

3. ONE ASGI web endpoint that returns the existing FastAPI app
   (`visawise.app.main:app`) — same routes, same SSE `/api/chat`. Modal serves
   the ASGI app directly, so the Dockerfile CMD is ignored and the app's
   `lifespan` still runs (warming the models + opening LanceDB on each cold
   container).

4. Secrets injected from a Modal secret named "visawise-secrets":
      GOOGLE_API_KEY        (required — Gemini generation)
      CORS_ALLOWED_ORIGINS  (comma-separated; include the Vercel origin)

5. Scale-to-zero (min_containers=0) to stay free, with enough memory/CPU for
   torch + the two bge models + LanceDB.

The endpoint's public URL is derived from the app + function name:
      https://<workspace>--visawise-<function-name>.modal.run
so name the function `fastapi_app` to get `visawise-fastapi-app.modal.run`.

------------------------- Prerequisites --------------------------
  * `modal token new`                                   (auth — DONE)
  * `modal secret create visawise-secrets \
        GOOGLE_API_KEY=<key> \
        CORS_ALLOWED_ORIGINS=https://<app>.vercel.app,http://localhost:5173`

--------------------------- Commands -----------------------------
  Run these FROM THE REPO ROOT (so the relative Dockerfile/context resolve):
    modal serve  deploy/modal_app.py   # ephemeral hot-reload URL (iterate)
    modal deploy deploy/modal_app.py   # persistent URL

---------------------------- Verify ------------------------------
  * GET  /health         -> 503 until warm, then {"status":"healthy"}
  * POST /api/chat (SSE)  -> token/done frames + citations
  * GET  /api/eval/runs   -> committed latest_summary.json
  * Vercel page (config.js -> this URL) streams an answer end-to-end.
==================================================================
"""

import modal

# TODO(you) 1 — the app:
#     app = modal.App("visawise")
app = modal.App("visawise")

# TODO(you) 2 — the image from the repo Dockerfile:
#     image = modal.Image.from_dockerfile("Dockerfile", context_dir=".")
#   (paths are relative to where you run `modal deploy`; run from repo root.)
image = modal.Image.from_dockerfile("Dockerfile", context_dir=".")

# TODO(you) 3 — reference the secret created above:
#     secret = modal.Secret.from_name("visawise-secrets")
secret = modal.Secret.from_name("visawise-secrets")

# TODO(you) 4 — the web endpoint. Decorator ORDER matters (function decorator
# outermost, asgi_app innermost):
#
#     @app.function(
#         image=image,
#         secrets=[secret],
#         min_containers=0,       # scale to zero -> $0 idle; ~5-15s cold start
#         scaledown_window=300,   # keep a container warm 5 min after last request
#         memory=4096,            # MB: torch + 2 bge models + LanceDB + headroom
#         cpu=2.0,                # cores: the cross-encoder rerank is CPU-bound
#         timeout=600,            # s: generous for cold-start model warmup
#     )
#     @modal.concurrent(max_inputs=8)   # allow concurrent SSE streams per container
#     @modal.asgi_app()
#     def fastapi_app():
#         # import INSIDE the function: it must run in the container (where the
#         # package is installed), not at deploy time on your laptop.
#         from visawise.app.main import app as web_app
#         return web_app
@app.function(
    image=image,
    secrets=[secret],
    min_containers=0,
    scaledown_window=300,
    memory=4096,
    cpu=2.0,
    timeout=600,
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def fastapi_app():
    from visawise.app.main import app as web_app
    return web_app