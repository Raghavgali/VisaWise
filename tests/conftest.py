"""Keep the suite hermetic: identical behavior with or without a .env.

Two invariants, both enforced before any test module runs (conftest is
imported first):

1. Telemetry OFF. The app module calls init_observability() at import time;
   with real Langfuse keys in a developer's .env that would install a global
   tracer provider and export TEST spans to the production Langfuse project.

2. Provider keys PRESENT but fake. build_graph() constructs the LLM client
   eagerly and refuses to run without a key string — with keys only in .env,
   the suite passes on a developer laptop and fails on a keyless CI runner.
   No test may ever depend on a real key: these placeholders guarantee any
   test that actually calls a provider fails loudly everywhere.
"""

from visawise.config import settings

settings.langfuse_public_key = ""
settings.langfuse_secret_key = ""
settings.otel_exporter_otlp_endpoint = ""
settings.otel_exporter_otlp_headers = ""

settings.google_api_key = "test-key-not-real"
settings.groq_api_key = "test-key-not-real"
settings.nvidia_api_key = "test-key-not-real"
settings.openai_api_key = "test-key-not-real"
