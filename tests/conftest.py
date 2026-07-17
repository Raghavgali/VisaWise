"""Keep the suite hermetic: force telemetry off before any test module runs.

The app module calls init_observability() at import time; with real Langfuse
keys in the developer's .env that would install a global tracer provider and
export spans from the TEST SUITE to the production Langfuse project. Blank
the settings here — conftest is imported before any test module, so this
runs before visawise.app.main can initialize telemetry.
"""

from visawise.config import settings

settings.langfuse_public_key = ""
settings.langfuse_secret_key = ""
settings.otel_exporter_otlp_endpoint = ""
settings.otel_exporter_otlp_headers = ""
