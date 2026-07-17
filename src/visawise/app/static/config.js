"use strict";
/* Backend API base URL. Loaded BEFORE app.js so it can read this global.
 *
 * ""  (empty) = same origin — correct when the FastAPI app serves this page
 *               itself (e.g. opening the Modal URL directly, or local dev).
 *
 * When the frontend is hosted separately (Vercel), set this to the Modal API
 * origin so the cross-origin fetch + SSE hit the backend, e.g.:
 *   window.__VISAWISE_API_BASE__ = "https://<workspace>--visawise-fastapi-app.modal.run";
 * (No trailing slash. The Modal origin must also be in the backend's
 *  CORS_ALLOWED_ORIGINS.)
 */
window.__VISAWISE_API_BASE__ = "";
