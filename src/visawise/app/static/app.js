/* VisaWise frontend — wired to the FastAPI backend.
   POST /api/chat streams SSE events: token {text} · done {answer, citations,
   disclaimer} · error {message}. GET /api/eval/runs feeds the docket. */

"use strict";

const ENGINE_UNREACHABLE =
  "VisaWise couldn't reach the answer engine just now. Try again in a moment.";
const RATE_LIMITED =
  "You're asking faster than the rate limit allows. Wait a minute and try again.";

/* Backend origin. "" = same origin; set by config.js (loaded before this file)
   to the Modal URL when the frontend is served from Vercel. */
const API_BASE = window.__VISAWISE_API_BASE__ || "";

/* Pre-warm the scale-to-zero backend the moment the page loads: the cold
   start (~30s) then happens while the visitor is still reading, not after
   they've asked their first question. Fire-and-forget; a 503 is fine — it's
   the warm-up itself we're after. */
fetch(API_BASE + "/health").catch(() => {});

const COLD_START_NOTICE =
  "Waking the answer engine — this demo scales to zero when idle, so the " +
  "first request can take 30–45 seconds. After that, answers are much faster.";

/* ---- theme ---------------------------------------------------------------- */

const themeToggle = document.getElementById("theme-toggle");

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("visawise-theme", theme);
  // the icon (CSS-swapped) shows the theme the button switches TO
  themeToggle.setAttribute("aria-label", `Switch to ${theme === "dark" ? "light" : "dark"} mode`);
}

applyTheme(document.documentElement.dataset.theme || "light");

themeToggle.addEventListener("click", () =>
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark")
);

/* ---- tabs ---------------------------------------------------------------- */

const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".panel");

function showTab(name) {
  tabs.forEach((t) =>
    t.dataset.tab === name
      ? t.setAttribute("aria-current", "page")
      : t.removeAttribute("aria-current")
  );
  panels.forEach((p) => (p.hidden = p.dataset.panel !== name));
}

function syncTabFromHash() {
  showTab(location.hash === "#evals" ? "evals" : "chat");
}

window.addEventListener("hashchange", syncTabFromHash);
syncTabFromHash();

/* ---- SSE transport --------------------------------------------------------- */

const FRAME_BOUNDARY = /\r?\n\r?\n/; // SSE spec allows CRLF (sse-starlette uses it)

function parseSseFrame(frame) {
  let event = "message";
  const dataLines = [];
  for (const line of frame.split(/\r?\n/)) {
    if (line.startsWith(":")) continue; // keep-alive comment
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return null;
  return { event, data: dataLines.join("\n") };
}

async function streamAsk(message, { onToken, onDone, onStatus }) {
  const response = await fetch(API_BASE + "/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });

  if (response.status === 429) throw new Error(RATE_LIMITED);
  if (!response.ok || !response.body) throw new Error(ENGINE_UNREACHABLE);

  // Stream open = the container is warm and working on this request.
  onStatus?.("retrieving");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false;

  for (;;) {
    const { value, done: eof } = await reader.read();
    if (eof) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary;
    while ((boundary = buffer.match(FRAME_BOUNDARY)) !== null) {
      const frame = parseSseFrame(buffer.slice(0, boundary.index));
      buffer = buffer.slice(boundary.index + boundary[0].length);
      if (!frame) continue;

      if (frame.event === "token") {
        onToken(JSON.parse(frame.data).text);
      } else if (frame.event === "status") {
        onStatus?.(JSON.parse(frame.data).stage);
      } else if (frame.event === "done") {
        onDone(JSON.parse(frame.data));
        finished = true;
      } else if (frame.event === "error") {
        throw new Error(JSON.parse(frame.data).message || ENGINE_UNREACHABLE);
      }
    }
  }

  if (!finished) throw new Error(ENGINE_UNREACHABLE);
}

/* ---- chat ----------------------------------------------------------------- */

const thread = document.getElementById("thread");
const intake = document.getElementById("intake");
const composer = document.getElementById("composer");
const textarea = document.getElementById("message");
const sendButton = composer.querySelector(".send");

const tplUser = document.getElementById("tpl-user");
const tplAnswer = document.getElementById("tpl-answer");

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function appendUserMessage(text) {
  const node = tplUser.content.cloneNode(true);
  node.querySelector(".msg-user-text").textContent = text;
  thread.appendChild(node);
}

function appendAnswerCard() {
  const node = tplAnswer.content.cloneNode(true);
  const card = node.querySelector(".determination");
  thread.appendChild(node);
  return card;
}

function renderCitations(card, citations) {
  const list = card.querySelector(".det-citations");
  list.textContent = "";
  for (const c of citations) {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = c.url;
    a.target = "_blank";
    a.rel = "noopener";
    a.textContent = c.url.replace("https://www.", "");
    a.title = c.title;
    li.appendChild(a);
    list.appendChild(li);
  }
}

/* ---- minimal markdown (model answers use bold + lists) ----------------------
   Escapes everything first, then rebuilds only: paragraphs, ordered lists,
   bullets (one nesting level), **bold**, `code`. No raw model HTML ever. */

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderInline(s) {
  return escapeHtml(s)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function renderAnswerHtml(text) {
  const out = [];
  const stack = []; // open lists, innermost last
  let paragraph = [];

  const closeLists = (toDepth = 0) => {
    while (stack.length > toDepth) out.push(`</${stack.pop()}>`);
  };
  const flushParagraph = () => {
    if (paragraph.length) {
      out.push(`<p>${paragraph.join("<br>")}</p>`);
      paragraph = [];
    }
  };

  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();
    const ordered = line.match(/^\s*\d+[.)]\s+(.*)/);
    const bullet = line.match(/^\s*[-*•]\s*(.+)/);

    if (ordered) {
      flushParagraph();
      if (stack[0] !== "ol") { closeLists(); out.push("<ol>"); stack.push("ol"); }
      else closeLists(1);
      out.push(`<li>${renderInline(ordered[1])}`);
    } else if (bullet) {
      flushParagraph();
      if (!stack.length) { out.push("<ul>"); stack.push("ul"); }
      else if (stack[stack.length - 1] !== "ul") { out.push("<ul>"); stack.push("ul"); }
      out.push(`<li>${renderInline(bullet[1])}`);
    } else if (!line.trim()) {
      flushParagraph(); // blank lines between list items keep the list open
    } else {
      closeLists();
      paragraph.push(renderInline(line));
    }
  }
  flushParagraph();
  closeLists();
  return out.join("");
}

/* Follow the stream only if the reader is already at the bottom — never
   fight a user who scrolled up (or is selecting text). */
function isNearBottom() {
  return window.innerHeight + window.scrollY >= document.body.scrollHeight - 160;
}

function scrollToEnd(force = false) {
  if (!force && !isNearBottom()) return;
  requestAnimationFrame(() =>
    window.scrollTo({ top: document.body.scrollHeight, behavior: "auto" })
  );
}

async function ask(message) {
  intake.hidden = true;
  appendUserMessage(message);

  const card = appendAnswerCard();
  const body = card.querySelector(".det-body");
  const receipt = card.querySelector(".det-receipt");
  receipt.hidden = true;

  const caret = document.createElement("span");
  caret.className = "caret";

  /* Progress line while there's nothing to stream yet. Lives inside .det-body,
     so the first token (which rewrites body.innerHTML) clears it naturally.
     Without this, a scale-to-zero cold start looks like a broken app: an
     empty card for 30+ seconds. */
  const status = document.createElement("p");
  status.className = "det-status";
  status.textContent = "Contacting the answer engine…";
  body.appendChild(status);
  body.appendChild(caret);

  // No stream open after a few seconds = we're paying the cold start; say so.
  const coldStartTimer = setTimeout(() => {
    status.textContent = COLD_START_NOTICE;
  }, 3000);

  const STAGE_LABELS = {
    retrieving: "Searching official USCIS sources…",
    generating: "Generating a cited answer…",
  };

  sendButton.disabled = true;
  scrollToEnd(true);

  let accumulated = "";

  try {
    await streamAsk(message, {
      onStatus(stage) {
        clearTimeout(coldStartTimer);
        if (STAGE_LABELS[stage]) status.textContent = STAGE_LABELS[stage];
      },
      onToken(text) {
        accumulated += text;
        body.innerHTML = renderAnswerHtml(accumulated); // clears status line
        body.appendChild(caret);
        scrollToEnd();
      },
      onDone(result) {
        caret.remove();
        body.innerHTML = renderAnswerHtml(result.answer);
        if (result.citations?.length) {
          renderCitations(card, result.citations);
          receipt.hidden = false;
        }
      },
    });
  } catch (error) {
    caret.remove();
    card.classList.add("det-error");
    body.textContent = error.message || ENGINE_UNREACHABLE;
  } finally {
    clearTimeout(coldStartTimer);
    sendButton.disabled = false;
    scrollToEnd();
  }
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = textarea.value.trim();
  if (!message || sendButton.disabled) return;
  textarea.value = "";
  autosize();
  ask(message);
});

textarea.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

function autosize() {
  textarea.style.height = "auto";
  textarea.style.height = `${Math.min(textarea.scrollHeight, 150)}px`;
}
textarea.addEventListener("input", autosize);

document.querySelectorAll(".starter").forEach((button) =>
  button.addEventListener("click", () => {
    if (sendButton.disabled) return;
    ask(button.textContent.trim());
  })
);

/* ?ask=<question> auto-fires a question (share links, screenshots). */
const presetQuestion = new URLSearchParams(location.search).get("ask");
if (presetQuestion) ask(presetQuestion);

/* ---- evaluation docket (live) ---------------------------------------------- */

function shortModel(llm) {
  if (!llm) return "";
  const m = llm.toLowerCase();
  if (m.includes("gemini")) return "Gemini Flash-Lite";
  if (m.includes("gpt-oss")) return "gpt-oss-20B";
  if (m.includes("scout")) return "Llama-4-Scout";
  if (m.includes("8b")) return "8B";
  if (m.includes("70b")) return "70B";
  return llm.split("/").pop();
}

// The deployed serving config is Gemini on the curated_v2 safety set.
function isServingConfig(run) {
  return !run.aborted && run.dataset === "curated_v2" && /gemini/i.test(run.llm ?? "");
}

/* Safety = share of hard-slice cases handled correctly, shown as a count
   (e.g. "14/15") using the abstention aggregate × its coverage total. */
function safetyCell(run) {
  const value = run.abstention;
  if (value == null) return "—";
  const total = run.coverage?.abstention?.total;
  return total ? `${Math.round(value * total)}/${total}` : fmt(value, 2);
}

function fmt(value, decimals = 3) {
  return value == null ? "—" : Number(value).toFixed(decimals);
}

/* Aggregate + coverage: "0.864" at full coverage, "0.866 ⚠29/52" when the
   judge only scored a subset — a partial average must never look complete. */
function metricCell(run, key) {
  const value = run[key];
  if (value == null) return "—";
  let cell = fmt(value);
  const cov = run.coverage?.[key];
  if (cov && cov.scored < cov.total) cell += ` ⚠${cov.scored}/${cov.total}`;
  return cell;
}

function renderDocket(runs) {
  const tbody = document.querySelector(".docket tbody");
  if (!tbody || !runs?.length) return;
  tbody.textContent = "";

  let servingMarked = false;
  for (const run of runs) {
    const tr = document.createElement("tr");
    if (run.aborted) tr.className = "aborted";

    // Contexts reaching the LLM: rerank_top_n when reranked, else top_k. This
    // is the axis the reranker ablation turns on, so it belongs in the label —
    // "rerank→4" vs "rerank→8" is the difference between two otherwise-identical rows.
    const rerankLabel =
      run.reranker === "none"
        ? "no rerank"
        : run.rerank_top_n != null
          ? `rerank→${run.rerank_top_n}`
          : "rerank";
    // Lead with the dataset so the three studies (retrieval / bake-off / safety)
    // are distinguishable in one flat table.
    const label = [
      run.dataset,
      `${run.retriever} ${rerankLabel}`,
      shortModel(run.llm) || null,
    ].filter(Boolean).join(" · ");

    const cfgCell = document.createElement("td");
    const cfg = document.createElement("span");
    cfg.className = "cfg";
    cfg.textContent = label;
    cfgCell.appendChild(cfg);

    if (run.aborted) {
      const badge = document.createElement("span");
      badge.className = "badge-aborted";
      badge.textContent = "Aborted";
      badge.title = "Run stopped early; too few samples to compare";
      cfgCell.appendChild(badge);
    } else if (!servingMarked && isServingConfig(run)) {
      servingMarked = true;
      tr.classList.add("serving");
      const badge = document.createElement("span");
      badge.className = "badge-serving";
      badge.textContent = "Serving";
      cfgCell.appendChild(badge);
    }
    tr.appendChild(cfgCell);

    const p50 = run.p50_ms == null ? null : run.p50_ms / 1000;
    const cells = [
      run.n_samples ?? "—",
      metricCell(run, "faithfulness"),
      metricCell(run, "response_relevancy"),
      metricCell(run, "context_precision"),
      metricCell(run, "mrr"),
      safetyCell(run),
      p50 == null ? "—" : `${p50.toFixed(1)}s`,
    ];
    for (const text of cells) {
      const td = document.createElement("td");
      td.className = "num";
      td.textContent = text;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

async function loadDocket() {
  try {
    const response = await fetch(API_BASE + "/api/eval/runs");
    if (!response.ok) return; // keep the baked-in rows
    const summary = await response.json();
    renderDocket(summary.runs);
  } catch {
    /* offline/static preview: baked-in rows stand */
  }
}

loadDocket();
