/* VisaWise frontend — wired to the FastAPI backend.
   POST /api/chat streams SSE events: token {text} · done {answer, citations,
   disclaimer} · error {message}. GET /api/eval/runs feeds the docket. */

"use strict";

const ENGINE_UNREACHABLE =
  "VisaWise couldn't reach the answer engine just now. Try again in a moment.";
const RATE_LIMITED =
  "You're asking faster than the rate limit allows. Wait a minute and try again.";

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

async function streamAsk(message, { onToken, onDone }) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });

  if (response.status === 429) throw new Error(RATE_LIMITED);
  if (!response.ok || !response.body) throw new Error(ENGINE_UNREACHABLE);

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
  body.appendChild(caret);

  sendButton.disabled = true;
  scrollToEnd(true);

  let accumulated = "";

  try {
    await streamAsk(message, {
      onToken(text) {
        accumulated += text;
        body.innerHTML = renderAnswerHtml(accumulated);
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
  if (/8b/i.test(llm)) return "8B";
  if (/70b/i.test(llm)) return "70B";
  return llm.split("/").pop();
}

function isServingConfig(run) {
  const e = run;
  return (
    !e.aborted &&
    e.retriever === "hybrid" &&
    e.reranker === "local" &&
    /8b/i.test(e.llm ?? "8b") // llm absent in older summaries -> assume serving default
  );
}

function fmt(value, decimals = 3) {
  return value == null ? "—" : Number(value).toFixed(decimals);
}

function renderDocket(runs) {
  const tbody = document.querySelector(".docket tbody");
  if (!tbody || !runs?.length) return;
  tbody.textContent = "";

  let servingMarked = false;
  for (const run of runs) {
    const tr = document.createElement("tr");
    if (run.aborted) tr.className = "aborted";

    const weight =
      run.retriever === "hybrid" && run.vector_weight != null
        ? ` ${run.vector_weight}/${(1 - run.vector_weight).toFixed(1)}`
        : "";
    const label = [
      `${run.retriever}${weight}`,
      run.reranker === "none" ? "no rerank" : "rerank",
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
      fmt(run.faithfulness),
      fmt(run.response_relevancy),
      fmt(run.context_precision),
      fmt(run.mrr),
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
    const response = await fetch("/api/eval/runs");
    if (!response.ok) return; // keep the baked-in rows
    const summary = await response.json();
    renderDocket(summary.runs);
  } catch {
    /* offline/static preview: baked-in rows stand */
  }
}

loadDocket();
