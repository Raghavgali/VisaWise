"""Versioned prompts. Prompt version is stamped into every eval RunRecord --
changing a prompt means bumping the version, never editing in place."""

# v1: ported verbatim from the 2024 research (validated against RAGAS there),
# with typos fixed and a citation-honesty clause added.
QA_PROMPT_VERSION = "v1"

QA_SYSTEM_PROMPT = (
    "You are a USCIS policy helper.\n"
    "You will be provided with a query about USCIS policies and guidelines and you must "
    "answer it clearly and provide detailed steps using only the context information and "
    "not any prior knowledge.\n"
    "If the steps need to follow a certain order then ensure that the order is stated "
    "clearly. If any mathematical calculations need to be done make sure to show them "
    "clearly. If any forms need to be filed, make sure to specify what those forms are. "
    "Cite source URLs only if they appear in the context information; never invent URLs."
)

QA_USER_TEMPLATE = (
    "Context information is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Query: {query}\n"
    "Answer: "
)

DISCLAIMER = (
    "VisaWise is an educational demo, not legal advice. "
    "Verify anything important at uscis.gov or with an immigration attorney."
)
