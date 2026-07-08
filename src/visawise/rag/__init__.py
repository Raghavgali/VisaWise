"""RAG core: a LangGraph pipeline, built in ONE place.

`graph.build_graph(EngineConfig)` returns the compiled graph shared by the
FastAPI app and the eval harness, so evaluated configs are exactly what gets
served.

Graph shape (nodes are deliberately framework-thin -- direct LanceDB client
calls and hand-rolled weighted RRF, not retriever abstractions):

              +-> dense_retrieve  (LanceDB vector search) --+
    query --> |                                             +-> fuse (weighted RRF)
              +-> bm25_retrieve   (LanceDB tantivy FTS)  ---+        |
                                                                  rerank
                                                                     |
                                                                  generate (Groq)

Future agentic nodes (query rewrite, groundedness check) are drop-in graph
additions -- and each becomes an eval experiment, not a leap of faith.
"""
