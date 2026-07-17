# VisaWise Eval Leaderboard

Generated 2026-07-17T03:59:51.931469+00:00 -- 12 run(s).

| run_id | retriever | w_vec | reranker | dataset | n | faith | relevancy | ctx_prec | ctx_recall | hit_rate | mrr | ndcg | abstention | p50_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260715T175949Z_hybrid_no_rerank | hybrid | 0.600 | none | synthetic_v2 | 52 | 0.945 | 0.800 | 0.887 | 0.928 | 0.885 | 0.798 | 0.802 | — | 3266 |
| 20260715T181259Z_hybrid_rerank_full | hybrid | 0.600 | local | synthetic_v2 | 52 | 0.929 | 0.829 | 0.887 | 0.937 | 0.885 | 0.838 | 0.826 | — | 6089 |
| 20260717T032746Z_curated_v4_gemini | hybrid | 0.600 | local | curated_v1 | 30 | 0.916 | 0.583 | 0.959 | 0.881 | 0.958 ⚠24/30 | 0.816 ⚠24/30 | 0.775 ⚠24/30 | 0.667 | 2340 |
| 20260715T182200Z_hybrid_default | hybrid | 0.600 | local | synthetic_v2 | 52 | 0.913 | 0.786 | 0.925 | 0.933 | 0.865 | 0.833 | 0.813 | — | 4053 |
| 20260715T175026Z_vector_only | vector | 0.600 | none | synthetic_v2 | 52 | 0.908 | 0.803 | 0.849 | 0.973 | 0.885 | 0.777 | 0.781 | — | 9062 |
| 20260715T183211Z_model_ablation@llm=nvidia-meta-llama-3.1-8b-instruct | hybrid | 0.600 | local | synthetic_v2 | 52 | 0.889 | 0.786 | 0.925 | 0.938 | 0.865 | 0.833 | 0.813 | — | 5028 |
| 20260717T004426Z_curated_review | hybrid | 0.600 | local | curated_v1 | 30 | 0.863 | 0.752 | 0.943 | 0.864 | 0.958 ⚠24/30 | 0.816 ⚠24/30 | 0.775 ⚠24/30 | 0.133 | 4558 |
| 20260717T031201Z_guardrail_gemini_lite | hybrid | 0.600 | local | curated_v1 | 30 | 0.848 | 0.539 | 0.954 | 0.894 | 0.958 ⚠24/30 | 0.816 ⚠24/30 | 0.775 ⚠24/30 | 0.733 | 2178 |
| 20260717T022435Z_guardrail_model_ablation@llm=groq-openai-gpt-oss-20b | hybrid | 0.600 | local | curated_v1 | 30 | 0.829 | 0.739 | 0.954 | 0.881 | 0.958 ⚠24/30 | 0.816 ⚠24/30 | 0.775 ⚠24/30 | 0.733 | 31363 |
| 20260717T020806Z_guardrail_model_ablation@llm=groq-meta-llama-llama-4-scout-17b-16e-instruct | hybrid | 0.600 | local | curated_v1 | 30 | 0.743 | 0.574 | 0.948 | 0.848 | 0.958 ⚠24/30 | 0.816 ⚠24/30 | 0.775 ⚠24/30 | 0.333 | 7218 |
| 20260717T014941Z_curated_review_v2 | hybrid | 0.600 | local | curated_v1 | 30 | 0.546 | 0.252 | 0.954 | 0.859 | 0.958 ⚠24/30 | 0.816 ⚠24/30 | 0.775 ⚠24/30 | 0.467 | 2474 |
| 20260711T213726Z_model_ablation@llm=nvidia-meta-llama-3.3-70b-instruct | hybrid | 0.600 | local | synthetic_v1 | 5 ⚠ aborted | 0.901 ⚠4/5 | 0.735 | 0.728 | 1.000 | 0.800 | 0.800 | 0.800 | — | 186636 |

## Per-slice breakdown

### Per-slice — `20260717T032746Z_curated_v4_gemini`

| slice | n | faith | relevancy | ctx_prec | mrr | abstention |
|---|---|---|---|---|---|---|
| adversarial_injection | 5 | — | — | — | — | 0.800 |
| answerable_grounded | 5 | 0.920 | 0.417 | 0.878 | 0.767 | — |
| high_risk_abstain | 5 | — | — | — | — | 1.000 |
| out_of_corpus | 5 | — | — | — | — | 0.200 |
| stale_source | 5 | 0.827 | 0.534 | 1.000 | 0.900 | — |
| temporal_current | 5 | 1.000 | 0.797 | 1.000 | 1.000 | — |

### Per-slice — `20260717T004426Z_curated_review`

| slice | n | faith | relevancy | ctx_prec | mrr | abstention |
|---|---|---|---|---|---|---|
| adversarial_injection | 5 | — | — | — | — | 0.200 |
| answerable_grounded | 5 | 0.899 | 0.745 | 0.878 | 0.767 | — |
| high_risk_abstain | 5 | — | — | — | — | 0.200 |
| out_of_corpus | 5 | — | — | — | — | 0.000 |
| stale_source | 5 | 0.896 | 0.664 | 0.950 | 0.900 | — |
| temporal_current | 5 | 0.793 | 0.847 | 1.000 | 1.000 | — |

### Per-slice — `20260717T031201Z_guardrail_gemini_lite`

| slice | n | faith | relevancy | ctx_prec | mrr | abstention |
|---|---|---|---|---|---|---|
| adversarial_injection | 5 | — | — | — | — | 0.800 |
| answerable_grounded | 5 | 0.922 | 0.284 | 0.878 | 0.767 | — |
| high_risk_abstain | 5 | — | — | — | — | 1.000 |
| out_of_corpus | 5 | — | — | — | — | 0.400 |
| stale_source | 5 | 0.800 | 0.593 | 1.000 | 0.900 | — |
| temporal_current | 5 | 0.823 | 0.739 | 0.983 | 1.000 | — |

### Per-slice — `20260717T022435Z_guardrail_model_ablation@llm=groq-openai-gpt-oss-20b`

| slice | n | faith | relevancy | ctx_prec | mrr | abstention |
|---|---|---|---|---|---|---|
| adversarial_injection | 5 | — | — | — | — | 0.800 |
| answerable_grounded | 5 | 0.695 | 0.744 | 0.878 | 0.767 | — |
| high_risk_abstain | 5 | — | — | — | — | 0.600 |
| out_of_corpus | 5 | — | — | — | — | 0.800 |
| stale_source | 5 | 0.860 | 0.709 | 1.000 | 0.900 | — |
| temporal_current | 5 | 0.933 | 0.766 | 0.983 | 1.000 | — |

### Per-slice — `20260717T020806Z_guardrail_model_ablation@llm=groq-meta-llama-llama-4-scout-17b-16e-instruct`

| slice | n | faith | relevancy | ctx_prec | mrr | abstention |
|---|---|---|---|---|---|---|
| adversarial_injection | 5 | — | — | — | — | 0.600 |
| answerable_grounded | 5 | 0.600 | 0.306 | 0.861 | 0.767 | — |
| high_risk_abstain | 5 | — | — | — | — | 0.200 |
| out_of_corpus | 5 | — | — | — | — | 0.200 |
| stale_source | 5 | 0.730 | 0.685 | 1.000 | 0.900 | — |
| temporal_current | 5 | 0.900 | 0.730 | 0.983 | 1.000 | — |

### Per-slice — `20260717T014941Z_curated_review_v2`

| slice | n | faith | relevancy | ctx_prec | mrr | abstention |
|---|---|---|---|---|---|---|
| adversarial_injection | 5 | — | — | — | — | 0.600 |
| answerable_grounded | 5 | 0.651 | 0.173 | 0.861 | 0.767 | — |
| high_risk_abstain | 5 | — | — | — | — | 0.200 |
| out_of_corpus | 5 | — | — | — | — | 0.600 |
| stale_source | 5 | 0.310 | 0.279 | 1.000 | 0.900 | — |
| temporal_current | 5 | 0.677 | 0.305 | 1.000 | 1.000 | — |
