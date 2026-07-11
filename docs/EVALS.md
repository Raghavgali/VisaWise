# VisaWise Eval Leaderboard

Generated 2026-07-11T22:59:55.936536+00:00 -- 4 run(s).

| run_id | retriever | w_vec | reranker | dataset | n | faith | relevancy | ctx_prec | ctx_recall | hit_rate | mrr | ndcg | p50_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260711T042930Z_vector_only | vector | 0.600 | none | synthetic_v1 | 52 | 0.866 | 0.806 | 0.836 | 0.970 | 0.898 | 0.784 | 0.788 | 3810 |
| 20260711T043643Z_hybrid_default | hybrid | 0.600 | local | synthetic_v1 | 52 | 0.864 | 0.792 | 0.916 | 0.938 | 0.878 | 0.844 | 0.822 | 3820 |
| 20260711T210755Z_model_ablation@llm=nvidia-meta-llama-3.1-8b-instruct | hybrid | 0.600 | local | synthetic_v1 | 52 | 0.836 | 0.805 | 0.915 | 0.948 | 0.878 | 0.844 | 0.822 | 3187 |
| 20260711T213726Z_model_ablation@llm=nvidia-meta-llama-3.3-70b-instruct | hybrid | 0.600 | local | synthetic_v1 | 5 ⚠ aborted | 0.901 | 0.735 | 0.728 | 1.000 | 0.800 | 0.800 | 0.800 | 186636 |
