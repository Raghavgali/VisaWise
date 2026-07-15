# VisaWise Eval Leaderboard

Generated 2026-07-15T18:36:06.135348+00:00 -- 6 run(s).

| run_id | retriever | w_vec | reranker | dataset | n | faith | relevancy | ctx_prec | ctx_recall | hit_rate | mrr | ndcg | p50_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260715T175949Z_hybrid_no_rerank | hybrid | 0.600 | none | synthetic_v2 | 52 | 0.945 | 0.800 | 0.887 | 0.928 | 0.885 | 0.798 | 0.802 | 3266 |
| 20260715T181259Z_hybrid_rerank_full | hybrid | 0.600 | local | synthetic_v2 | 52 | 0.929 | 0.829 | 0.887 | 0.937 | 0.885 | 0.838 | 0.826 | 6089 |
| 20260715T182200Z_hybrid_default | hybrid | 0.600 | local | synthetic_v2 | 52 | 0.913 | 0.786 | 0.925 | 0.933 | 0.865 | 0.833 | 0.813 | 4053 |
| 20260715T175026Z_vector_only | vector | 0.600 | none | synthetic_v2 | 52 | 0.908 | 0.803 | 0.849 | 0.973 | 0.885 | 0.777 | 0.781 | 9062 |
| 20260715T183211Z_model_ablation@llm=nvidia-meta-llama-3.1-8b-instruct | hybrid | 0.600 | local | synthetic_v2 | 52 | 0.889 | 0.786 | 0.925 | 0.938 | 0.865 | 0.833 | 0.813 | 5028 |
| 20260711T213726Z_model_ablation@llm=nvidia-meta-llama-3.3-70b-instruct | hybrid | 0.600 | local | synthetic_v1 | 5 ⚠ aborted | 0.901 ⚠4/5 | 0.735 | 0.728 | 1.000 | 0.800 | 0.800 | 0.800 | 186636 |
