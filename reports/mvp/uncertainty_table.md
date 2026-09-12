# Uncertainty (A2 backbone)

| Method | AUSE↓ | Spearman ρ↑ | 50% coverage | 90% coverage | Cost (× forward) | Collapse verdict | Split / n |
|---|---|---|---|---|---|---|---|
| TTA-4 | 0.0019 | 0.525 | — | — | — | — | val / 128 |
| TTA-8 | 0.0018 | 0.542 | — | — | — | — | val / 128 |
| Learned Laplace | 0.0027 | 0.379 | {'nominal': 0.5, 'empirical': 0.5326952338218689} | {'nominal': 0.9, 'empirical': 0.9039052724838257} | — | — | val / 128 |
