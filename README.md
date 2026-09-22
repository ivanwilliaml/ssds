# SSDS 2026 — Water-Level Prediction & Data-Quality Audit

Predicting water level (TMA) at monitoring posts, plus a standalone unsupervised
data-quality auditor for the test set.

## Open this first
- [`SSDS2026_rebuild_v7.ipynb`](./SSDS2026_rebuild_v7.ipynb) — the main prediction model.
- [`anomali_detector.ipynb`](./anomali_detector.ipynb) — a separate unsupervised model
  that scores test rows by how far they deviate from a post's normal behavior, surfacing
  likely recording errors for manual review. Uses only official competition data, no test
  labels.
- [`ERROR_ANALYSIS.md`](./ERROR_ANALYSIS.md) — error-analysis notes.

No dataset is committed here.
