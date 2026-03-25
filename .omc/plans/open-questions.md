# Open Questions Tracker

This file tracks unresolved questions across all plans that need user or team input before or during implementation.

---

## MLflow Integration for PaddleOCR - 2026-03-24

- [ ] **Should `--mlflow` CLI flag override YAML config or require it?**
  Why it matters: Affects implementation complexity and user experience. Override is more flexible but requires priority handling.

- [ ] **What is the default experiment name if not specified in config?**
  Why it matters: MLflow requires an experiment name. Proposed default: "PaddleOCR". User may want project-specific naming.

- [ ] **Should training log file path be configurable or always use default location?**
  Why it matters: Currently logs go to `save_model_dir`. Configurable path adds flexibility but increases config surface.

- [ ] **Should MLflow authentication be supported in initial implementation?**
  Why it matters: Remote servers may require authentication (basic auth, token, etc.). User has not specified auth requirements.
