# MLflow Integration for PaddleOCR Training

## Metadata
- Plan ID: mlflow-integration-paddleocr-001
- Created: 2026-03-24
- Revised: 2026-03-24 (Critic feedback incorporated)
- Source Spec: `.omc/specs/deep-interview-mlflow-paddleocr.md`
- Mode: CONSENSUS (RALPLAN-DR)
- Status: REVISED

---

## Context

PaddleOCR currently supports W&B for experiment tracking via `WandbLogger`. The user requires MLflow integration to connect to a remote MLflow Tracking Server and record complete experiment artifacts (metrics, models, configs, logs).

### Existing Architecture
```
ppocr/utils/loggers/
  base_logger.py      # Abstract base: log_metrics(), close()
  wandb_logger.py     # W&B implementation
  loggers.py          # Aggregator wrapper
  __init__.py         # Exports

tools/program.py      # Training entry point
  Lines 938-952: Logger initialization from config
```

### Key Interfaces

**BaseLogger (abstract class):**
```python
class BaseLogger(ABC):
    def __init__(self, save_dir):
        self.save_dir = save_dir

    @abstractmethod
    def log_metrics(self, metrics, prefix=None):
        pass

    @abstractmethod
    def close(self):
        pass
```

**Loggers (aggregator wrapper):**
```python
class Loggers(object):
    def log_metrics(self, metrics, prefix=None, step=None):
        for logger in self.loggers:
            logger.log_metrics(metrics, prefix=prefix, step=step)  # Passes step!

    def log_model(self, is_best, prefix, metadata=None):
        for logger in self.loggers:
            logger.log_model(is_best=is_best, prefix=prefix, metadata=metadata)

    def close(self):
        for logger in self.loggers:
            logger.close()
```

**WandbLogger (reference implementation):**
```python
class WandbLogger(BaseLogger):
    def log_metrics(self, metrics, prefix=None, step=None):  # EXTENDS base with step
        updated_metrics = {prefix.lower() + "/" + k: v for k, v in metrics.items()}
        self.run.log(updated_metrics, step=step)
```

> **IMPORTANT INTERFACE NOTE:** `BaseLogger` declares `log_metrics(metrics, prefix=None)` WITHOUT `step`.
> However, `Loggers` wrapper passes `step=None` to all loggers, and `WandbLogger` extends the interface
> to accept `step`. `MLflowLogger` must follow this pattern - extend the interface with `step` parameter
> to match `WandbLogger`. This is an intentional extension, not a violation.

---

## RALPLAN-DR Summary

### Principles
1. **Follow existing patterns**: Inherit `BaseLogger`, mirror `WandbLogger` structure
2. **Minimal invasiveness**: Add new files only, no core training logic changes
3. **Configuration priority**: Environment variables > YAML config > CLI defaults
4. **Optional installation**: `pip install mlflow`, graceful error if missing

### Decision Drivers (Top 3)
1. **Remote server requirement**: User explicitly needs remote MLflow Tracking Server
2. **Complete experiment tracking**: Must record metrics, models, configs, and logs
3. **Integration simplicity**: Leverage existing `BaseLogger` abstraction

### Viable Options Considered

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A: MLflow Python SDK (CHOSEN)** | Use `mlflow` package with remote tracking URI | Mature SDK, simple integration, community support | Requires MLflow server setup |
| B: Custom HTTP client | Direct REST API calls to MLflow | No SDK dependency | High implementation effort, must handle auth/retry |

### Decision: Option A (MLflow Python SDK)

**Why chosen:**
1. MLflow SDK is mature and well-documented
2. Remote server is the primary use case (user requirement)
3. Environment variable configuration is standard MLflow practice
4. Low implementation cost following `WandbLogger` pattern

**Consequences:**
- Users must have access to a remote MLflow Tracking Server
- `mlflow` package added as optional dependency
- Can extend to Model Registry in future if needed

---

## Work Objectives

1. Create `MLflowLogger` class inheriting from `BaseLogger`
2. Support remote MLflow Tracking Server via environment variable or YAML config
3. Record training metrics, model artifacts, configs, and logs
4. Integrate with existing `Loggers` aggregator
5. Enable via CLI flag or YAML configuration

---

## Guardrails

### Must Have
- [x] Inherit from `BaseLogger` abstract class
- [x] Support `MLFLOW_TRACKING_URI` environment variable
- [x] Support YAML config: `mlflow.tracking_uri`, `mlflow.experiment_name`
- [x] Record metrics: loss, lr, batch_time, IPS, memory, eval_metrics
- [x] Record artifacts: .pdparams, .pdopt, .states, .info.json (conditional)
- [x] Record training config as parameters
- [x] Record training log file as artifact
- [x] Handle `uniform_output_enabled` mode for artifact paths

### Must NOT Have
- [ ] Do NOT modify W&B logger code
- [ ] Do NOT implement MLflow Model Registry (out of scope)
- [ ] Do NOT support local file storage mode (remote only per spec)
- [ ] Do NOT require code changes to core training loop

---

## Task Flow

```
+-------------------------------------------------------------+
| Step 1: Create MLflowLogger class                           |
|   - Inherit BaseLogger                                      |
|   - Extend log_metrics() with step param (like WandbLogger) |
|   - Implement log_model(), close()                          |
|   - Handle tracking URI from env/config                     |
|   - Handle uniform_output_enabled for artifact paths        |
|   - Conditional .info.json logging                          |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| Step 2: Register MLflowLogger in __init__.py                |
|   - Export MLflowLogger class                               |
|   - Update Loggers if needed                                |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| Step 3: Integrate with tools/program.py                     |
|   - Add MLflow config parsing (similar to wandb)            |
|   - Initialize MLflowLogger when enabled                    |
|   - Add to loggers list                                     |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| Step 4: Add documentation and examples                      |
|   - YAML config example                                     |
|   - Usage documentation                                     |
|   - Update requirements (optional dep)                      |
+-------------------------------------------------------------+
```

---

## Detailed TODOs

### Step 1: Create MLflowLogger Class
**File:** `ppocr/utils/loggers/mlflow_logger.py`

**Tasks:**
- [x] Create class `MLflowLogger(BaseLogger)`
- [x] `__init__(tracking_uri, experiment_name, run_name, save_dir, config, uniform_output_enabled=False)`
  - Check for `mlflow` package, raise `ModuleNotFoundError` with install hint if missing
  - Set tracking URI: priority `MLFLOW_TRACKING_URI` env > config param
  - Set or create experiment
  - Start MLflow run with `mlflow.start_run()`
  - Log config as parameters via `mlflow.log_params()`
  - Store `uniform_output_enabled` flag for artifact path resolution
  - Store `save_dir` for log file path derivation
- [x] `log_metrics(metrics, prefix=None, step=None)` **[EXTENDS BASE INTERFACE]**
  - **Note:** This extends `BaseLogger.log_metrics(metrics, prefix=None)` with `step` param
  - **Rationale:** `Loggers` wrapper passes `step`, and `WandbLogger` follows same pattern
  - Format metric keys with prefix (e.g., `train/loss`, `eval/accuracy`)
  - Call `mlflow.log_metrics(formatted_metrics, step=step)`
  - Handle `step=None` gracefully (MLflow accepts None)
- [x] `log_model(is_best, prefix, metadata=None)`
  - **Resolve artifact base path:**
    ```python
    if self.uniform_output_enabled:
        artifact_dir = os.path.join(self.save_dir, prefix)
    else:
        artifact_dir = self.save_dir
    ```
  - Log `.pdparams` file: `os.path.join(artifact_dir, f"{prefix}.pdparams")`
  - Log `.pdopt` file: `os.path.join(artifact_dir, f"{prefix}.pdopt")`
  - Log `.states` file: `os.path.join(artifact_dir, f"{prefix}.states")`
  - **Conditionally log `.info.json`** (ONLY if file exists):
    ```python
    info_path = os.path.join(artifact_dir, f"{prefix}.info.json")
    if os.path.exists(info_path):
        mlflow.log_artifact(info_path, artifact_path="model")
    ```
  - Tag run with `best` if `is_best=True`
- [x] **Add method `log_training_log()`** (call at end of training)
  - Derive log file path: `os.path.join(self.save_dir, "train.log")`
  - Check file existence before logging
  - Log as artifact with `mlflow.log_artifact(log_path, artifact_path="logs")`
- [x] `close()`
  - Call `self.log_training_log()` to record training log
  - Call `mlflow.end_run()`
- [x] **Error handling specifics:**
  ```python
  import logging
  self.logger = logging.getLogger(__name__)

  # In each method:
  try:
      # MLflow operations
  except mlflow.exceptions.MlflowException as e:
      self.logger.warning(f"MLflow logging failed: {e}")
      # Continue training - don't crash on logging failures
  except Exception as e:
      self.logger.error(f"Unexpected error in MLflowLogger: {e}")
  ```

**Acceptance Criteria:**
- Class compiles without errors
- All abstract methods implemented
- `step` parameter accepted (extends base interface)
- Tracking URI configurable via environment variable
- Artifact paths correctly handle `uniform_output_enabled` mode
- `.info.json` only logged if file exists
- Training log file logged at close
- Errors logged but don't crash training

---

### Step 2: Register MLflowLogger
**File:** `ppocr/utils/loggers/__init__.py`

**Tasks:**
- [x] Add `from .mlflow_logger import MLflowLogger` import
- [x] Export `MLflowLogger` in `__all__` if present
- [x] Update `Loggers` class if interface changes needed (should not be needed)

**Acceptance Criteria:**
- `from ppocr.utils.loggers import MLflowLogger` works
- No import errors

---

### Step 3: Integrate with Training Pipeline
**File:** `tools/program.py`

**Tasks:**
- [x] Add import for `MLflowLogger`
- [x] Add MLflow config parsing (around lines 938-952, after wandb section):
  ```python
  if config.get("mlflow", {}).get("use_mlflow", False) or "mlflow" in config:
      mlflow_params = config.get("mlflow", {})
      mlflow_params["save_dir"] = save_model_dir
      mlflow_params["uniform_output_enabled"] = config["Global"].get("uniform_output_enabled", False)
      log_writer = MLflowLogger(**mlflow_params, config=config)
      loggers.append(log_writer)
  ```
- [x] Support CLI flag `--mlflow` to enable (optional enhancement)
- [x] Ensure `log_writer` is passed through `Loggers` wrapper

**Acceptance Criteria:**
- MLflowLogger initialized when config has `mlflow` section
- Metrics logged during training
- Artifacts logged at training end
- `uniform_output_enabled` flag passed correctly

---

### Step 4: Documentation and Examples
**Files:** `doc/`, `requirements.txt`

**Tasks:**
- [ ] Add `mlflow>=2.0.0` to requirements (optional section or comment)
- [ ] Create YAML config example:
  ```yaml
  mlflow:
    use_mlflow: true
    tracking_uri: "http://your-mlflow-server:5000"
    experiment_name: "paddleocr-experiments"
    run_name: null  # auto-generated if null
  ```
- [ ] Add usage documentation:
  - Environment variable: `export MLFLOW_TRACKING_URI=http://server:5000`
  - **MLflow Authentication:** MLflow SDK automatically uses `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` environment variables for basic auth
  - YAML configuration example
  - How to view results in MLflow UI

**Acceptance Criteria:**
- Users can configure MLflow via YAML
- Environment variable override documented
- Authentication via env vars documented
- Example config file provided

---

## Success Criteria

| Criterion | Verification Method |
|-----------|---------------------|
| MLflowLogger inherits BaseLogger | Code review, type checking |
| `step` param extends base interface | Compare with WandbLogger pattern |
| Remote server connection works | Manual test with MLflow server |
| Metrics appear in MLflow UI | Visual verification in UI |
| Model artifacts logged (correct paths) | Check artifacts in MLflow UI, test with uniform_output_enabled |
| Config params logged | Check parameters in MLflow UI |
| Training log logged as artifact | Check artifacts in MLflow UI |
| .info.json logged conditionally | Verify file existence check |
| CLI/YAML enable/disable works | Manual test both methods |
| Auth works with env vars | Test with MLFLOW_TRACKING_USERNAME/PASSWORD |

---

## ADR (Architecture Decision Record)

### Decision
Use MLflow Python SDK with remote tracking URI for experiment tracking.

### Drivers
1. User requirement for remote MLflow server
2. Need for complete experiment tracking (metrics, models, configs)
3. Existing `BaseLogger` abstraction enables clean integration

### Alternatives Considered
1. **Local file storage**: Rejected - user explicitly needs remote server
2. **Custom HTTP client**: Rejected - high implementation effort, MLflow SDK is mature
3. **W&B only**: Rejected - user does not care about W&B

### Why Chosen
- MLflow SDK provides complete functionality with minimal code
- Remote tracking is native to MLflow
- Pattern matches existing `WandbLogger` implementation

### Consequences
- Positive: Quick implementation, standard MLflow patterns
- Positive: Easy to extend with Model Registry later
- Positive: Authentication handled automatically by MLflow SDK
- Negative: Requires users to set up MLflow server
- Negative: Additional dependency (`mlflow` package)

### Follow-ups
- [ ] Add unit tests for MLflowLogger (future iteration)
- [ ] Consider Model Registry support if requested
- [x] Authentication support via MLFLOW_TRACKING_USERNAME/PASSWORD env vars (automatic)

---

## Implementation Notes

### Interface Extension Pattern
`MLflowLogger` extends `BaseLogger` interface with `step` parameter in `log_metrics()`:
- **BaseLogger**: `log_metrics(self, metrics, prefix=None)` - abstract
- **MLflowLogger**: `log_metrics(self, metrics, prefix=None, step=None)` - extends with optional param
- **Consistency**: Matches `WandbLogger` pattern exactly
- **Compatibility**: `Loggers` wrapper already passes `step` to all loggers

### Artifact Path Resolution
```python
def _get_artifact_dir(self, prefix):
    if self.uniform_output_enabled:
        return os.path.join(self.save_dir, prefix)
    return self.save_dir
```

### Training Log File Path
- Always: `os.path.join(self.save_dir, "train.log")`
- Derived from same `save_dir` passed to logger constructor
- Logged at `close()` time, not during training

### Conditional .info.json Logging
- `.info.json` is created by `save_model()` ONLY when `save_model_info=True`
- MLflowLogger must check file existence before logging:
  ```python
  info_path = os.path.join(artifact_dir, f"{prefix}.info.json")
  if os.path.exists(info_path):
      mlflow.log_artifact(info_path, artifact_path="model")
  ```

### MLflow Authentication
MLflow SDK automatically reads these environment variables:
- `MLFLOW_TRACKING_USERNAME` - username for basic auth
- `MLFLOW_TRACKING_PASSWORD` - password for basic auth
- No code changes needed - SDK handles authentication automatically

---

## Open Questions

Tracked in: `.omc/plans/open-questions.md`

- [x] ~~Should `--mlflow` CLI flag override YAML config or require it?~~ (Deferred - optional enhancement)
- [x] ~~What is the default experiment name if not specified?~~ (Use "PaddleOCR" as default)
- [x] ~~Should training log file path be configurable or always use default location?~~ (Always use `{save_dir}/train.log`)

---

## Estimated Effort
- **Complexity:** MEDIUM
- **Files Modified:** 3-4
- **New Files:** 1
- **Estimated Time:** 2-4 hours for implementation + testing
