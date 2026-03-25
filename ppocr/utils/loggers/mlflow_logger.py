import json
import numbers
import os
import tempfile

import numpy as np

from ppocr.utils.logging import get_logger

from .base_logger import BaseLogger


class MLflowLogger(BaseLogger):
    """MLflow logger for tracking experiments with remote MLflow Tracking Server.

    Configuration priority:
    1. MLFLOW_TRACKING_URI environment variable
    2. tracking_uri in YAML config
    3. Default (local)

    Usage in YAML config:
        mlflow:
            use_mlflow: true
            tracking_uri: "http://your-mlflow-server:5000"
            experiment_name: "paddleocr-experiments"
            run_name: null  # auto-generated if null
    """

    def __init__(
        self,
        tracking_uri=None,
        experiment_name=None,
        run_name=None,
        save_dir=None,
        config=None,
        uniform_output_enabled=False,
        **kwargs,
    ):
        try:
            import mlflow

            self.mlflow = mlflow
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install mlflow using `pip install mlflow`"
            )

        self.save_dir = save_dir
        self.config = config
        self.uniform_output_enabled = uniform_output_enabled
        self.experiment_name = experiment_name or "PaddleOCR"
        self.run_name = run_name
        self.logger = get_logger()
        self._run = None
        self._owns_run = False
        self.kwargs = kwargs

        # Set tracking URI with priority: env var > config param > default
        self.tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", tracking_uri)
        if self.tracking_uri:
            self.mlflow.set_tracking_uri(self.tracking_uri)
            self.logger.info(f"MLflow tracking URI set to: {self.tracking_uri}")

        self._init_run()

        # Log config as parameters if provided
        if self.config:
            self._log_config_params(self.config)

    def _init_run(self):
        """Reuse active run if present, otherwise start a new run."""
        active_run = None
        if hasattr(self.mlflow, "active_run"):
            try:
                active_run = self.mlflow.active_run()
            except Exception as e:
                self.logger.warning(f"Unable to query active MLflow run: {e}")

        if active_run is not None:
            self._run = active_run
            self._owns_run = False
            self.logger.info("Reusing existing active MLflow run")
            return

        self.mlflow.set_experiment(self.experiment_name)
        self._run = self.mlflow.start_run(run_name=self.run_name)
        self._owns_run = True

    def _sanitize_param_value(self, value):
        if isinstance(value, (dict, list, tuple, set)):
            return str(value)[:500]
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return value.item()
            return str(value.tolist())[:500]
        if isinstance(value, np.generic):
            return value.item()
        if value is None:
            return "None"
        return value

    def _safe_log_param(self, key, value):
        sanitized = self._sanitize_param_value(value)
        try:
            self.mlflow.log_param(key, sanitized)
        except Exception:
            # Fallback to string for unsupported or long values.
            try:
                self.mlflow.log_param(key, str(sanitized)[:500])
            except Exception:
                self.logger.warning(f"Failed to log param to MLflow: {key}")

    def _log_config_params(self, config, prefix=""):
        """Recursively log config as MLflow parameters."""
        if isinstance(config, dict):
            for key, value in config.items():
                new_prefix = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(value, dict):
                    self._log_config_params(value, new_prefix)
                elif isinstance(value, list):
                    for idx, item in enumerate(value):
                        self._log_config_params(item, f"{new_prefix}[{idx}]")
                else:
                    self._safe_log_param(new_prefix, value)
            return

        # Base case for non-dict values.
        self._safe_log_param(prefix, config)

    @staticmethod
    def _to_mlflow_metric_value(value):
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, numbers.Real):
            return float(value)
        if isinstance(value, np.generic):
            return float(value.item())
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return float(value.item())
            return None
        return None

    def log_metrics(self, metrics, prefix=None, step=None):
        """Log metrics to MLflow.

        Args:
            metrics: dict of metric names and values
            prefix: optional prefix for metric names (e.g., "train", "eval")
            step: training step number
        """
        if not prefix:
            prefix = ""

        formatted_metrics = {}
        for k, v in metrics.items():
            metric_value = self._to_mlflow_metric_value(v)
            if metric_value is None:
                continue
            if prefix:
                metric_key = f"{prefix.lower()}/{k}"
            else:
                metric_key = k
            formatted_metrics[metric_key] = metric_value

        if not formatted_metrics:
            return

        try:
            self.mlflow.log_metrics(formatted_metrics, step=step)
        except Exception as e:
            self.logger.warning(f"Failed to log metrics to MLflow: {e}")

    def _get_artifact_dir(self, prefix):
        """Get the correct artifact directory based on uniform_output_enabled.

        When uniform_output_enabled=True, artifacts are saved in subdirectories
        like {save_dir}/{prefix}/{prefix}.pdparams instead of {save_dir}/{prefix}.pdparams
        """
        if self.uniform_output_enabled:
            return os.path.join(self.save_dir, prefix)
        return self.save_dir

    def log_model(self, is_best, prefix, metadata=None):
        """Log model artifacts to MLflow.

        Args:
            is_best: whether this is the best model so far
            prefix: model file prefix (e.g., "best_accuracy", "latest")
            metadata: optional metadata dict
        """
        if not self.save_dir:
            self.logger.warning("save_dir not set, skipping model logging")
            return

        try:
            # Get correct artifact directory based on uniform_output_enabled
            artifact_dir = self._get_artifact_dir(prefix)

            # Log model parameter file
            pdparams_path = os.path.join(artifact_dir, f"{prefix}.pdparams")
            if os.path.exists(pdparams_path):
                self.mlflow.log_artifact(pdparams_path, artifact_path="models")

            # Log optimizer state file
            pdopt_path = os.path.join(artifact_dir, f"{prefix}.pdopt")
            if os.path.exists(pdopt_path):
                self.mlflow.log_artifact(pdopt_path, artifact_path="models")

            # Log states file
            states_path = os.path.join(artifact_dir, f"{prefix}.states")
            if os.path.exists(states_path):
                self.mlflow.log_artifact(states_path, artifact_path="models")

            # Log info file (optional, only if created)
            info_path = os.path.join(artifact_dir, f"{prefix}.info.json")
            if os.path.exists(info_path):
                self.mlflow.log_artifact(info_path, artifact_path="models")

            # Log metadata if provided
            if metadata:
                if hasattr(self.mlflow, "log_dict"):
                    self.mlflow.log_dict(
                        metadata, artifact_file=f"models/{prefix}_metadata.json"
                    )
                else:
                    metadata_path = None
                    try:
                        tmp_dir = artifact_dir if os.path.isdir(artifact_dir) else None
                        with tempfile.NamedTemporaryFile(
                            mode="w",
                            suffix="_metadata.json",
                            delete=False,
                            dir=tmp_dir,
                        ) as tmp:
                            metadata_path = tmp.name
                            json.dump(metadata, tmp, indent=2, ensure_ascii=False)
                        self.mlflow.log_artifact(metadata_path, artifact_path="models")
                    finally:
                        if metadata_path and os.path.exists(metadata_path):
                            os.remove(metadata_path)

            # Tag as best if applicable
            if is_best:
                self.mlflow.set_tag("best", "true")
                self.logger.info(f"Tagged run as 'best' model: {prefix}")

        except Exception as e:
            self.logger.warning(f"Failed to log model artifacts to MLflow: {e}")

    def log_artifact(self, artifact_path, artifact_path_in_mlflow=None):
        """Log an arbitrary artifact file.

        Args:
            artifact_path: local path to the artifact file
            artifact_path_in_mlflow: optional path within MLflow artifact directory
        """
        try:
            if os.path.exists(artifact_path):
                self.mlflow.log_artifact(artifact_path, artifact_path_in_mlflow)
            else:
                self.logger.warning(f"Artifact not found: {artifact_path}")
        except Exception as e:
            self.logger.warning(f"Failed to log artifact to MLflow: {e}")

    def close(self):
        """End the MLflow run."""
        try:
            # Log training log file if available
            if self.save_dir:
                # Look for common log file locations
                log_patterns = ["train.log", "training.log", "output.log"]
                for log_name in log_patterns:
                    log_path = os.path.join(self.save_dir, log_name)
                    if os.path.exists(log_path):
                        self.mlflow.log_artifact(log_path, artifact_path="logs")
                        break

            if self._owns_run:
                self.mlflow.end_run()
                self.logger.info("MLflow run ended successfully")
        except Exception as e:
            self.logger.warning(f"Error closing MLflow run: {e}")
