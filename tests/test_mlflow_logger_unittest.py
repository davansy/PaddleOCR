import sys
import types
import unittest


def _install_fake_paddle():
    if "paddle" in sys.modules and "paddle.distributed" in sys.modules:
        return

    fake_dist = types.ModuleType("paddle.distributed")
    fake_dist.get_rank = lambda: 0

    fake_paddle = types.ModuleType("paddle")
    fake_paddle.distributed = fake_dist

    sys.modules["paddle"] = fake_paddle
    sys.modules["paddle.distributed"] = fake_dist


class _FakeRun:
    def __init__(self, run_id="run-1"):
        self.info = types.SimpleNamespace(run_id=run_id)


class _FakeMLflow:
    def __init__(self, active_run=None, reject_non_numeric_metrics=False):
        self._active_run = active_run
        self.reject_non_numeric_metrics = reject_non_numeric_metrics
        self.tracking_uri = None
        self.experiment_name = None
        self.started_runs = []
        self.logged_params = []
        self.logged_metrics = []
        self.logged_artifacts = []
        self.ended = False

    def active_run(self):
        return self._active_run

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def set_experiment(self, experiment_name):
        self.experiment_name = experiment_name

    def start_run(self, run_name=None):
        run = _FakeRun(run_id=run_name or "generated")
        self.started_runs.append(run)
        self._active_run = run
        return run

    def log_param(self, key, value):
        self.logged_params.append((key, value))

    def log_metrics(self, metrics, step=None):
        if self.reject_non_numeric_metrics:
            for value in metrics.values():
                if not isinstance(value, (int, float)):
                    raise TypeError("metric value must be numeric")
        self.logged_metrics.append((metrics, step))

    def log_artifact(self, path, artifact_path=None):
        self.logged_artifacts.append((path, artifact_path))

    def set_tag(self, key, value):
        pass

    def end_run(self):
        self.ended = True
        self._active_run = None


class TestMLflowLogger(unittest.TestCase):
    def setUp(self):
        _install_fake_paddle()
        self._orig_mlflow = sys.modules.get("mlflow")

    def tearDown(self):
        if self._orig_mlflow is None:
            sys.modules.pop("mlflow", None)
        else:
            sys.modules["mlflow"] = self._orig_mlflow

    def test_reuses_existing_active_run(self):
        active_run = _FakeRun(run_id="already-active")
        fake_mlflow = _FakeMLflow(active_run=active_run)
        sys.modules["mlflow"] = fake_mlflow

        from ppocr.utils.loggers.mlflow_logger import MLflowLogger

        logger = MLflowLogger(experiment_name="exp")
        self.assertIs(logger._run, active_run)
        self.assertEqual(len(fake_mlflow.started_runs), 0)

    def test_log_metrics_skips_non_numeric_values(self):
        fake_mlflow = _FakeMLflow(reject_non_numeric_metrics=True)
        sys.modules["mlflow"] = fake_mlflow

        from ppocr.utils.loggers.mlflow_logger import MLflowLogger

        logger = MLflowLogger(experiment_name="exp")
        logger.log_metrics({"loss": 0.123, "status": "ok"}, prefix="TRAIN", step=7)

        self.assertEqual(len(fake_mlflow.logged_metrics), 1)
        metrics, step = fake_mlflow.logged_metrics[0]
        self.assertEqual(step, 7)
        self.assertEqual(metrics, {"train/loss": 0.123})

    def test_close_does_not_end_reused_active_run(self):
        active_run = _FakeRun(run_id="external")
        fake_mlflow = _FakeMLflow(active_run=active_run)
        sys.modules["mlflow"] = fake_mlflow

        from ppocr.utils.loggers.mlflow_logger import MLflowLogger

        logger = MLflowLogger(experiment_name="exp")
        logger.close()

        self.assertFalse(fake_mlflow.ended)


if __name__ == "__main__":
    unittest.main()
