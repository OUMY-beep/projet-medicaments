import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
c = MlflowClient()
for v in sorted(c.search_model_versions("name='medicament-generique'"),
                key=lambda v: int(v.version)):
    run = c.get_run(v.run_id)
    print("version", v.version, "|", v.status,
          "| F1", run.data.metrics.get("f1"),
          "| max_iter", run.data.params.get("hp_max_iter"))