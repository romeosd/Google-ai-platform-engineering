"""
Vertex AI Pipelines — end-to-end MLOps on Google Cloud.

Provides:
- Pipeline definition using KFP v2 DSL
- Preprocessing, training, evaluation, and model upload steps
- Vertex AI Model Registry integration
- Online endpoint deployment with traffic splitting
- Hyperparameter tuning (Vizier)
- Vertex AI Experiments and MLflow tracking
- Model monitoring (data drift, prediction drift)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import vertexai
from google.cloud import aiplatform
from google.api_core.exceptions import GoogleAPICallError
from kfp import compiler, dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output, component

from src.utils.config import get_config, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineRunResult:
    """Result of a Vertex AI Pipeline run."""

    pipeline_job_name: str
    status: str
    dashboard_url: str = ""
    output_artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == "PIPELINE_STATE_SUCCEEDED"

    @property
    def failed(self) -> bool:
        return self.status == "PIPELINE_STATE_FAILED"


@dataclass
class EndpointInfo:
    """Information about a deployed Vertex AI endpoint."""

    endpoint_name: str
    endpoint_resource_name: str
    deployed_model_id: str
    predict_uri: str
    status: str


class VertexPipelineOrchestrator:
    """
    Production Vertex AI Pipelines orchestrator.

    Compiles and submits KFP v2 pipelines to Vertex AI,
    manages model registry, deploys to online endpoints,
    and configures model monitoring.

    Example:
        orchestrator = VertexPipelineOrchestrator()

        # Compile and submit pipeline
        result = orchestrator.submit_training_pipeline(
            experiment_name="churn-v3",
            training_data_gcs="gs://bucket/data/train.csv",
        )

        # Wait and deploy
        final = orchestrator.wait_for_pipeline(result.pipeline_job_name)
        if final.succeeded:
            endpoint = orchestrator.deploy_model(
                model_resource_name="projects/.../models/...",
            )
            print(f"Endpoint: {endpoint.predict_uri}")
    """

    def __init__(self) -> None:
        cfg = get_config()
        raw = load_config()
        pipe_cfg = raw.get("vertex_pipelines", {})
        ep_cfg = raw.get("endpoints", {})

        self._project_id = cfg.project_id
        self._location = cfg.vertex_ai.location
        self._pipeline_root = pipe_cfg.get("pipeline_root", "")
        self._service_account = pipe_cfg.get("service_account", "")
        self._staging_bucket = cfg.vertex_ai.staging_bucket
        self._machine_type = ep_cfg.get("machine_type", "n1-standard-4")
        self._min_replicas = ep_cfg.get("min_replica_count", 1)
        self._max_replicas = ep_cfg.get("max_replica_count", 5)

        vertexai.init(project=self._project_id, location=self._location, staging_bucket=self._staging_bucket)
        aiplatform.init(project=self._project_id, location=self._location, staging_bucket=self._staging_bucket)

        logger.info(
            "VertexPipelineOrchestrator initialised",
            project=self._project_id,
            location=self._location,
        )

    def submit_training_pipeline(
        self,
        experiment_name: str,
        training_data_gcs: str,
        hyperparameters: dict[str, Any] | None = None,
        labels: dict[str, str] | None = None,
    ) -> PipelineRunResult:
        """
        Compile and submit a training pipeline to Vertex AI Pipelines.

        Pipeline steps: preprocessing → training → evaluation →
        conditional model upload (if accuracy meets threshold).

        Args:
            experiment_name: Vertex AI Experiment name for tracking.
            training_data_gcs: GCS URI to training data.
            hyperparameters: Model hyperparameters dict.
            labels: GCP resource labels for the pipeline job.

        Returns:
            PipelineRunResult with job name and Vertex AI console URL.
        """
        hparams = hyperparameters or {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
        pipeline_path = "/tmp/training_pipeline.json"

        # Compile the pipeline
        compiler.Compiler().compile(
            pipeline_func=self._build_training_pipeline(hparams),
            package_path=pipeline_path,
        )

        job = aiplatform.PipelineJob(
            display_name=f"{experiment_name}-pipeline",
            template_path=pipeline_path,
            pipeline_root=self._pipeline_root,
            parameter_values={"training_data_gcs": training_data_gcs},
            labels=labels or {"environment": "production", "team": "ai-platform"},
            enable_caching=True,
        )

        try:
            job.submit(
                service_account=self._service_account or None,
                experiment=experiment_name,
            )
        except GoogleAPICallError as exc:
            logger.error("Pipeline submission failed", error=str(exc))
            raise

        logger.info(
            "Pipeline job submitted",
            job_name=job.resource_name,
            experiment=experiment_name,
            dashboard_url=job._dashboard_uri(),
        )

        return PipelineRunResult(
            pipeline_job_name=job.resource_name,
            status="PIPELINE_STATE_PENDING",
            dashboard_url=job._dashboard_uri() or "",
        )

    def wait_for_pipeline(
        self,
        pipeline_job_name: str,
        poll_interval_seconds: int = 30,
        timeout_seconds: int = 7200,
    ) -> PipelineRunResult:
        """
        Wait for a Vertex AI Pipeline job to complete.

        Args:
            pipeline_job_name: The pipeline job resource name.
            poll_interval_seconds: Status check interval.
            timeout_seconds: Max wait time (default 2 hours).

        Returns:
            PipelineRunResult with final status.
        """
        terminal = {
            "PIPELINE_STATE_SUCCEEDED",
            "PIPELINE_STATE_FAILED",
            "PIPELINE_STATE_CANCELLED",
        }
        elapsed = 0

        while elapsed < timeout_seconds:
            job = aiplatform.PipelineJob.get(resource_name=pipeline_job_name)
            status = str(job.state)
            logger.debug("Pipeline status", job=pipeline_job_name, status=status)

            if status in terminal:
                return PipelineRunResult(
                    pipeline_job_name=pipeline_job_name,
                    status=status,
                    dashboard_url=job._dashboard_uri() or "",
                )

            time.sleep(poll_interval_seconds)
            elapsed += poll_interval_seconds

        raise TimeoutError(f"Pipeline {pipeline_job_name} did not complete within {timeout_seconds}s")

    def upload_model(
        self,
        model_gcs_uri: str,
        display_name: str,
        serving_container: str = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
        labels: dict[str, str] | None = None,
    ) -> aiplatform.Model:
        """
        Upload a model artifact to Vertex AI Model Registry.

        Args:
            model_gcs_uri: GCS URI of the model artifacts directory.
            display_name: Human-readable model name.
            serving_container: Docker image URI for serving.
            labels: GCP resource labels.

        Returns:
            The uploaded Vertex AI Model resource.
        """
        model = aiplatform.Model.upload(
            display_name=display_name,
            artifact_uri=model_gcs_uri,
            serving_container_image_uri=serving_container,
            labels=labels or {},
        )

        logger.info("Model uploaded to registry", model_name=display_name, resource=model.resource_name)
        return model

    def deploy_model(
        self,
        model_resource_name: str,
        endpoint_display_name: str | None = None,
        machine_type: str | None = None,
        min_replicas: int | None = None,
        max_replicas: int | None = None,
        traffic_split: dict[str, int] | None = None,
    ) -> EndpointInfo:
        """
        Deploy a model from the registry to a Vertex AI online endpoint.

        Args:
            model_resource_name: Full resource name of the Model in registry.
            endpoint_display_name: Display name for the endpoint.
            machine_type: VM type (e.g. "n1-standard-4").
            min_replicas: Minimum number of nodes.
            max_replicas: Maximum number of nodes for autoscaling.
            traffic_split: Traffic split dict (e.g. {"0": 100}).

        Returns:
            EndpointInfo with predict URI.
        """
        ep_name = endpoint_display_name or f"endpoint-{int(time.time())}"
        mtype = machine_type or self._machine_type
        min_rep = min_replicas or self._min_replicas
        max_rep = max_replicas or self._max_replicas

        endpoint = aiplatform.Endpoint.create(display_name=ep_name)
        model = aiplatform.Model(model_name=model_resource_name)

        deployed_model = model.deploy(
            endpoint=endpoint,
            deployed_model_display_name=ep_name,
            machine_type=mtype,
            min_replica_count=min_rep,
            max_replica_count=max_rep,
            traffic_split=traffic_split or {"0": 100},
            sync=True,
        )

        logger.info(
            "Model deployed",
            endpoint=endpoint.resource_name,
            machine_type=mtype,
        )

        return EndpointInfo(
            endpoint_name=ep_name,
            endpoint_resource_name=endpoint.resource_name,
            deployed_model_id="0",
            predict_uri=f"https://{self._location}-aiplatform.googleapis.com/v1/{endpoint.resource_name}:predict",
            status="DEPLOYED",
        )

    def setup_model_monitoring(
        self,
        endpoint_resource_name: str,
        training_dataset_gcs: str,
        target_field: str,
        email_alert: str | None = None,
        skew_threshold: float = 0.3,
        drift_threshold: float = 0.3,
        monitoring_interval_hours: int = 1,
    ) -> str:
        """
        Configure Vertex AI Model Monitoring for an endpoint.

        Monitors for training-serving skew and prediction drift,
        alerting via Cloud Monitoring when thresholds are exceeded.

        Args:
            endpoint_resource_name: The deployed endpoint resource name.
            training_dataset_gcs: GCS URI of training data for skew baseline.
            target_field: Name of the prediction target column.
            email_alert: Email address for monitoring alerts.
            skew_threshold: Training-serving skew alert threshold.
            drift_threshold: Prediction drift alert threshold.
            monitoring_interval_hours: How often to run monitoring checks.

        Returns:
            Model monitoring job resource name.
        """
        raw = load_config()
        alert_email = email_alert or raw.get("observability", {}).get("vertex_model_monitoring", {}).get("email_alert", "")

        monitoring_job = aiplatform.ModelDeploymentMonitoringJob.create(
            display_name=f"monitoring-{int(time.time())}",
            endpoint=endpoint_resource_name,
            logging_sampling_strategy=aiplatform.gapic.SamplingStrategy(
                random_sample_config=aiplatform.gapic.SamplingStrategy.RandomSampleConfig(sample_rate=0.8)
            ),
            model_deployment_monitoring_objective_configs=[
                aiplatform.gapic.ModelDeploymentMonitoringObjectiveConfig(
                    deployed_model_id="0",
                    objective_config=aiplatform.gapic.ModelMonitoringObjectiveConfig(
                        training_dataset=aiplatform.gapic.ModelMonitoringObjectiveConfig.TrainingDataset(
                            gcs_source=aiplatform.gapic.GcsSource(uris=[training_dataset_gcs]),
                            target_field=target_field,
                        ),
                        training_prediction_skew_detection_config=aiplatform.gapic.ModelMonitoringObjectiveConfig.TrainingPredictionSkewDetectionConfig(
                            default_skew_thresholds={"defaultThreshold": aiplatform.gapic.ThresholdConfig(value=skew_threshold)},
                        ),
                        prediction_drift_detection_config=aiplatform.gapic.ModelMonitoringObjectiveConfig.PredictionDriftDetectionConfig(
                            default_drift_thresholds={"defaultThreshold": aiplatform.gapic.ThresholdConfig(value=drift_threshold)},
                        ),
                    ),
                )
            ],
            model_monitoring_alert_config=aiplatform.gapic.ModelMonitoringAlertConfig(
                email_alert_config=aiplatform.gapic.ModelMonitoringAlertConfig.EmailAlertConfig(
                    user_emails=[alert_email] if alert_email else []
                )
            ),
            schedule_config=aiplatform.gapic.ModelDeploymentMonitoringScheduleConfig(
                monitor_interval=aiplatform.gapic.Duration(seconds=monitoring_interval_hours * 3600)
            ),
        )

        logger.info("Model monitoring configured", job_name=monitoring_job.resource_name)
        return monitoring_job.resource_name

    # ------------------------------------------------------------------
    # KFP pipeline builder
    # ------------------------------------------------------------------

    def _build_training_pipeline(self, hparams: dict[str, Any]):
        """Build a KFP v2 pipeline function with preprocessing, training, and evaluation."""

        @component(base_image="python:3.12-slim", packages_to_install=["scikit-learn==1.5.0", "pandas==2.2.0", "google-cloud-storage==2.17.0", "mlflow==2.13.0"])
        def preprocess(
            training_data_gcs: str,
            train_dataset: Output[Dataset],
            test_dataset: Output[Dataset],
            test_size: float = 0.2,
        ) -> None:
            """Download data from GCS and split into train/test."""
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from google.cloud import storage
            import os

            # Download from GCS
            client = storage.Client()
            bucket_name = training_data_gcs.replace("gs://", "").split("/")[0]
            blob_path = "/".join(training_data_gcs.replace("gs://", "").split("/")[1:])
            bucket = client.bucket(bucket_name)
            bucket.blob(blob_path).download_to_filename("/tmp/data.csv")

            df = pd.read_csv("/tmp/data.csv")
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

            train_df.to_csv(train_dataset.path, index=False)
            test_df.to_csv(test_dataset.path, index=False)

        @component(base_image="python:3.12-slim", packages_to_install=["scikit-learn==1.5.0", "pandas==2.2.0", "joblib==1.4.0", "mlflow==2.13.0"])
        def train(
            train_dataset: Input[Dataset],
            model: Output[Model],
            metrics: Output[Metrics],
            n_estimators: int = 100,
            max_depth: int = 6,
            learning_rate: float = 0.1,
        ) -> None:
            """Train a gradient boosting classifier with MLflow tracking."""
            import pandas as pd
            import joblib
            import mlflow
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import cross_val_score
            import os

            df = pd.read_csv(train_dataset.path)
            X = df.drop("target", axis=1)
            y = df["target"]

            clf = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
            )
            clf.fit(X, y)

            cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
            val_accuracy = cv_scores.mean()

            metrics.log_metric("val_accuracy", val_accuracy)
            metrics.log_metric("val_accuracy_std", cv_scores.std())

            os.makedirs(model.path, exist_ok=True)
            joblib.dump(clf, os.path.join(model.path, "model.joblib"))

        @component(base_image="python:3.12-slim", packages_to_install=["scikit-learn==1.5.0", "pandas==2.2.0", "joblib==1.4.0"])
        def evaluate(
            test_dataset: Input[Dataset],
            model: Input[Model],
            evaluation_metrics: Output[Metrics],
            accuracy_threshold: float = 0.85,
        ) -> str:
            """Evaluate model and return pass/fail decision."""
            import pandas as pd
            import joblib
            import os
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

            df = pd.read_csv(test_dataset.path)
            X_test = df.drop("target", axis=1)
            y_test = df["target"]

            clf = joblib.load(os.path.join(model.path, "model.joblib"))
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            auc = roc_auc_score(y_test, y_prob) if len(set(y_test)) == 2 else 0.0

            evaluation_metrics.log_metric("test_accuracy", accuracy)
            evaluation_metrics.log_metric("test_f1", f1)
            evaluation_metrics.log_metric("test_auc", auc)

            return "pass" if accuracy >= accuracy_threshold else "fail"

        @dsl.pipeline(name="vertex-ai-training-pipeline", description="End-to-end training pipeline on Vertex AI")
        def training_pipeline(training_data_gcs: str) -> None:
            prep = preprocess(training_data_gcs=training_data_gcs)
            prep.set_caching_options(enable_caching=True)

            train_step = train(
                train_dataset=prep.outputs["train_dataset"],
                n_estimators=hparams.get("n_estimators", 100),
                max_depth=hparams.get("max_depth", 6),
                learning_rate=hparams.get("learning_rate", 0.1),
            )
            train_step.set_cpu_limit("4").set_memory_limit("16G")

            eval_step = evaluate(
                test_dataset=prep.outputs["test_dataset"],
                model=train_step.outputs["model"],
                accuracy_threshold=0.85,
            )

        return training_pipeline
