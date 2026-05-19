"""
Vertex AI Evaluation Service — automated LLM and RAG quality assessment.

Provides:
- Pointwise evaluation (single response scoring)
- Pairwise evaluation (A/B response comparison)
- Vertex AI Autorater (Gemini-powered judge)
- RAG-specific metrics: groundedness, fulfillment, summarisation quality
- Batch evaluation over datasets
- Cloud Monitoring metric publishing
- BigQuery result export
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import vertexai
from google.cloud import aiplatform, bigquery, monitoring_v3
from vertexai.evaluation import EvalTask, MetricPromptTemplateExamples, PointwiseMetric, PairwiseMetric
from vertexai.generative_models import GenerativeModel

from src.utils.config import get_config, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PointwiseResult:
    """Result of a pointwise (single response) evaluation."""

    question: str
    response: str
    reference: str = ""

    groundedness: float = 0.0
    fulfillment: float = 0.0
    summarization_quality: float = 0.0
    question_answering_quality: float = 0.0
    coherence: float = 0.0
    fluency: float = 0.0
    safety: float = 0.0

    explanations: dict[str, str] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        scores = [s for s in [
            self.groundedness, self.fulfillment,
            self.question_answering_quality, self.coherence, self.fluency
        ] if s > 0]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def passed(self) -> bool:
        return self.overall_score >= 3.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "groundedness": self.groundedness,
            "fulfillment": self.fulfillment,
            "question_answering_quality": self.question_answering_quality,
            "coherence": self.coherence,
            "fluency": self.fluency,
            "safety": self.safety,
            "overall_score": self.overall_score,
            "passed": self.passed,
        }


@dataclass
class PairwiseResult:
    """Result of a pairwise (A/B comparison) evaluation."""

    question: str
    response_a: str
    response_b: str
    winner: str = ""          # "A" | "B" | "SAME"
    preference_score_a: float = 0.0
    preference_score_b: float = 0.0
    explanation: str = ""


@dataclass
class BatchEvalSummary:
    """Summary of a batch evaluation run."""

    total: int
    passed: int
    avg_groundedness: float = 0.0
    avg_fulfillment: float = 0.0
    avg_qa_quality: float = 0.0
    avg_overall: float = 0.0
    results: list[PointwiseResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


class VertexAIEvaluator:
    """
    Production Vertex AI Evaluation Service client.

    Uses Gemini as the autorater for LLM-as-judge evaluation,
    aligned with Vertex AI's built-in metric definitions.

    Example:
        evaluator = VertexAIEvaluator()

        # Evaluate a single RAG response
        result = evaluator.evaluate_pointwise(
            question="What is the incident response SLA?",
            response=rag_answer,
            context=retrieved_context,
            reference="The SLA is 4 hours for P1 incidents.",
        )
        print(f"Groundedness: {result.groundedness}/5")
        print(f"Overall: {result.overall_score:.1f}/5")

        # Compare two model responses
        comparison = evaluator.evaluate_pairwise(
            question="Explain APRA CPS 234",
            response_a=gemini_response,
            response_b=gpt4_response,
        )
        print(f"Winner: {comparison.winner}")
    """

    _GROUNDEDNESS_PROMPT = """You are an expert AI evaluator assessing whether a response is grounded in the provided context.

CONTEXT:
{context}

QUESTION: {question}

RESPONSE: {response}

Score GROUNDEDNESS (1-5):
1 = Response contradicts or fabricates beyond context
2 = Mostly unsupported claims
3 = Partially supported
4 = Mostly supported with minor gaps
5 = Every claim directly supported by context

Respond with valid JSON only:
{{"score": <int 1-5>, "explanation": "<brief reasoning>"}}"""

    _FULFILLMENT_PROMPT = """You are an expert AI evaluator assessing whether a response fulfills the user's request.

QUESTION: {question}
RESPONSE: {response}

Score FULFILLMENT (1-5):
1 = Does not address the question
2 = Tangentially related
3 = Partially fulfills
4 = Mostly fulfills
5 = Completely and directly fulfills the request

Respond with valid JSON only:
{{"score": <int 1-5>, "explanation": "<brief reasoning>"}}"""

    _QA_QUALITY_PROMPT = """You are an expert AI evaluator assessing the quality of a question-answering response.

QUESTION: {question}
RESPONSE: {response}
REFERENCE ANSWER: {reference}

Score QUESTION_ANSWERING_QUALITY (1-5):
1 = Incorrect or completely misses the answer
2 = Partially correct with major errors
3 = Mostly correct with some gaps
4 = Correct with minor issues
5 = Perfectly accurate and complete

Respond with valid JSON only:
{{"score": <int 1-5>, "explanation": "<brief reasoning>"}}"""

    _PAIRWISE_PROMPT = """You are an expert AI evaluator comparing two responses to the same question.

QUESTION: {question}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

Which response is better overall (accuracy, completeness, clarity)?
Respond with valid JSON only:
{{"winner": "<A|B|SAME>", "score_a": <float 1-5>, "score_b": <float 1-5>, "explanation": "<brief reasoning>"}}"""

    def __init__(self) -> None:
        cfg = get_config()
        raw = load_config()
        eval_cfg = raw.get("vertex_evaluation", {})

        self._project_id = cfg.project_id
        self._location = cfg.vertex_ai.location
        self._autorater_model = eval_cfg.get("autorater_model", "gemini-2.0-flash-001")

        vertexai.init(project=self._project_id, location=self._location)

        self._judge = GenerativeModel(model_name=self._autorater_model)

        logger.info(
            "VertexAIEvaluator initialised",
            project=self._project_id,
            autorater=self._autorater_model,
        )

    def evaluate_pointwise(
        self,
        question: str,
        response: str,
        context: str = "",
        reference: str = "",
        metrics: list[str] | None = None,
    ) -> PointwiseResult:
        """
        Evaluate a single response across multiple quality metrics.

        Args:
            question: The original user question.
            response: The model/RAG response to evaluate.
            context: Retrieved context (for groundedness evaluation).
            reference: Reference answer (for QA quality evaluation).
            metrics: Subset of metrics to evaluate. Default: all.

        Returns:
            PointwiseResult with per-metric scores (1-5 scale).
        """
        requested = set(metrics or ["groundedness", "fulfillment", "qa_quality"])
        result = PointwiseResult(question=question, response=response, reference=reference)

        if "groundedness" in requested and context:
            score, expl = self._score(
                self._GROUNDEDNESS_PROMPT.format(context=context, question=question, response=response)
            )
            result.groundedness = score
            result.explanations["groundedness"] = expl

        if "fulfillment" in requested:
            score, expl = self._score(
                self._FULFILLMENT_PROMPT.format(question=question, response=response)
            )
            result.fulfillment = score
            result.explanations["fulfillment"] = expl

        if "qa_quality" in requested and reference:
            score, expl = self._score(
                self._QA_QUALITY_PROMPT.format(
                    question=question, response=response, reference=reference
                )
            )
            result.question_answering_quality = score
            result.explanations["qa_quality"] = expl

        logger.info(
            "Pointwise evaluation complete",
            overall=f"{result.overall_score:.2f}",
            groundedness=f"{result.groundedness:.1f}",
            fulfillment=f"{result.fulfillment:.1f}",
            passed=result.passed,
        )

        return result

    def evaluate_pairwise(
        self,
        question: str,
        response_a: str,
        response_b: str,
    ) -> PairwiseResult:
        """
        Compare two responses and determine which is better.

        Useful for A/B testing between model versions or comparing
        different RAG configurations.

        Args:
            question: The original user question.
            response_a: First response (model A / baseline).
            response_b: Second response (model B / candidate).

        Returns:
            PairwiseResult with winner and preference scores.
        """
        _, raw = self._score_raw(
            self._PAIRWISE_PROMPT.format(
                question=question, response_a=response_a, response_b=response_b
            )
        )

        winner = raw.get("winner", "SAME")
        score_a = float(raw.get("score_a", 0))
        score_b = float(raw.get("score_b", 0))
        explanation = raw.get("explanation", "")

        result = PairwiseResult(
            question=question,
            response_a=response_a,
            response_b=response_b,
            winner=winner,
            preference_score_a=score_a,
            preference_score_b=score_b,
            explanation=explanation,
        )

        logger.info(
            "Pairwise evaluation complete",
            winner=winner,
            score_a=score_a,
            score_b=score_b,
        )

        return result

    def evaluate_batch(
        self,
        test_cases: list[dict[str, Any]],
        publish_to_monitoring: bool = False,
        export_to_bigquery: bool = False,
        bq_table: str | None = None,
    ) -> BatchEvalSummary:
        """
        Evaluate a batch of RAG responses.

        Args:
            test_cases: List of dicts with keys: question, response, context, reference.
            publish_to_monitoring: Push aggregate metrics to Cloud Monitoring.
            export_to_bigquery: Write individual results to BigQuery.
            bq_table: BigQuery table spec (project.dataset.table).

        Returns:
            BatchEvalSummary with aggregate statistics.
        """
        results: list[PointwiseResult] = []

        for i, case in enumerate(test_cases):
            logger.info("Evaluating test case", index=i, total=len(test_cases))
            result = self.evaluate_pointwise(**case)
            results.append(result)

        n = len(results)
        passed = sum(1 for r in results if r.passed)

        summary = BatchEvalSummary(
            total=n,
            passed=passed,
            avg_groundedness=sum(r.groundedness for r in results) / n if n else 0,
            avg_fulfillment=sum(r.fulfillment for r in results) / n if n else 0,
            avg_qa_quality=sum(r.question_answering_quality for r in results) / n if n else 0,
            avg_overall=sum(r.overall_score for r in results) / n if n else 0,
            results=results,
        )

        logger.info(
            "Batch evaluation complete",
            total=n,
            passed=passed,
            pass_rate=f"{summary.pass_rate:.1%}",
            avg_overall=f"{summary.avg_overall:.2f}",
        )

        if publish_to_monitoring:
            self._publish_metrics(summary)

        if export_to_bigquery and bq_table:
            self._export_to_bq(results, bq_table)

        return summary

    def run_vertex_eval_task(
        self,
        dataset: list[dict[str, Any]],
        experiment_name: str,
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run a native Vertex AI EvalTask using the SDK's built-in eval framework.

        This uses Vertex AI's managed autorater infrastructure rather than
        the custom LLM-as-judge approach above.

        Args:
            dataset: List of dicts with prompt, response, reference fields.
            experiment_name: Vertex AI Experiment name for tracking.
            metrics: Vertex AI built-in metric names.

        Returns:
            Evaluation summary dict from Vertex AI.
        """
        import pandas as pd

        df = pd.DataFrame(dataset)

        requested_metrics = metrics or [
            MetricPromptTemplateExamples.Pointwise.GROUNDEDNESS,
            MetricPromptTemplateExamples.Pointwise.FULFILLMENT,
            MetricPromptTemplateExamples.Pointwise.QUESTION_ANSWERING_QUALITY,
            MetricPromptTemplateExamples.Pointwise.COHERENCE,
            MetricPromptTemplateExamples.Pointwise.FLUENCY,
        ]

        eval_task = EvalTask(
            dataset=df,
            metrics=requested_metrics,
            experiment=experiment_name,
        )

        eval_result = eval_task.evaluate(
            model=self._autorater_model,
        )

        logger.info("Vertex AI EvalTask complete", experiment=experiment_name)
        return eval_result.summary_metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score(self, prompt: str) -> tuple[float, str]:
        """Call the LLM judge and return (score, explanation)."""
        _, raw = self._score_raw(prompt)
        return float(raw.get("score", 0)), raw.get("explanation", "")

    def _score_raw(self, prompt: str) -> tuple[float, dict[str, Any]]:
        """Call the judge and return parsed dict."""
        try:
            response = self._judge.generate_content(
                prompt,
                generation_config={"temperature": 0.0, "max_output_tokens": 512, "response_mime_type": "application/json"},
            )
            raw = json.loads(response.text)
            return float(raw.get("score", 0)), raw
        except Exception as exc:
            logger.error("Evaluation scoring failed", error=str(exc))
            return 0.0, {"score": 0, "explanation": f"Error: {exc}"}

    def _publish_metrics(self, summary: BatchEvalSummary) -> None:
        """Publish aggregate evaluation metrics to Google Cloud Monitoring."""
        raw = load_config()
        project_id = raw.get("gcp", {}).get("project_id", "")
        prefix = raw.get("observability", {}).get("cloud_monitoring", {}).get("metrics_prefix", "custom.googleapis.com/ai_platform")

        client = monitoring_v3.MetricServiceClient()
        project_name = f"projects/{project_id}"
        now = time.time()

        series = []
        for name, value in [
            (f"{prefix}/rag/groundedness", summary.avg_groundedness),
            (f"{prefix}/rag/fulfillment", summary.avg_fulfillment),
            (f"{prefix}/rag/qa_quality", summary.avg_qa_quality),
            (f"{prefix}/rag/overall", summary.avg_overall),
            (f"{prefix}/rag/pass_rate", summary.pass_rate),
        ]:
            ts = monitoring_v3.TimeSeries()
            ts.metric.type = name
            ts.resource.type = "global"
            point = monitoring_v3.Point()
            point.value.double_value = value
            point.interval.end_time.seconds = int(now)
            ts.points.append(point)
            series.append(ts)

        try:
            client.create_time_series(name=project_name, time_series=series)
            logger.info("Evaluation metrics published to Cloud Monitoring", count=len(series))
        except Exception as exc:
            logger.warning("Failed to publish metrics", error=str(exc))

    def _export_to_bq(self, results: list[PointwiseResult], bq_table: str) -> None:
        """Export evaluation results to BigQuery."""
        client = bigquery.Client()
        rows = [r.to_dict() for r in results]
        errors = client.insert_rows_json(bq_table, rows)
        if errors:
            logger.warning("BigQuery export had errors", errors=errors)
        else:
            logger.info("Results exported to BigQuery", table=bq_table, rows=len(rows))
