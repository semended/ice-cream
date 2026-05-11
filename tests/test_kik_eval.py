from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from vlm_eval.config import ModelConfig
from vlm_eval.providers import MockProvider, OpenAICompatibleProvider
from vlm_eval.run import find_reference_images, get_task_spec, run_one
from vlm_eval.tasks.kik.data import (
    generate_kik_jsonl_from_csv,
    load_kik_cases,
    load_kik_labels,
    resolve_kik_labels_path,
)
from vlm_eval.tasks.kik.schema import KIK_REQUIRED_FIELDS, make_mock_kik_prediction, validate_kik_prediction
from vlm_eval.tasks.kik.scoring import (
    aggregate_kik_by_model,
    boolean_metrics,
    business_scores,
    kik_hallucination_metrics,
    score_kik_fields,
    score_kik_value,
    status_metrics,
)


class KikEvalTests(unittest.TestCase):
    def test_loading_manual_ground_truth_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manual_ground_truth.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "image_id": "photo_001.jpg",
                        "kik_present": True,
                        "has_kik_grouped_block": True,
                        "kik_sku_count": 3,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            labels = load_kik_labels(path)
            self.assertTrue(labels["photo_001.jpg"]["kik_present"])
            self.assertTrue(labels["photo_001.jpg"]["has_monobrand_block"])
            self.assertEqual(labels["photo_001.jpg"]["kik_sku_count"], 3)

    def test_loading_or_generating_from_kik_report_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_dir = root / "data" / "ground_truth"
            gt_dir.mkdir(parents=True)
            csv_path = gt_dir / "kik_report_ground_truth.csv"
            csv_path.write_text(
                "image_id,is_trade_equipment_photo,is_ice_cream_equipment,equipment_type,"
                "photo_crop_quality,kik_present,kik_sku_count,"
                "kik_share_percent,has_kik_grouped_block,has_poleno,has_briquette,status\n"
                "a.jpg,true,true,display_freezer,full,true,4,30,true,false,true,attention\n",
                encoding="utf-8",
            )
            labels_path, mode = resolve_kik_labels_path(None, project_root=root)
            self.assertEqual(mode, "generated_from_kik_report_ground_truth_csv")
            self.assertTrue(labels_path.exists())
            labels = load_kik_labels(labels_path)
            self.assertEqual(set(labels["a.jpg"]), set(KIK_REQUIRED_FIELDS))
            self.assertTrue(labels["a.jpg"]["photo_crop_is_full"])
            self.assertTrue(labels["a.jpg"]["has_poleno_or_briquette"])
            self.assertEqual(labels["a.jpg"]["status_score"], 1)

            explicit_output = root / "manual_copy.jsonl"
            generate_kik_jsonl_from_csv(csv_path, explicit_output)
            self.assertTrue(explicit_output.exists())

    def test_kik_schema_validation(self) -> None:
        prediction = make_mock_kik_prediction()
        result = validate_kik_prediction(prediction)
        self.assertTrue(result.ok, result.errors)

        broken = dict(prediction)
        broken["kik_sku_count"] = 31
        result = validate_kik_prediction(broken)
        self.assertFalse(result.ok)

    def test_boolean_scoring(self) -> None:
        expected = {"kik_present": True, "has_posm": False}
        predicted = {**make_mock_kik_prediction(), "kik_present": False, "has_posm": False}
        scores = score_kik_fields(expected, predicted)
        self.assertEqual(scores["kik_present"], 0.0)
        self.assertEqual(scores["has_posm"], 1.0)

    def test_boolean_f1_counts_complete_miss_as_zero(self) -> None:
        rows = [
            _result_row("a.jpg", {"has_eskimo": True}, {"has_eskimo": False}),
            _result_row("b.jpg", {"has_eskimo": True}, {"has_eskimo": False}),
        ]
        metrics = boolean_metrics(rows, "has_eskimo")
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)

    def test_numeric_scoring(self) -> None:
        self.assertEqual(score_kik_value("kik_sku_count", 5, 7), 0.8)
        self.assertEqual(score_kik_value("kik_share_percent", 50, 60), 0.8)
        self.assertEqual(score_kik_value("status_score", 1, 1), 1.0)
        self.assertEqual(score_kik_value("status_score", 1, 2), 0.0)

    def test_kik_business_score_weighted_calculation(self) -> None:
        expected = make_mock_kik_prediction()
        scores = business_scores(expected, dict(expected))
        self.assertEqual(scores["kik_business_score_pct"], 100.0)
        self.assertEqual(scores["core_kik_score_pct"], 100.0)

    def test_positive_gt_zero_predicted_sku_count_zeroes_photo_score(self) -> None:
        expected = {**make_mock_kik_prediction(), "kik_sku_count": 3}
        predicted = dict(expected)
        predicted["kik_sku_count"] = 0

        scores = business_scores(expected, predicted)

        self.assertEqual(scores["kik_business_score_pct"], 0.0)
        self.assertEqual(scores["core_kik_score_pct"], 0.0)
        self.assertEqual(scores["sku_family_score_pct"], 0.0)
        self.assertEqual(scores["execution_score_pct"], 0.0)
        self.assertEqual(scores["equipment_photo_score_pct"], 0.0)
        self.assertEqual(scores["status_actionability_score_pct"], 0.0)

    def test_sku_zero_gate_affects_aggregate_business_score(self) -> None:
        expected = {**make_mock_kik_prediction(), "kik_sku_count": 3}
        exact = _result_row("exact.jpg", expected, dict(expected))
        missed = _result_row("missed.jpg", expected, {**expected, "kik_sku_count": 0})

        aggregate = aggregate_kik_by_model([exact, missed])
        summary = aggregate["summaries"][0]

        self.assertEqual(summary["kik_business_score_pct"], 50.0)

    def test_status_critical_recall(self) -> None:
        rows = [
            _result_row("a.jpg", {"status_score": 2}, {"status_score": 2}),
            _result_row("b.jpg", {"status_score": 2}, {"status_score": 0}),
            _result_row("c.jpg", {"status_score": 1}, {"status_score": 2}),
        ]
        metrics = status_metrics(rows)
        self.assertEqual(metrics["critical_recall"], 0.5)
        self.assertEqual(metrics["critical_precision"], 0.5)
        self.assertEqual(metrics["false_normal_on_critical_count"], 1)

    def test_kik_hallucination_metrics(self) -> None:
        rows = [
            _result_row(
                "absent.jpg",
                {"kik_present": False, "kik_sku_count": 0, "kik_share_percent": 0},
                {"kik_present": True, "kik_sku_count": 2, "kik_share_percent": 20},
            ),
            _result_row(
                "present.jpg",
                {"kik_present": True, "kik_sku_count": 2, "kik_share_percent": 20},
                {"kik_present": False, "kik_sku_count": 0, "kik_share_percent": 0},
            ),
        ]
        metrics = kik_hallucination_metrics(rows)
        self.assertEqual(metrics["kik_false_positive_rate"], 1.0)
        self.assertEqual(metrics["kik_false_negative_rate"], 1.0)
        self.assertEqual(metrics["sku_hallucination_on_absent_kik"], 1.0)
        self.assertEqual(metrics["share_hallucination_on_absent_kik"], 1.0)

    def test_mock_provider_output_for_kik(self) -> None:
        provider = MockProvider()
        response = provider.complete(
            _mock_model(),
            "data:image/jpeg;base64,abc",
            "json_schema",
            1,
            mock_prediction=make_mock_kik_prediction(),
        )
        parsed = json.loads(response.raw_response)
        self.assertTrue(validate_kik_prediction(parsed).ok)

    def test_reference_images_are_sent_before_target(self) -> None:
        provider = OpenAICompatibleProvider("openrouter", "https://example.test", "token")
        payload = provider._make_payload(
            _mock_model(),
            "data:image/jpeg;base64,target",
            "json_object",
            user_prompt="Analyze target.",
            schema_instruction="Return JSON only.",
            reference_images=[
                ("cups reference", "data:image/jpeg;base64,ref1"),
                ("cone reference", "data:image/jpeg;base64,ref2"),
            ],
        )
        content = payload["messages"][1]["content"]
        image_urls = [item["image_url"]["url"] for item in content if item.get("type") == "image_url"]
        text_blocks = [item["text"] for item in content if item.get("type") == "text"]

        self.assertEqual(
            image_urls,
            [
                "data:image/jpeg;base64,ref1",
                "data:image/jpeg;base64,ref2",
                "data:image/jpeg;base64,target",
            ],
        )
        self.assertTrue(any("REFERENCE IMAGE 1" in text for text in text_blocks))
        self.assertTrue(any("TARGET IMAGE TO ANALYZE" in text for text in text_blocks))

    def test_reference_image_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (20, 20), color=(255, 255, 255)).save(root / "ref_cups.jpg")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")

            paths = find_reference_images(root)

            self.assertEqual([path.name for path in paths], ["ref_cups.jpg"])

    def test_reference_image_discovery_uses_canonical_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["ref_sandwich.jpg", "ref_bucket.jpg", "ref_briquette.jpg", "ref_cone.jpg"]:
                Image.new("RGB", (20, 20), color=(255, 255, 255)).save(root / name)

            paths = find_reference_images(root)

            self.assertEqual(
                [path.name for path in paths],
                ["ref_briquette.jpg", "ref_bucket.jpg", "ref_cone.jpg", "ref_sandwich.jpg"],
            )

    def test_summary_aggregation(self) -> None:
        expected = make_mock_kik_prediction()
        row = _result_row("photo.jpg", expected, dict(expected))
        row.update({"field_scores": score_kik_fields(expected, row["parsed"])})
        aggregate = aggregate_kik_by_model([row])
        summary = aggregate["summaries"][0]
        self.assertEqual(summary["model_key"], "mock")
        self.assertEqual(summary["schema_valid_rate"], 1.0)
        self.assertEqual(summary["kik_business_score_pct"], 100.0)
        self.assertEqual(summary["field_coverage_rate"], 1.0)

    def test_kik_run_one_with_mock_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "photo.jpg"
            labels_path = root / "manual_ground_truth.jsonl"
            Image.new("RGB", (20, 20), color=(255, 255, 255)).save(image_path)
            labels_path.write_text(
                json.dumps({"image_id": image_path.name, **make_mock_kik_prediction()}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            case = load_kik_cases(root, labels_path)[0]

            import vlm_eval.run as run_module

            original = run_module.create_provider
            run_module.create_provider = lambda provider_name: MockProvider()
            try:
                result = run_one(case, _mock_model(), timeout_seconds=1, task_spec=get_task_spec("kik"))
            finally:
                run_module.create_provider = original

            self.assertTrue(result["json_parse_ok"])
            self.assertTrue(result["schema_valid"])
            self.assertEqual(result["task"], "kik")


def _result_row(image: str, expected: dict[str, object], parsed: dict[str, object]) -> dict[str, object]:
    return {
        "image": image,
        "task": "kik",
        "model_key": "mock",
        "model": "mock",
        "role": "mock",
        "provider": "mock",
        "provider_model": "mock",
        "latency_ms": 100,
        "json_parse_ok": True,
        "schema_valid": True,
        "retry_count": 0,
        "api_retry_count": 0,
        "token_usage": None,
        "raw_response": json.dumps(parsed, ensure_ascii=False),
        "parsed": parsed,
        "expected": expected,
        "field_scores": score_kik_fields(expected, parsed),
        "error": None,
        "response_format_mode": "json_schema",
    }


def _mock_model() -> ModelConfig:
    return ModelConfig(
        key="mock",
        role="mock",
        canonical_model="mock",
        provider="mock",
        provider_model="mock",
        enabled_by_default=False,
        heavy=False,
        temperature=0,
        max_output_tokens=256,
        image_max_side=1024,
        response_format="json_schema",
    )


if __name__ == "__main__":
    unittest.main()
