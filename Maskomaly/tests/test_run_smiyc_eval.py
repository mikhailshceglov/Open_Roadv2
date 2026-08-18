import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_smiyc_eval as smiyc


class FakeFrame(SimpleNamespace):
    pass


class FakeModel:
    def __init__(self):
        self.calls = 0

    def get_soft_mask(self, bgr_image):
        self.calls += 1
        np.testing.assert_array_equal(bgr_image[0, 0], [30, 20, 10])
        return np.full(bgr_image.shape[:2], 1.2, dtype=np.float32)


def make_args(root, output, phase="all"):
    config = root / "config.yaml"
    weights = root / "weights.pkl"
    config.touch()
    weights.touch()
    return argparse.Namespace(
        datasets_root=root,
        output=output,
        models=["maskomaly"],
        phase=phase,
        config_file=config,
        weights=weights,
        visualize=False,
        masks=4,
        analysis_file=None,
    )


def make_dataset_layout(root):
    for spec in smiyc.DATASETS.values():
        dataset = root / spec["directory"]
        (dataset / "images").mkdir(parents=True)
        (dataset / "labels_masks").mkdir()


class TestSMIYCEval(unittest.TestCase):
    def test_rgb_to_bgr_is_contiguous_and_reverses_channels(self):
        rgb = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        bgr = smiyc.rgb_to_bgr(rgb)
        np.testing.assert_array_equal(bgr, [[[3, 2, 1], [6, 5, 4]]])
        self.assertTrue(bgr.flags.c_contiguous)

    def test_prepare_anomaly_map_resizes_clips_and_rejects_nan(self):
        prediction = np.array([[-1.0, 2.0]], dtype=np.float32)

        def resize(value, target_hw):
            self.assertEqual(target_hw, (2, 2))
            return np.repeat(value, 2, axis=0)

        result = smiyc.prepare_anomaly_map(prediction, (2, 2), resize_fn=resize)
        np.testing.assert_array_equal(result, [[0.0, 1.0], [0.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            smiyc.prepare_anomaly_map(np.array([[np.nan]]), (1, 1))

    def test_non_finite_official_metric_becomes_json_null(self):
        self.assertIsNone(smiyc._as_percent(np.nan))

    def test_preflight_reports_missing_dataset_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = make_args(root, root / "out")
            with self.assertRaisesRegex(FileNotFoundError, "dataset_AnomalyTrack"):
                smiyc.preflight(args)

    def test_full_fake_run_saves_every_frame_and_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "results"
            make_dataset_layout(root)
            args = make_args(root, output)
            model = FakeModel()

            class FakeEvaluation:
                def __init__(self, method_name, dataset_name, threaded_saver=False):
                    self.method_name = method_name
                    self.dataset_name = dataset_name
                    self.frames = [
                        FakeFrame(
                            image=np.full((2, 3, 3), [10, 20, 30], dtype=np.uint8),
                            dset_name=dataset_name,
                            fid="frame_{:02d}".format(index),
                        )
                        for index in range(2)
                    ]

                def __len__(self):
                    return len(self.frames)

                def get_frames(self):
                    return iter(self.frames)

                def save_output(self, frame, anomaly_map):
                    self.assert_map(anomaly_map)
                    path = (
                        output
                        / "anomaly_p"
                        / self.method_name
                        / frame.dset_name
                        / "{}.hdf5".format(frame.fid)
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()

                @staticmethod
                def assert_map(anomaly_map):
                    self.assertEqual(anomaly_map.shape, (2, 3))
                    self.assertEqual(float(anomaly_map.min()), 1.0)
                    self.assertEqual(float(anomaly_map.max()), 1.0)

                def wait_to_finish_saving(self):
                    pass

                def calculate_metric_from_saved_outputs(self, metric, **kwargs):
                    if metric == "PixBinaryClass":
                        return SimpleNamespace(area_PRC=0.91, tpr95_fpr=0.08)
                    return SimpleNamespace(sIoU_gt=0.71, prec_pred=0.61, f1_mean=0.66)

            # Bind TestCase assertions for the nested fake class.
            FakeEvaluation.assert_map = lambda _, anomaly_map: (
                self.assertEqual(anomaly_map.shape, (2, 3)),
                self.assertEqual(float(anomaly_map.min()), 1.0),
                self.assertEqual(float(anomaly_map.max()), 1.0),
            )

            rows = smiyc.execute(
                args,
                evaluation_class=FakeEvaluation,
                model_loader=lambda name, parsed: model,
            )
            self.assertEqual(model.calls, 4)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["AUPR"], 91.0)

            with (output / "summary.csv").open(newline="", encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 2)
            payload = json.loads((output / "summary.json").read_text())
            self.assertEqual(payload["unit"], "percent")
            self.assertEqual(len(payload["results"]), 2)

    def test_metrics_phase_never_loads_a_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "results"
            make_dataset_layout(root)
            args = make_args(root, output, phase="metrics")

            class MetricsEvaluation:
                def __init__(self, method_name, dataset_name, threaded_saver=False):
                    self.method_name = method_name
                    self.dataset_name = dataset_name
                    self.frames = [
                        FakeFrame(
                            image=np.zeros((1, 1, 3), dtype=np.uint8),
                            dset_name=dataset_name,
                            fid="one",
                        )
                    ]
                    path = output / "anomaly_p" / method_name / dataset_name / "one.hdf5"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()

                def get_frames(self):
                    return iter(self.frames)

                def calculate_metric_from_saved_outputs(self, metric, **kwargs):
                    if metric == "PixBinaryClass":
                        return SimpleNamespace(area_PRC=0.5, tpr95_fpr=0.4)
                    return SimpleNamespace(sIoU_gt=0.3, prec_pred=0.2, f1_mean=0.1)

            def forbidden_loader(*_):
                raise AssertionError("metrics phase loaded a model")

            rows = smiyc.execute(
                args,
                evaluation_class=MetricsEvaluation,
                model_loader=forbidden_loader,
            )
            self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
