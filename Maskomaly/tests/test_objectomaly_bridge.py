from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np


MASKOMALY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MASKOMALY_DIR / "maskomaly"))
sys.path.insert(0, str(MASKOMALY_DIR / "scripts"))

import export_objectomaly_inputs as exporter
import import_objectomaly_outputs as importer
import run_objectomaly_infer as folder_inference
from global_fusion import apply_global_fusion
from objectomaly_cache import (
    index_entries,
    read_manifest,
    resolve_cached_path,
    validate_anomaly_map,
    validate_frame_id,
    validate_semantic_map,
    write_manifest,
)
import run_objectomaly_refinement as refinement


class FakeBundle:
    n = 1


class FakeGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, image):
        self.calls += 1
        np.testing.assert_array_equal(image[0, 0], [30, 20, 10])
        return FakeBundle()


class TestObjectomalyCache(unittest.TestCase):
    def test_rejects_unsafe_ids_and_paths(self):
        with self.assertRaises(ValueError):
            validate_frame_id("../frame")
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "manifest.json"
            write_manifest(manifest, {"entries": []})
            with self.assertRaises(ValueError):
                resolve_cached_path(manifest, "../outside.npy")

    def test_manifest_round_trip_and_duplicate_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            entry = {"dataset": "d", "fid": "one"}
            write_manifest(path, {"kind": "test", "entries": [entry]})
            self.assertEqual(read_manifest(path)["kind"], "test")
            with self.assertRaises(ValueError):
                index_entries([entry, entry])

    def test_anomaly_map_contract(self):
        result = validate_anomaly_map(np.zeros((2, 3)), (2, 3))
        self.assertEqual(result.dtype, np.float32)
        with self.assertRaises(ValueError):
            validate_anomaly_map(np.full((2, 3), 1.1), (2, 3))

    def test_semantic_map_contract(self):
        result = validate_semantic_map(np.zeros((2, 3), np.int64), (2, 3))
        self.assertEqual(result.dtype, np.int16)
        with self.assertRaises(ValueError):
            validate_semantic_map(np.full((2, 3), 1.5), (2, 3))


class TestObjectomalyBridge(unittest.TestCase):
    def test_sam_cpu_nms_workaround_normalizes_devices(self):
        class Tensor:
            def __init__(self, device):
                self.device = device

            def detach(self):
                return self

            def cpu(self):
                return Tensor("cpu")

            def to(self, device):
                return Tensor(device)

        calls = []

        def batched_nms(boxes, scores, idxs, threshold):
            calls.append((boxes.device, scores.device, idxs.device, threshold))
            return Tensor("cpu")

        automatic = SimpleNamespace(batched_nms=batched_nms)
        package = SimpleNamespace(automatic_mask_generator=automatic)
        with mock.patch.dict(sys.modules, {"segment_anything": package}):
            refinement.install_sam_cpu_nms_workaround()
            keep = automatic.batched_nms(
                Tensor("cuda"), Tensor("cuda"), Tensor("cuda"), 0.7
            )
        self.assertEqual(calls, [("cpu", "cpu", "cpu", 0.7)])
        self.assertEqual(keep.device, "cuda")

    def test_refinement_accepts_custom_folder_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_rel = Path("images/custom-folder/000000_one.npy")
            coarse_rel = Path("coarse/maskomaly/custom-folder/000000_one.npy")
            (root / image_rel).parent.mkdir(parents=True)
            (root / coarse_rel).parent.mkdir(parents=True)
            np.save(
                str(root / image_rel),
                np.full((2, 3, 3), [30, 20, 10], np.uint8),
            )
            np.save(str(root / coarse_rel), np.full((2, 3), 0.4, np.float32))
            source_manifest = root / "manifest-maskomaly.json"
            write_manifest(
                source_manifest,
                {
                    "kind": "raas-objectomaly-folder-inputs",
                    "source_model": "maskomaly",
                    "entries": [
                        {
                            "dataset": "custom-folder",
                            "fid": "000000_one",
                            "height": 2,
                            "width": 3,
                            "image_bgr": str(image_rel),
                            "coarse_map": str(coarse_rel),
                        }
                    ],
                },
            )
            config = root / "config.json"
            config.write_text(
                '{"sam":{"variant":"vit_h"},"postprocess":{},'
                '"oasc":{"variant":"test","params":{}},'
                '"mbp":{"variant":"test","params":{}}}',
                encoding="utf-8",
            )
            dependencies = {
                "has_bundle": lambda *args, **kwargs: False,
                "postprocess": lambda bundle, **kwargs: bundle,
                "save_bundle": lambda *args, **kwargs: None,
                "load_bundle": lambda *args, **kwargs: FakeBundle(),
                "apply_oasc": lambda coarse, *args, **kwargs: coarse,
                "apply_mbp": lambda coarse, *args, **kwargs: coarse,
            }
            args = SimpleNamespace(
                manifest=source_manifest,
                output=root / "refined-output",
                config=config,
                sam_checkpoint=None,
                device="cpu",
                phase="all",
            )
            with mock.patch.object(refinement, "verify_objectomaly_commit"):
                result = refinement.execute(
                    args, dependencies=dependencies, generator=FakeGenerator()
                )
            payload = read_manifest(result)
            self.assertEqual(payload["kind"], "raas-objectomaly-folder-refined")
            self.assertEqual(payload["input_manifest"], str(source_manifest.resolve()))

    def test_custom_folder_export_preserves_float_map_and_source_name(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is not installed")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "input"
            input_dir.mkdir()
            cv2.imwrite(str(input_dir / "my example.jpg"), np.full((4, 5, 3), 60, np.uint8))
            config = root / "config.yaml"
            weights = root / "weights.pkl"
            config.touch()
            weights.touch()

            class Model:
                def get_soft_mask(self, image):
                    self.last_semantic_segmentation = np.full(
                        image.shape[:2], 10, np.int16
                    )
                    return np.full(image.shape[:2], 0.375, np.float32)

            args = SimpleNamespace(
                model="maskomaly",
                input=input_dir,
                output=root / "cache",
                recursive=False,
                config_file=config,
                weights=weights,
                masks=4,
                analysis_file=None,
            )
            manifest = folder_inference.export_folder(
                args, model_loader=lambda model_name, model_args: Model()
            )
            payload = read_manifest(manifest)
            self.assertEqual(payload["kind"], "raas-objectomaly-folder-inputs")
            self.assertEqual(payload["entries"][0]["source_name"], "my example.jpg")
            coarse = np.load(
                str(resolve_cached_path(manifest, payload["entries"][0]["coarse_map"])),
                allow_pickle=False,
            )
            self.assertEqual(coarse.dtype, np.float32)
            np.testing.assert_allclose(coarse, 0.375)

    def test_custom_visualizations_are_written_from_separate_cache_roots(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is not installed")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_root = root / "source"
            refined_root = root / "refined"
            image_rel = Path("images/custom-folder/000000_one.npy")
            coarse_rel = Path("coarse/maskomaly/custom-folder/000000_one.npy")
            semantic_rel = Path("semantic/maskomaly/custom-folder/000000_one.npy")
            fused_rel = Path("fused/maskomaly/custom-folder/000000_one.npy")
            refined_rel = Path("refined/maskomaly/custom-folder/000000_one.npy")
            protected_rel = Path(
                "protected_candidates/maskomaly/custom-folder/000000_one.npy"
            )
            for base, relative, value in (
                (source_root, image_rel, np.full((3, 4, 3), 80, np.uint8)),
                (source_root, coarse_rel, np.full((3, 4), 0.25, np.float32)),
                (source_root, semantic_rel, np.full((3, 4), 10, np.int16)),
                (refined_root, fused_rel, np.full((3, 4), 0.65, np.float32)),
                (refined_root, refined_rel, np.full((3, 4), 0.75, np.float32)),
                (refined_root, protected_rel, np.ones((3, 4), np.uint8)),
            ):
                (base / relative).parent.mkdir(parents=True, exist_ok=True)
                np.save(str(base / relative), value, allow_pickle=False)
            source_manifest = source_root / "manifest-maskomaly.json"
            entry = {
                "dataset": "custom-folder",
                "fid": "000000_one",
                "source_name": "one.jpg",
                "height": 3,
                "width": 4,
                "image_bgr": str(image_rel),
                "coarse_map": str(coarse_rel),
                "semantic_map": str(semantic_rel),
                "fused_map": str(fused_rel),
                "refined_map": str(refined_rel),
                "protected_candidates": str(protected_rel),
            }
            write_manifest(
                source_manifest,
                {
                    "kind": "raas-objectomaly-folder-inputs",
                    "source_model": "maskomaly",
                    "entries": [entry],
                },
            )
            refined_manifest = refined_root / "manifest-objectomaly-maskomaly.json"
            write_manifest(
                refined_manifest,
                {
                    "kind": "raas-objectomaly-folder-refined",
                    "source_model": "maskomaly",
                    "input_manifest": str(source_manifest),
                    "entries": [entry],
                },
            )
            folder_inference.render_outputs(refined_manifest, refined_root, 0.5)
            visual = refined_root / "visualizations"
            self.assertTrue((visual / "comparison/000000_one.jpg").is_file())
            binary = cv2.imread(
                str(visual / "binary_mask/000000_one.png"), cv2.IMREAD_GRAYSCALE
            )
            np.testing.assert_array_equal(binary, 255)

    def test_global_fusion_suppresses_sky_but_preserves_compact_contrast(self):
        masks = np.zeros((1, 10, 10), dtype=bool)
        masks[0, 4:6, 4:6] = True
        bundle = SimpleNamespace(
            n=1,
            masks=masks,
            area=np.array([4], np.int32),
            quality=np.array([0.95], np.float32),
            bbox=np.array([[4, 4, 6, 6]], np.int32),
        )
        score = np.full((10, 10), 0.6, np.float32)
        score[4:6, 4:6] = 0.9
        config = {
            "class_factors": {"10": 0.25},
            "background_class_ids": [10],
            "candidates": {
                "max_area_frac": 0.1,
                "min_score": 0.5,
                "min_contrast": 0.2,
                "ring_radius": 1,
            },
        }
        fused, protected, diagnostics = apply_global_fusion(
            score,
            np.full((10, 10), 10, np.int16),
            np.zeros((10, 10, 3), np.uint8),
            bundle,
            config,
        )
        self.assertAlmostEqual(float(fused[0, 0]), 0.15, places=5)
        self.assertAlmostEqual(float(fused[4, 4]), 0.9, places=5)
        self.assertTrue(protected[4, 4])
        self.assertEqual(diagnostics["protected_count"], 1)

    def test_global_fusion_clip_can_restore_airborne_candidate(self):
        masks = np.zeros((1, 10, 10), dtype=bool)
        masks[0, 4:6, 4:6] = True
        bundle = SimpleNamespace(
            n=1,
            masks=masks,
            area=np.array([4], np.int32),
            quality=np.array([0.95], np.float32),
            bbox=np.array([[4, 4, 6, 6]], np.int32),
        )

        class Validator:
            def score_regions(self, image, input_bundle, indices):
                return {0: {"clip_margin": 0.1, "ood_prompt": "flying tire"}}

        config = {
            "class_factors": {"10": 0.2},
            "background_class_ids": [10],
            "candidates": {
                "max_area_frac": 0.1,
                "clip_probe_min_score": 0.1,
                "min_score": 0.9,
                "min_contrast": 0.5,
                "clip_margin_threshold": 0.0,
                "clip_candidate_floor": 0.65,
            },
        }
        fused, protected, diagnostics = apply_global_fusion(
            np.full((10, 10), 0.4, np.float32),
            np.full((10, 10), 10, np.int16),
            np.zeros((10, 10, 3), np.uint8),
            bundle,
            config,
            clip_validator=Validator(),
        )
        self.assertAlmostEqual(float(fused[4, 4]), 0.65, places=5)
        self.assertTrue(protected[4, 4])
        self.assertEqual(diagnostics["candidates"][0]["reasons"], ["clip_ood"])

    def test_export_is_lossless_bgr_and_float32(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            frame = SimpleNamespace(
                image=np.full((2, 3, 3), [10, 20, 30], dtype=np.uint8),
                dset_name="AnomalyTrack-validation",
                fid="one",
            )

            class Evaluation:
                dataset_name = "AnomalyTrack-validation"

                def __len__(self):
                    return 1

                def get_frames(self):
                    return iter([frame])

            class Model:
                def get_soft_mask(self, image):
                    self.last_semantic_segmentation = np.full(
                        image.shape[:2], 10, np.int16
                    )
                    np.testing.assert_array_equal(image[0, 0], [30, 20, 10])
                    return np.full((2, 3), 0.25, dtype=np.float64)

            entries = exporter.export_evaluation(Evaluation(), Model(), "maskomaly", output)
            self.assertEqual(len(entries), 1)
            image = np.load(str(output / entries[0]["image_bgr"]), allow_pickle=False)
            coarse = np.load(str(output / entries[0]["coarse_map"]), allow_pickle=False)
            semantic = np.load(
                str(output / entries[0]["semantic_map"]), allow_pickle=False
            )
            np.testing.assert_array_equal(image[0, 0], [30, 20, 10])
            self.assertEqual(coarse.dtype, np.float32)
            self.assertEqual(semantic.dtype, np.int16)

    def test_refinement_calls_oasc_before_mbp_and_writes_map(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_rel = Path("images") / "d" / "one.npy"
            coarse_rel = Path("coarse") / "maskomaly" / "d" / "one.npy"
            (root / image_rel).parent.mkdir(parents=True)
            (root / coarse_rel).parent.mkdir(parents=True)
            np.save(str(root / image_rel), np.full((2, 3, 3), [30, 20, 10], np.uint8))
            np.save(str(root / coarse_rel), np.full((2, 3), 0.4, np.float32))
            manifest = root / "inputs.json"
            write_manifest(manifest, {"entries": []})
            calls = []

            def apply_oasc(coarse, bundle, variant, **params):
                calls.append(("oasc", variant))
                return coarse * 0.5

            def apply_mbp(calibrated, variant, **kwargs):
                calls.append(("mbp", variant))
                return calibrated + 0.1

            dependencies = {
                "has_bundle": lambda *args, **kwargs: False,
                "postprocess": lambda bundle, **kwargs: bundle,
                "save_bundle": lambda *args, **kwargs: None,
                "load_bundle": lambda *args, **kwargs: FakeBundle(),
                "apply_oasc": apply_oasc,
                "apply_mbp": apply_mbp,
            }
            config = {
                "sam": {"variant": "vit_h"},
                "postprocess": {},
                "oasc": {"variant": "quality_aware_residual_blending", "params": {}},
                "mbp": {"variant": "boundary_band_residual", "params": {}},
            }
            entry = {
                "dataset": "d",
                "evaluation_dataset": "d",
                "fid": "one",
                "height": 2,
                "width": 3,
                "image_bgr": str(image_rel),
                "coarse_map": str(coarse_rel),
                "source_model": "maskomaly",
            }
            result = refinement.refine_entry(
                entry,
                manifest,
                root / "out",
                config,
                dependencies,
                "all",
                FakeGenerator(),
            )
            self.assertEqual(calls, [
                ("oasc", "quality_aware_residual_blending"),
                ("mbp", "boundary_band_residual"),
            ])
            refined = np.load(str(root / "out" / result["refined_map"]), allow_pickle=False)
            np.testing.assert_allclose(refined, 0.3)

    def test_refinement_writes_global_fusion_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_rel = Path("images/d/one.npy")
            coarse_rel = Path("coarse/maskomaly/d/one.npy")
            semantic_rel = Path("semantic/maskomaly/d/one.npy")
            for relative, value in (
                (image_rel, np.zeros((10, 10, 3), np.uint8)),
                (coarse_rel, np.full((10, 10), 0.6, np.float32)),
                (semantic_rel, np.full((10, 10), 10, np.int16)),
            ):
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                np.save(str(root / relative), value, allow_pickle=False)
            manifest = root / "inputs.json"
            write_manifest(manifest, {"entries": []})
            masks = np.zeros((1, 10, 10), dtype=bool)
            masks[0, 4:6, 4:6] = True
            bundle = SimpleNamespace(
                n=1,
                masks=masks,
                area=np.array([4], np.int32),
                quality=np.array([0.95], np.float32),
                bbox=np.array([[4, 4, 6, 6]], np.int32),
            )
            dependencies = {
                "has_bundle": lambda *args, **kwargs: False,
                "postprocess": lambda value, **kwargs: value,
                "save_bundle": lambda *args, **kwargs: None,
                "load_bundle": lambda *args, **kwargs: bundle,
                "apply_oasc": lambda coarse, *args, **kwargs: coarse,
                "apply_mbp": lambda fused, *args, **kwargs: fused,
            }
            config = {
                "sam": {"variant": "vit_h"},
                "postprocess": {},
                "oasc": {"variant": "test", "params": {}},
                "global_fusion": {
                    "enabled": True,
                    "class_factors": {"10": 0.25},
                    "background_class_ids": [10],
                    "candidates": {"max_area_frac": 0.1, "min_score": 0.9},
                    "clip": {"enabled": False},
                },
                "mbp": {"variant": "test", "params": {}},
            }
            entry = {
                "dataset": "d",
                "fid": "one",
                "height": 10,
                "width": 10,
                "image_bgr": str(image_rel),
                "coarse_map": str(coarse_rel),
                "semantic_map": str(semantic_rel),
                "source_model": "maskomaly",
            }
            result = refinement.refine_entry(
                entry,
                manifest,
                root / "out",
                config,
                dependencies,
                "all",
                generator=SimpleNamespace(generate=lambda image: bundle),
            )
            self.assertIn("fused_map", result)
            self.assertIn("protected_candidates", result)
            fused = np.load(str(root / "out" / result["fused_map"]), allow_pickle=False)
            np.testing.assert_allclose(fused, 0.15)

    def test_import_maps_every_official_frame_and_calculates_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            datasets_root = root / "datasets"
            output = root / "official"
            entries = []
            for dataset_name, spec in importer.smiyc.DATASETS.items():
                dataset_root = datasets_root / spec["directory"]
                (dataset_root / "images").mkdir(parents=True)
                (dataset_root / "labels_masks").mkdir()
                rel = Path("refined") / dataset_name / "one.npy"
                (root / rel).parent.mkdir(parents=True)
                np.save(str(root / rel), np.full((2, 3), 0.7, np.float32))
                entries.append(
                    {
                        "dataset": dataset_name,
                        "evaluation_dataset": dataset_name,
                        "fid": "one",
                        "height": 2,
                        "width": 3,
                        "refined_map": str(rel),
                    }
                )
            manifest = root / "refined.json"
            write_manifest(
                manifest,
                {
                    "kind": "raas-objectomaly-refined",
                    "source_model": "maskomaly",
                    "entries": entries,
                },
            )

            class Evaluation:
                def __init__(self, method_name, dataset_name, threaded_saver=False):
                    self.method_name = method_name
                    self.dataset_name = dataset_name
                    self.frame = SimpleNamespace(
                        image=np.zeros((2, 3, 3), np.uint8),
                        dset_name=dataset_name,
                        fid="one",
                    )

                def __len__(self):
                    return 1

                def get_frames(self):
                    return iter([self.frame])

                def save_output(self, frame, prediction):
                    np.testing.assert_allclose(prediction, 0.7)
                    path = (
                        output
                        / "anomaly_p"
                        / self.method_name
                        / frame.dset_name
                        / "one.hdf5"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()

                def wait_to_finish_saving(self):
                    pass

                def calculate_metric_from_saved_outputs(self, metric, **kwargs):
                    if metric == "PixBinaryClass":
                        return SimpleNamespace(area_PRC=0.8, tpr95_fpr=0.2)
                    return SimpleNamespace(sIoU_gt=0.6, prec_pred=0.5, f1_mean=0.55)

            args = SimpleNamespace(
                manifest=manifest,
                datasets_root=datasets_root,
                output=output,
                method_name=None,
                phase="all",
                visualize=False,
            )
            rows = importer.execute(args, evaluation_class=Evaluation)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["model"], "objectomaly_maskomaly")
            self.assertEqual(rows[0]["AUPR"], 80.0)


if __name__ == "__main__":
    unittest.main()
