"""Automated Unit & Integration Test Suite for Steps 1-7."""
import unittest
from pathlib import Path
from backend.app.services.dicom_processor import DicomPreprocessor
from backend.app.services.inference_engine import MedicalAIInferenceEngine
from backend.app.services.gradcam_explainer import GradCAMExplainer
from backend.app.models_db import compute_audit_hash


class TestMedAIPipeline(unittest.TestCase):
    def test_step1_to_step6_end_to_end(self):
        dcm_files = sorted(Path("data/raw_dicom_samples").glob("*.dcm"))
        self.assertEqual(len(dcm_files), 3)

        pre = DicomPreprocessor()
        eng = MedicalAIInferenceEngine()
        cam = GradCAMExplainer()

        for dcm_file in dcm_files:
            raw_bytes = dcm_file.read_bytes()
            tensor_2d, display_u8, meta = pre.preprocess_bytes(raw_bytes)
            self.assertEqual(tensor_2d.shape, (1, 3, 224, 224))
            self.assertTrue(meta["patient_pseudo_id"].startswith("PT-"))

            pred = eng.predict_2d_classification(tensor_2d, hint_id=dcm_file.stem)
            self.assertIn("predicted_label", pred)
            self.assertTrue(0.0 <= pred["confidence_score"] <= 1.0)

            pseudo_vol = pre.preprocess_3d_volume([display_u8.astype("float32") / 255.0] * 16)
            self.assertEqual(pseudo_vol.shape, (1, 1, 96, 96, 96))
            mask_3d, seg_metrics = eng.predict_3d_segmentation(pseudo_vol)
            self.assertEqual(mask_3d.shape, (1, 1, 96, 96, 96))
            self.assertIn("lesion_voxel_fraction_pct", seg_metrics)

            cam_norm, rgba, blended = cam.generate_heatmap(tensor_2d, pred["predicted_index"], display_u8)
            self.assertEqual(rgba.shape, (256, 256, 4))
            self.assertEqual(blended.shape, (256, 256, 3))

        h1 = compute_audit_hash(1, "RAD-101", "APPROVE", "Verified RLL opacity", "2026-10-30T08:00:00Z", "0" * 64)
        self.assertEqual(len(h1), 64)


if __name__ == "__main__":
    unittest.main()
