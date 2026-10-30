"""
Asynchronous Orthanc DICOM Polling Service.
Continuously polls Orthanc `/changes` REST API, pulls newly arrived instances (`/instances/{id}/file`),
runs PyTorch/MONAI classification + 3D segmentation + Grad-CAM, and persists results for the React UI.
"""
import asyncio
import json
import logging
from pathlib import Path

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from backend.app.models_db import StudyRecord

logger = logging.getLogger("orthanc_poller")


class OrthancPollingWorker:
    def __init__(
        self,
        orthanc_url: str,
        session_factory,
        preprocessor,
        inference_engine,
        explainer,
        artifact_dir: str = "artifacts_store",
    ):
        self.orthanc_url = orthanc_url.rstrip("/")
        self.session_factory = session_factory
        self.preprocessor = preprocessor
        self.inference_engine = inference_engine
        self.explainer = explainer
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.last_seq = 0

    def process_dicom_bytes(self, instance_id: str, dicom_bytes: bytes) -> StudyRecord:
        db = self.session_factory()
        try:
            existing = db.query(StudyRecord).filter_by(orthanc_instance_id=instance_id).first()
            if existing:
                return existing

            tensor_2d, display_uint8, meta = self.preprocessor.preprocess_bytes(dicom_bytes)
            pred = self.inference_engine.predict_2d_classification(tensor_2d, hint_id=instance_id)

            pseudo_vol = self.preprocessor.preprocess_3d_volume([display_uint8.astype("float32") / 255.0] * 16)
            _, seg_metrics = self.inference_engine.predict_3d_segmentation(pseudo_vol)

            _, rgba_heatmap, _ = self.explainer.generate_heatmap(
                tensor_2d, pred["predicted_index"], display_uint8
            )

            scan_path = self.artifact_dir / f"{instance_id}_scan.png"
            heat_path = self.artifact_dir / f"{instance_id}_gradcam.png"
            scan_path.write_bytes(self.preprocessor.encode_png_bytes(display_uint8))
            heat_path.write_bytes(self.preprocessor.encode_png_bytes(rgba_heatmap))

            record = StudyRecord(
                orthanc_instance_id=instance_id,
                study_instance_uid=meta["study_instance_uid"],
                patient_pseudo_id=meta["patient_pseudo_id"],
                modality=meta["modality"],
                body_part=meta["body_part"],
                view_position=meta["view_position"],
                predicted_label=pred["predicted_label"],
                confidence_score=pred["confidence_score"],
                all_probabilities_json=json.dumps(pred["all_probabilities"]),
                segmentation_metrics_json=json.dumps(seg_metrics),
                model_version=pred["model_version"],
                status="PENDING_REVIEW",
                scan_png_path=str(scan_path),
                heatmap_png_path=str(heat_path),
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            return record
        finally:
            db.close()

    async def poll_forever(self, interval_seconds: int = 5):
        if not HTTPX_AVAILABLE:
            return
        async with httpx.AsyncClient(timeout=10.0) as client:
            while True:
                try:
                    resp = await client.get(
                        f"{self.orthanc_url}/changes", params={"since": self.last_seq, "limit": 50}
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        for change in data.get("Changes", []):
                            if change.get("ChangeType") == "NewInstance":
                                inst_id = change["ID"]
                                dcm_resp = await client.get(f"{self.orthanc_url}/instances/{inst_id}/file")
                                if dcm_resp.status_code == 200:
                                    self.process_dicom_bytes(inst_id, dcm_resp.content)
                        self.last_seq = data.get("Last", self.last_seq)
                except Exception as exc:
                    logger.debug("Orthanc poll waiting (%s)", exc)
                await asyncio.sleep(interval_seconds)
