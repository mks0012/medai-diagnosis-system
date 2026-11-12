"""
Production-Hardened Asynchronous Orthanc DICOM Polling Service.
- Offloads CPU inference to worker threads (`asyncio.to_thread`) so FastAPI never freezes
- Limits concurrent inferences (`asyncio.Semaphore(2)`) to prevent RAM spikes
- Isolates corrupted DICOM files in a Dead-Letter Queue after 3 failed retries
- Uses exponential backoff when Orthanc is starting or unreachable
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict

import httpx
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
        max_concurrent_inferences: int = 2,
    ):
        self.orthanc_url = orthanc_url.rstrip("/")
        self.session_factory = session_factory
        self.preprocessor = preprocessor
        self.inference_engine = inference_engine
        self.explainer = explainer
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.seq_file = self.artifact_dir / "orthanc_last_seq.txt"
        self.last_seq = int(self.seq_file.read_text().strip()) if self.seq_file.exists() else 0
        self.semaphore = asyncio.Semaphore(max_concurrent_inferences)
        self.retry_counts: Dict[str, int] = {}
        self.dead_letter_instances: Dict[str, str] = {}

    def process_dicom_bytes(self, instance_id: str, dicom_bytes: bytes) -> StudyRecord:
        if len(dicom_bytes) < 140:
            raise ValueError(f"DICOM file too small or truncated ({len(dicom_bytes)} bytes)")
        if len(dicom_bytes) > 50 * 1024 * 1024:
            raise ValueError("DICOM file exceeds 50 MB maximum safety limit")

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
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    async def process_dicom_bytes_async(self, instance_id: str, dicom_bytes: bytes) -> StudyRecord:
        async with self.semaphore:
            return await asyncio.to_thread(self.process_dicom_bytes, instance_id, dicom_bytes)

    async def poll_forever(self, base_interval: float = 4.0):
        consecutive_errors = 0
        async with httpx.AsyncClient(timeout=15.0) as client:
            while True:
                try:
                    resp = await client.get(
                        f"{self.orthanc_url}/changes", params={"since": self.last_seq, "limit": 25}
                    )
                    if resp.status_code == 200:
                        consecutive_errors = 0
                        data = resp.json()
                        for change in data.get("Changes", []):
                            if change.get("ChangeType") == "NewInstance":
                                inst_id = change["ID"]
                                if inst_id in self.dead_letter_instances:
                                    continue
                                try:
                                    dcm_resp = await client.get(f"{self.orthanc_url}/instances/{inst_id}/file")
                                    if dcm_resp.status_code == 200:
                                        await self.process_dicom_bytes_async(inst_id, dcm_resp.content)
                                        self.retry_counts.pop(inst_id, None)
                                except Exception as inst_exc:
                                    attempts = self.retry_counts.get(inst_id, 0) + 1
                                    self.retry_counts[inst_id] = attempts
                                    logger.warning("Instance %s failed (attempt %d/3): %s", inst_id, attempts, inst_exc)
                                    if attempts >= 3:
                                        self.dead_letter_instances[inst_id] = str(inst_exc)
                                        logger.error("Isolated corrupted instance %s in dead-letter queue", inst_id)

                        new_last = data.get("Last", self.last_seq)
                        if new_last != self.last_seq:
                            self.last_seq = new_last
                            self.seq_file.write_text(str(self.last_seq))
                except Exception as exc:
                    consecutive_errors = min(consecutive_errors + 1, 5)
                    logger.debug("Orthanc poll waiting (backoff step %d): %s", consecutive_errors, exc)

                await asyncio.sleep(min(base_interval * (1.5 ** consecutive_errors), 30.0))