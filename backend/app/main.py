"""
FastAPI Service for AI-Assisted Medical Imaging & Diagnosis System.
Uses non-blocking threadpool inference (`process_dicom_bytes_async`) and exposes dead-letter diagnostics.
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.app.models_db import init_db, StudyRecord, RadiologistAuditLog
from backend.app.services.dicom_processor import DicomPreprocessor
from backend.app.services.inference_engine import MedicalAIInferenceEngine
from backend.app.services.gradcam_explainer import GradCAMExplainer
from backend.app.services.orthanc_poller import OrthancPollingWorker

CLINICAL_NOTICE = (
    "FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY — NOT FOR CLINICAL USE. "
    "Predictions and Grad-CAM heatmaps must not be used for clinical diagnosis."
)

DB_URL = os.getenv("DATABASE_URL", "sqlite:///./medai_audit.db")
ORTHANC_URL = os.getenv("ORTHANC_URL", "http://orthanc:8042")
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "./artifacts_store")

SessionLocal = init_db(DB_URL)
preprocessor = DicomPreprocessor()
engine = MedicalAIInferenceEngine()
explainer = GradCAMExplainer(engine.classifier)
worker = OrthancPollingWorker(ORTHANC_URL, SessionLocal, preprocessor, engine, explainer, ARTIFACT_DIR)


def seed_initial_studies():
    sample_dir = Path("data/raw_dicom_samples")
    if not sample_dir.exists():
        return
    for dcm_path in sorted(sample_dir.glob("*.dcm")):
        try:
            worker.process_dicom_bytes(dcm_path.stem, dcm_path.read_bytes())
        except Exception as exc:
            print(f"Seed warning for {dcm_path.name}: {exc}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await asyncio.to_thread(seed_initial_studies)
    task = asyncio.create_task(worker.poll_forever(base_interval=4.0))
    yield
    task.cancel()


app = FastAPI(
    title="AI-Assisted Medical Imaging & Diagnosis API",
    description=CLINICAL_NOTICE,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewRequest(BaseModel):
    radiologist_id: str = Field(..., min_length=2, examples=["RAD-0042"])
    radiologist_name: str = Field(..., min_length=2, examples=["Dr. A. Sharma, MD (Radiology)"])
    decision: str = Field(..., pattern="^(APPROVE|REJECT)$")
    clinical_notes: str = Field(..., min_length=5, examples=["Grad-CAM focuses appropriately on RLL consolidation."])


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "disclaimer": CLINICAL_NOTICE,
        "models_loaded": ["DenseNet121-2D", "MONAI-3D-UNet"],
        "orthanc_last_seq": worker.last_seq,
        "dead_letter_count": len(worker.dead_letter_instances),
        "dead_letter_instances": worker.dead_letter_instances,
    }


@app.post("/api/v1/studies/upload")
async def upload_dicom_study(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    instance_id = f"UPLOAD-{file.filename}-{int(datetime.now(timezone.utc).timestamp() * 1000)}"
    try:
        record = await worker.process_dicom_bytes_async(instance_id, raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted DICOM file: {exc}") from exc
    return _serialize_study(record)


@app.get("/api/v1/studies")
def list_studies():
    db = SessionLocal()
    try:
        studies = db.query(StudyRecord).order_by(StudyRecord.created_at.desc()).all()
        return {"disclaimer": CLINICAL_NOTICE, "studies": [_serialize_study(s) for s in studies]}
    finally:
        db.close()


@app.get("/api/v1/studies/{study_id}/image/{kind}")
def get_study_image(study_id: int, kind: str):
    db = SessionLocal()
    try:
        study = db.query(StudyRecord).filter_by(id=study_id).first()
        if not study:
            raise HTTPException(status_code=404, detail="Study not found")
        path = study.scan_png_path if kind == "scan" else study.heatmap_png_path
        if not Path(path).exists():
            raise HTTPException(status_code=404, detail="Rendered image missing")
        return FileResponse(path, media_type="image/png")
    finally:
        db.close()


@app.post("/api/v1/studies/{study_id}/review")
def submit_radiologist_review(study_id: int, payload: ReviewRequest):
    db = SessionLocal()
    try:
        study = db.query(StudyRecord).filter_by(id=study_id).first()
        if not study:
            raise HTTPException(status_code=404, detail="Study not found")

        last_entry = db.query(RadiologistAuditLog).order_by(RadiologistAuditLog.id.desc()).first()
        prev_hash = last_entry.record_hash if last_entry else "0" * 64
        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()

        record_hash = RadiologistAuditLog.compute_hash(
            study_id=study.id,
            radiologist_id=payload.radiologist_id,
            decision=payload.decision,
            clinical_notes=payload.clinical_notes,
            timestamp_iso=now_iso,
            previous_hash=prev_hash,
        )

        audit_entry = RadiologistAuditLog(
            study_id=study.id,
            radiologist_id=payload.radiologist_id,
            radiologist_name=payload.radiologist_name,
            decision=payload.decision,
            ai_predicted_label=study.predicted_label,
            ai_confidence=study.confidence_score,
            clinical_notes=payload.clinical_notes,
            previous_hash=prev_hash,
            record_hash=record_hash,
            timestamp_utc=now_utc,
        )
        study.status = "APPROVED" if payload.decision == "APPROVE" else "REJECTED"
        db.add(audit_entry)
        db.commit()
        db.refresh(audit_entry)

        return {
            "status": "recorded",
            "study_status": study.status,
            "audit_log_id": audit_entry.id,
            "record_hash": audit_entry.record_hash,
            "previous_hash": audit_entry.previous_hash,
            "timestamp_utc": now_iso,
            "disclaimer": CLINICAL_NOTICE,
        }
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database transaction failed: {exc}") from exc
    finally:
        db.close()


app.get("/api/v1/audit-logs")
def list_audit_logs():
    db = SessionLocal()
    try:
        logs = db.query(RadiologistAuditLog).order_by(RadiologistAuditLog.id.desc()).all()
        return {
            "disclaimer": CLINICAL_NOTICE,
            "count": len(logs),
            "audit_logs": [
                {
                    "id": item.id,
                    "study_id": item.study_id,
                    "radiologist_id": item.radiologist_id,
                    "radiologist_name": item.radiologist_name,
                    "decision": item.decision,
                    "ai_predicted_label": item.ai_predicted_label,
                    "ai_confidence": item.ai_confidence,
                    "clinical_notes": item.clinical_notes,
                    "previous_hash": item.previous_hash,
                    "record_hash": item.record_hash,
                    "timestamp_utc": item.timestamp_utc.isoformat(),
                }
                for item in logs
            ],
        }
    finally:
        db.close()


def _serialize_study(s: StudyRecord) -> dict:
    return {
        "id": s.id,
        "orthanc_instance_id": s.orthanc_instance_id,
        "study_instance_uid": s.study_instance_uid,
        "patient_pseudo_id": s.patient_pseudo_id,
        "modality": s.modality,
        "body_part": s.body_part,
        "view_position": s.view_position,
        "predicted_label": s.predicted_label,
        "confidence_score": s.confidence_score,
        "all_probabilities": json.loads(s.all_probabilities_json),
        "segmentation_metrics": json.loads(s.segmentation_metrics_json or "{}"),
        "model_version": s.model_version,
        "status": s.status,
        "scan_url": f"/api/v1/studies/{s.id}/image/scan",
        "heatmap_url": f"/api/v1/studies/{s.id}/image/heatmap",
        "created_at": s.created_at.isoformat(),
        "disclaimer": CLINICAL_NOTICE,
    }