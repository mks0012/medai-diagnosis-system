"""
SQLAlchemy Models for Studies, AI Predictions, and Hash-Chained Radiologist Audit Logs.
Also includes a pure-SQLite helper for environments without SQLAlchemy installed.
"""
from datetime import datetime, timezone
import hashlib

try:
    from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, create_engine
    from sqlalchemy.orm import declarative_base, relationship, sessionmaker
    SQLALCHEMY_AVAILABLE = True
    Base = declarative_base()

    class StudyRecord(Base):
        __tablename__ = "studies"

        id = Column(Integer, primary_key=True, index=True)
        orthanc_instance_id = Column(String(128), unique=True, index=True, nullable=False)
        study_instance_uid = Column(String(128), index=True, nullable=False)
        patient_pseudo_id = Column(String(64), nullable=False)
        modality = Column(String(16), nullable=False)
        body_part = Column(String(32), default="CHEST")
        view_position = Column(String(16), default="PA")
        predicted_label = Column(String(64), nullable=False)
        confidence_score = Column(Float, nullable=False)
        all_probabilities_json = Column(Text, nullable=False)
        segmentation_metrics_json = Column(Text, nullable=True)
        model_version = Column(String(64), nullable=False)
        status = Column(String(32), default="PENDING_REVIEW")
        scan_png_path = Column(String(256), nullable=False)
        heatmap_png_path = Column(String(256), nullable=False)
        created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

        audit_logs = relationship("RadiologistAuditLog", back_populates="study", cascade="all, delete-orphan")

    class RadiologistAuditLog(Base):
        __tablename__ = "radiologist_audit_logs"

        id = Column(Integer, primary_key=True, index=True)
        study_id = Column(Integer, ForeignKey("studies.id"), nullable=False, index=True)
        radiologist_id = Column(String(64), nullable=False)
        radiologist_name = Column(String(128), nullable=False)
        decision = Column(String(16), nullable=False)
        ai_predicted_label = Column(String(64), nullable=False)
        ai_confidence = Column(Float, nullable=False)
        clinical_notes = Column(Text, nullable=False)
        previous_hash = Column(String(64), nullable=False)
        record_hash = Column(String(64), nullable=False, unique=True)
        timestamp_utc = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

        study = relationship("StudyRecord", back_populates="audit_logs")

        @staticmethod
        def compute_hash(study_id: int, radiologist_id: str, decision: str, clinical_notes: str, timestamp_iso: str, previous_hash: str) -> str:
            payload = f"{study_id}|{radiologist_id}|{decision}|{clinical_notes}|{timestamp_iso}|{previous_hash}"
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def init_db(db_url: str = "sqlite:///./medai_audit.db"):
        engine = create_engine(db_url, connect_args={"check_same_thread": False} if "sqlite" in db_url else {})
        Base.metadata.create_all(bind=engine)
        return sessionmaker(autocommit=False, autoflush=False, bind=engine)

except ImportError:
    SQLALCHEMY_AVAILABLE = False
    StudyRecord = None
    RadiologistAuditLog = None
    init_db = None


def compute_audit_hash(study_id: int, radiologist_id: str, decision: str, clinical_notes: str, timestamp_iso: str, previous_hash: str) -> str:
    payload = f"{study_id}|{radiologist_id}|{decision}|{clinical_notes}|{timestamp_iso}|{previous_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
