# System Architecture & REST API Documentation

## 1. End-to-End Architecture
1. **Orthanc DICOM Server (`:4242` / `:8042`)**: Receives DICOM instances (`C-STORE` or HTTP upload) and exposes `/changes` and `/instances/{id}/file`.
2. **FastAPI + PyTorch/MONAI Service (`:8000`)**:
   - Polls Orthanc `/changes` via `OrthancPollingWorker`.
   - Scrubs PHI tags (`DicomPreprocessor`), applies Hounsfield Unit windowing (`[-1000, 400]` HU), and normalizes intensities.
   - Runs `DenseNet121` (2D 14-class thoracic classification) and `MONAI 3D U-Net` (volumetric segmentation).
   - Generates `Grad-CAM` spatial attribution heatmaps and stores PNG artifacts + predictions in SQLite (`medai_audit.db`).
3. **React Radiology Workstation (`:3000`)**:
   - Displays the DICOM worklist, interactive scan viewer with Grad-CAM opacity slider, confidence scores, and Approve/Reject sign-off controls backed by an immutable SHA-256 hash-chained audit table.

## 2. REST API Endpoints
- `GET /health` — Returns service health, loaded model identifiers, and mandatory clinical disclaimer.
- `GET /api/v1/studies` — Lists all analyzed DICOM studies with predictions and review status.
- `POST /api/v1/studies/upload` — Accepts a `.dcm` file (`multipart/form-data`), runs inference + Grad-CAM, and returns the created study record.
- `GET /api/v1/studies/{study_id}/image/{scan|heatmap}` — Serves the rendered PNG scan or RGBA Grad-CAM overlay.
- `POST /api/v1/studies/{study_id}/review` — Records an `APPROVE` or `REJECT` decision with clinical notes and computes a tamper-evident SHA-256 chain hash.
- `GET /api/v1/audit-logs` — Returns the chronological SHA-256 hash-chained radiologist decision log.
