# Individual Intern Reflection — AI / Machine Learning Specialization Track

## 1. What I Owned End-to-End
- Designed and built the full-stack medical imaging pipeline integrating **Orthanc DICOM Server**, **PyTorch/MONAI** (`DenseNet121` & `3D U-Net`), **Grad-CAM** explainability, **DVC** data versioning, **FastAPI**, and the **React** radiologist viewer.
- Implemented the tamper-evident **SHA-256 hash-chained radiologist audit log** and containerized the 3-service deployment using **Docker Compose**.

## 2. Key Technical Learnings
- **DICOM Standard & HU Physics**: Learned how to parse NEMA PS3.10 tags, scrub PHI fields for HIPAA Safe-Harbor compliance, and apply `RescaleSlope`/`RescaleIntercept` Hounsfield Unit windowing (`[-1000, 400]` HU).
- **Clinical Explainability (Grad-CAM)**: Learned how spatial activation maps expose shortcut learning on medical hardware and support human-in-the-loop radiologist verification.
- **Zero-Cost MLOps**: Combined free cloud GPU training (Kaggle Dual-T4) with DVC-tracked checkpoints and fast local CPU container inference (`< 1.2s` per study).
