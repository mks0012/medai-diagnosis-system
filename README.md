# AI-Assisted Medical Imaging and Diagnosis System



An end-to-end medical imaging AI platform where X-ray, CT, or MRI DICOM scans are uploaded or pulled from an **Orthanc DICOM Server**, analyzed by deep learning models (**DenseNet121** 2D classifier and **MONAI 3D U-Net** volumetric segmenter), visualized with **Grad-CAM** explanation heatmaps, and reviewed by a radiologist (**Approve / Reject**) with an immutable **SHA-256 hash-chained audit log**.

---

## Option A: Run Everything in 1 Command with Docker Compose (Recommended)

1. Unzip `medai-diagnosis-system.zip` and open the `medai-diagnosis-system` folder in **VS Code**.
2. Make sure **Docker Desktop** is running, then run in your VS Code terminal:
   ```bash
   docker compose up --build
   ```
3. Open in your browser:
   - **React Radiology Workstation**: http://localhost:3000
   - **FastAPI Interactive Swagger Docs**: http://localhost:8000/docs
   - **Orthanc DICOM Server UI**: http://localhost:8042

---

## Option B: Run Locally in VS Code Without Docker (Python + Node.js)

If you prefer to run the backend and frontend directly in VS Code without Docker:

### Terminal 1 — Start the FastAPI + PyTorch/MONAI Backend
```bash
cd medai-diagnosis-system
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Verify Step 1 & Step 2 DICOM preprocessing & run automated unit tests:
python -m backend.app.services.dicom_processor --verify
python -m unittest discover -s tests -p "test_*.py" -v

# Launch the FastAPI server (auto-seeds the 3 sample DICOM studies in data/raw_dicom_samples/):
uvicorn backend.app.main:app --reload --port 8000
```

### Terminal 2 — Start the React Radiology Viewer
```bash
cd medai-diagnosis-system/frontend
npm install
npm run dev
```
Then open **http://localhost:3000** in your browser! You can also click **+ Upload DICOM (.dcm)** in the top-right corner and select any of the `.dcm` files inside `data/raw_dicom_samples/`.
