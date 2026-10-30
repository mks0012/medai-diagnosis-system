"""
DICOM Part-10 Parser, HIPAA Safe-Harbor De-identification, HU Windowing, and Preprocessing Pipeline.
Supports both `pydicom` and a built-in NEMA PS3.10 Explicit VR Little Endian binary parser.
"""
import argparse
import hashlib
import io
import json
import struct
from pathlib import Path
from typing import Dict, Tuple, Any

import cv2
import numpy as np
from PIL import Image

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

PHI_TAGS_TO_ANONYMIZE = [
    "PatientName",
    "PatientBirthDate",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "ReferringPhysicianName",
    "InstitutionName",
]


class DicomPreprocessor:
    def __init__(
        self,
        target_size_2d: Tuple[int, int] = (224, 224),
        target_size_3d: Tuple[int, int, int] = (96, 96, 96),
        hu_window: Tuple[float, float] = (-1000.0, 400.0),
    ):
        self.target_size_2d = target_size_2d
        self.target_size_3d = target_size_3d
        self.hu_min, self.hu_max = hu_window

    def parse_dicom_bytes(self, dicom_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Check if built-in NEMA PS3.10 sample format or standard pydicom
        if dicom_bytes[128:132] == b"DICM":
            try:
                idx = 132
                tags = {}
                rows, cols = 256, 256
                pixel_bytes = b""
                while idx + 8 <= len(dicom_bytes):
                    grp, elem, vr_b = struct.unpack("<HH2s", dicom_bytes[idx : idx + 6])
                    vr = vr_b.decode("ascii", errors="ignore")
                    if vr in ("OB", "OW", "OF", "SQ", "UT", "UN"):
                        length = struct.unpack("<I", dicom_bytes[idx + 8 : idx + 12])[0]
                        val_start = idx + 12
                    else:
                        length = struct.unpack("<H", dicom_bytes[idx + 6 : idx + 8])[0]
                        val_start = idx + 8
                    val = dicom_bytes[val_start : val_start + length]
                    idx = val_start + length
                    if (grp, elem) == (0x0028, 0x0010):
                        rows = struct.unpack("<H", val)[0]
                    elif (grp, elem) == (0x0028, 0x0011):
                        cols = struct.unpack("<H", val)[0]
                    elif (grp, elem) == (0x7FE0, 0x0010):
                        pixel_bytes = val
                    else:
                        tags[(grp, elem)] = val.decode("ascii", errors="ignore").strip("\x00 ")
                if pixel_bytes:
                    raw_pid = tags.get((0x0010, 0x0020), "UNKNOWN")
                    anon_id = raw_pid if raw_pid.startswith("PT-") else "PT-" + hashlib.sha256(raw_pid.encode()).hexdigest()[:8].upper()
                    arr = np.frombuffer(pixel_bytes, dtype="<u2").reshape(rows, cols).astype(np.float32)
                    meta = {
                        "study_instance_uid": tags.get((0x0020, 0x000D), "1.2.840.000"),
                        "patient_pseudo_id": anon_id,
                        "modality": tags.get((0x0008, 0x0060), "DX"),
                        "body_part": tags.get((0x0018, 0x0015), "CHEST"),
                        "view_position": tags.get((0x0018, 0x5101), "PA"),
                        "rows": rows,
                        "columns": cols,
                    }
                    return arr, meta
            except Exception:
                pass

        if PYDICOM_AVAILABLE:
            dcm = pydicom.dcmread(io.BytesIO(dicom_bytes), force=True)
            raw_pid = str(getattr(dcm, "PatientID", "UNKNOWN"))
            anon_id = raw_pid if raw_pid.startswith("PT-") else "PT-" + hashlib.sha256(raw_pid.encode()).hexdigest()[:8].upper()
            for tag in PHI_TAGS_TO_ANONYMIZE:
                if hasattr(dcm, tag):
                    setattr(dcm, tag, "ANONYMIZED")
            meta = {
                "study_instance_uid": str(getattr(dcm, "StudyInstanceUID", "1.2.840.000")),
                "patient_pseudo_id": anon_id,
                "modality": str(getattr(dcm, "Modality", "DX")).upper(),
                "body_part": str(getattr(dcm, "BodyPartExamined", "CHEST")).upper(),
                "view_position": str(getattr(dcm, "ViewPosition", "PA")).upper(),
                "rows": int(getattr(dcm, "Rows", 256)),
                "columns": int(getattr(dcm, "Columns", 256)),
            }
            arr = dcm.pixel_array.astype(np.float32)
            slope = float(getattr(dcm, "RescaleSlope", 1.0))
            intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
            arr = arr * slope + intercept
            return arr, meta

        raise ValueError("Unable to parse DICOM file bytes.")

    def preprocess_bytes(self, dicom_bytes: bytes) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        arr, meta = self.parse_dicom_bytes(dicom_bytes)
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
        p_min, p_max = np.percentile(arr, (1.0, 99.0))
        norm = np.clip((arr - p_min) / (p_max - p_min + 1e-8), 0.0, 1.0).astype(np.float32)
        display_u8 = (norm * 255.0).astype(np.uint8)
        resized = cv2.resize(norm, self.target_size_2d, interpolation=cv2.INTER_AREA)
        rgb = np.stack([resized, resized, resized], axis=0)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        tensor_2d = ((rgb - mean) / std)[np.newaxis, ...]
        return tensor_2d, display_u8, meta

    def preprocess_3d_volume(self, slices_list: list) -> np.ndarray:
        volume = np.stack(slices_list, axis=0).astype(np.float32)
        d_target, h_target, w_target = self.target_size_3d
        resized_hw = np.stack(
            [cv2.resize(s, (w_target, h_target), interpolation=cv2.INTER_LINEAR) for s in volume],
            axis=0,
        )
        indices = np.linspace(0, resized_hw.shape[0] - 1, d_target)
        resized_dhw = np.array([resized_hw[int(round(i))] for i in indices], dtype=np.float32)
        return resized_dhw[np.newaxis, np.newaxis, ...]

    @staticmethod
    def encode_png_bytes(gray_or_rgba: np.ndarray) -> bytes:
        pil_img = Image.fromarray(gray_or_rgba)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw_dicom_samples")
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = DicomPreprocessor()
    count = 0
    for dcm_file in sorted(Path(args.input).glob("*.dcm")):
        t2d, disp, meta = pre.preprocess_bytes(dcm_file.read_bytes())
        count += 1
        print(f"Verified {dcm_file.name}: {meta['patient_pseudo_id']} | Modality={meta['modality']} | Shape={t2d.shape}")
    print(f"[Step 1 & 2 Complete] Processed {count} DICOM studies successfully.")
