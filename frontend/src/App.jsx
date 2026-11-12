import React, { useState, useEffect } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export default function App() {
  const [studies, setStudies] = useState([]);
  const [selectedStudy, setSelectedStudy] = useState(null);
  const [auditLogs, setAuditLogs] = useState([]);
  const [showOverlay, setShowOverlay] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(0.58);
  const [radiologistId, setRadiologistId] = useState("RAD-104");
  const [radiologistName, setRadiologistName] = useState("Dr. K. Rao, MD (Radiology)");
  const [clinicalNotes, setClinicalNotes] = useState("");
  const [uploading, setUploading] = useState(false);
  const [toast, setToast] = useState("");

  const fetchAll = async () => {
    try {
      const [stResp, logResp] = await Promise.all([
        fetch(`${API_BASE}/api/v1/studies`),
        fetch(`${API_BASE}/api/v1/audit-logs`),
      ]);
      const stData = await stResp.json();
      const logData = await logResp.json();
      const list = stData.studies || [];
      setStudies(list);
      setAuditLogs(logData.audit_logs || []);
      setSelectedStudy((prev) => {
        if (!prev) return list[0] || null;
        return list.find((s) => s.id === prev.id) || list[0] || null;
      });
    } catch (err) {
      console.error("Waiting for backend API...", err);
    }
  };

  useEffect(() => {
    fetchAll();
    const timer = setInterval(fetchAll, 4000);
    return () => clearInterval(timer);
  }, []);

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);
    const resp = await fetch(`${API_BASE}/api/v1/studies/upload`, {
      method: "POST",
      body: formData,
    });
    if (resp.ok) {
      const created = await resp.json();
      await fetchAll();
      setSelectedStudy(created);
      setToast(`Uploaded and analyzed ${file.name} (${created.predicted_label})`);
    } else {
      const errData = await resp.json().catch(() => ({}));
      setToast(`Upload rejected: ${errData.detail || "Invalid DICOM file"}`);
    }
    setUploading(false);
    e.target.value = "";
  };

  const handleDecision = async (decision) => {
    if (!selectedStudy) return;
    if (clinicalNotes.trim().length < 5) {
      setToast("Please enter at least 5 characters of clinical notes before signing off.");
      return;
    }
    const resp = await fetch(`${API_BASE}/api/v1/studies/${selectedStudy.id}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        radiologist_id: radiologistId,
        radiologist_name: radiologistName,
        decision,
        clinical_notes: clinicalNotes,
      }),
    });
    if (resp.ok) {
      setClinicalNotes("");
      setToast(`Recorded ${decision} decision in SHA-256 Audit Log.`);
      await fetchAll();
    }
  };

  return (
    <div style={{ background: "#0B0F19", color: "#F1F5F9", minHeight: "100vh", fontFamily: "system-ui, sans-serif" }}>
      <div style={{ background: "#7F1D1D", color: "#FECACA", padding: "8px 16px", fontSize: "13px", fontWeight: 600, textAlign: "center" }}>
        FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY — NOT FOR CLINICAL USE. AI predictions and Grad-CAM heatmaps require qualified radiologist verification.
      </div>

      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "14px 24px", borderBottom: "1px solid #1E293B" }}>
        <div>
          <h1 style={{ margin: 0, fontSize: "20px", fontWeight: 700 }}>MedAI PACS & Explainable Diagnosis Workstation</h1>
          <span style={{ fontSize: "12px", color: "#94A3B8" }}>Orthanc DICOM Server • MONAI 3D U-Net • DenseNet121 • Grad-CAM</span>
        </div>
        <label style={{ background: "#2563EB", color: "#fff", padding: "9px 16px", borderRadius: "6px", cursor: "pointer", fontSize: "14px", fontWeight: 600 }}>
          {uploading ? "Analyzing DICOM..." : "+ Upload DICOM (.dcm)"}
          <input type="file" accept=".dcm,application/dicom" onChange={handleUpload} style={{ display: "none" }} />
        </label>
      </header>

      {toast && (
        <div style={{ background: "#1E3A8A", color: "#DBEAFE", padding: "8px 24px", fontSize: "13px", display: "flex", justifyContent: "space-between" }}>
          <span>{toast}</span>
          <button onClick={() => setToast("")} style={{ background: "transparent", border: "none", color: "#93C5FD", cursor: "pointer" }}>Dismiss</button>
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "300px 1fr 360px", gap: "16px", padding: "16px" }}>
        <div style={{ background: "#111827", borderRadius: "8px", padding: "14px", border: "1px solid #1E293B" }}>
          <h3 style={{ marginTop: 0, fontSize: "13px", color: "#94A3B8", textTransform: "uppercase" }}>DICOM Worklist ({studies.length})</h3>
          {studies.map((s) => (
            <div
              key={s.id}
              onClick={() => setSelectedStudy(s)}
              style={{
                padding: "10px",
                marginBottom: "8px",
                borderRadius: "6px",
                cursor: "pointer",
                background: selectedStudy?.id === s.id ? "#1E3A8A" : "#1F2937",
                border: "1px solid #334155",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "13px", fontWeight: 600 }}>
                <span>{s.patient_pseudo_id}</span>
                <span style={{ color: s.status === "APPROVED" ? "#4ADE80" : s.status === "REJECTED" ? "#F87171" : "#FBBF24" }}>{s.status}</span>
              </div>
              <div style={{ fontSize: "12px", color: "#CBD5E1", marginTop: "4px" }}>
                {s.modality} ({s.view_position}) → <strong>{s.predicted_label}</strong> ({(s.confidence_score * 100).toFixed(1)}%)
              </div>
            </div>
          ))}
        </div>

        <div style={{ background: "#111827", borderRadius: "8px", padding: "16px", border: "1px solid #1E293B", textAlign: "center" }}>
          {selectedStudy ? (
            <>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
                <span style={{ fontSize: "13px", fontWeight: 600 }}>
                  Study UID: {selectedStudy.study_instance_uid} | Modality: {selectedStudy.modality} ({selectedStudy.view_position})
                </span>
                <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                  <label style={{ fontSize: "13px", cursor: "pointer" }}>
                    <input type="checkbox" checked={showOverlay} onChange={(e) => setShowOverlay(e.target.checked)} /> Grad-CAM Overlay
                  </label>
                  <input type="range" min="0" max="1" step="0.05" value={overlayOpacity} onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))} />
                </div>
              </div>

              <div style={{ position: "relative", display: "inline-block", width: "440px", height: "440px", background: "#000", borderRadius: "8px", overflow: "hidden", border: "1px solid #334155" }}>
                <img src={`${API_BASE}${selectedStudy.scan_url}`} alt="DICOM Scan" style={{ width: "100%", height: "100%", objectFit: "contain" }} />
                {showOverlay && (
                  <img
                    src={`${API_BASE}${selectedStudy.heatmap_url}`}
                    alt="Grad-CAM Overlay"
                    style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", objectFit: "contain", opacity: overlayOpacity, pointerEvents: "none" }}
                  />
                )}
              </div>
            </>
          ) : (
            <p style={{ color: "#64748B", marginTop: "120px" }}>Select or upload a DICOM study to inspect.</p>
          )}
        </div>

        <div style={{ background: "#111827", borderRadius: "8px", padding: "16px", border: "1px solid #1E293B" }}>
          <h3 style={{ marginTop: 0, fontSize: "13px", color: "#94A3B8", textTransform: "uppercase" }}>AI Diagnostic Findings</h3>
          {selectedStudy && (
            <>
              <div style={{ background: "#1E293B", padding: "12px", borderRadius: "6px", marginBottom: "14px" }}>
                <div style={{ fontSize: "12px", color: "#94A3B8" }}>Primary Prediction ({selectedStudy.model_version})</div>
                <div style={{ fontSize: "22px", fontWeight: 700, color: "#38BDF8", margin: "4px 0" }}>{selectedStudy.predicted_label}</div>
                <div style={{ fontSize: "14px", color: "#E2E8F0" }}>Confidence Score: <strong>{(selectedStudy.confidence_score * 100).toFixed(1)}%</strong></div>
                {selectedStudy.segmentation_metrics?.lesion_voxel_fraction_pct !== undefined && (
                  <div style={{ fontSize: "12px", color: "#94A3B8", marginTop: "6px" }}>
                    MONAI 3D U-Net Volumetric Mask Ratio: <strong>{selectedStudy.segmentation_metrics.lesion_voxel_fraction_pct}%</strong>
                  </div>
                )}
              </div>

              <h4 style={{ fontSize: "13px", color: "#94A3B8", marginBottom: "8px" }}>Radiologist Review & Sign-Off</h4>
              <input value={radiologistId} onChange={(e) => setRadiologistId(e.target.value)} placeholder="Radiologist ID" style={{ width: "100%", boxSizing: "border-box", padding: "8px", marginBottom: "8px", borderRadius: "4px", background: "#0F172A", color: "#fff", border: "1px solid #334155" }} />
              <input value={radiologistName} onChange={(e) => setRadiologistName(e.target.value)} placeholder="Radiologist Name" style={{ width: "100%", boxSizing: "border-box", padding: "8px", marginBottom: "8px", borderRadius: "4px", background: "#0F172A", color: "#fff", border: "1px solid #334155" }} />
              <textarea rows={3} value={clinicalNotes} onChange={(e) => setClinicalNotes(e.target.value)} placeholder="Enter clinical review notes for the immutable audit log..." style={{ width: "100%", boxSizing: "border-box", padding: "8px", marginBottom: "10px", borderRadius: "4px", background: "#0F172A", color: "#fff", border: "1px solid #334155" }} />
              <div style={{ display: "flex", gap: "10px" }}>
                <button onClick={() => handleDecision("APPROVE")} style={{ flex: 1, background: "#16A34A", color: "#fff", border: "none", padding: "10px", borderRadius: "6px", fontWeight: 700, cursor: "pointer" }}>
                  ✓ Approve
                </button>
                <button onClick={() => handleDecision("REJECT")} style={{ flex: 1, background: "#DC2626", color: "#fff", border: "none", padding: "10px", borderRadius: "6px", fontWeight: 700, cursor: "pointer" }}>
                  ✕ Reject
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      <div style={{ margin: "0 16px 24px 16px", background: "#111827", borderRadius: "8px", padding: "16px", border: "1px solid #1E293B" }}>
        <h3 style={{ marginTop: 0, fontSize: "13px", color: "#94A3B8", textTransform: "uppercase" }}>Immutable Radiologist Decision Audit Log (SHA-256 Hash-Chained)</h3>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid #334155", textAlign: "left", color: "#94A3B8" }}>
              <th style={{ padding: "8px" }}>Timestamp (UTC)</th>
              <th style={{ padding: "8px" }}>Study</th>
              <th style={{ padding: "8px" }}>Radiologist</th>
              <th style={{ padding: "8px" }}>AI Finding</th>
              <th style={{ padding: "8px" }}>Decision</th>
              <th style={{ padding: "8px" }}>Clinical Notes</th>
              <th style={{ padding: "8px" }}>SHA-256 Record Hash</th>
            </tr>
          </thead>
          <tbody>
            {auditLogs.map((log) => (
              <tr key={log.id} style={{ borderBottom: "1px solid #1E293B" }}>
                <td style={{ padding: "8px" }}>{log.timestamp_utc}</td>
                <td style={{ padding: "8px" }}>#{log.study_id}</td>
                <td style={{ padding: "8px" }}>{log.radiologist_name} ({log.radiologist_id})</td>
                <td style={{ padding: "8px" }}>{log.ai_predicted_label} ({(log.ai_confidence * 100).toFixed(1)}%)</td>
                <td style={{ padding: "8px", fontWeight: 700, color: log.decision === "APPROVE" ? "#4ADE80" : "#F87171" }}>{log.decision}</td>
                <td style={{ padding: "8px" }}>{log.clinical_notes}</td>
                <td style={{ padding: "8px", fontFamily: "monospace", fontSize: "11px", color: "#94A3B8" }}>{log.record_hash.slice(0, 18)}...</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}