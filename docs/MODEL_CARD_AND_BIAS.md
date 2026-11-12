# Model Card: Limitations, Clinical Bias Analysis & Regulatory Notice


## 1. Model Architectures & Datasets
- **2D Classification**: `DenseNet121` trained via transfer learning on NIH ChestX-ray14 / RSNA Pneumonia Detection (Macro AUROC `0.834`).
- **3D Segmentation**: `MONAI 3D U-Net` (`spatial_dims=3`, residual units, DiceCE loss) trained on Medical Segmentation Decathlon `Task06_Lung` (Mean 3D Dice `0.784`).

## 2. Known Limitations & Algorithmic Bias Risks
1. **Shortcut Learning on Medical Hardware (Chest Tubes & Pacemakers)**:
   - Public radiographs of treated *Pneumothorax* frequently include thoracic drainage tubes. CNNs may attend to the chest tube rather than the visceral pleural line. Our Grad-CAM overlay exposes hardware shortcut activation so the reviewing radiologist can click **Reject**.
2. **AP (Portable) vs. PA (Standing) Projection Bias**:
   - Anterior-Posterior (AP) portable scans magnify the heart shadow due to beam divergence, increasing false-positive *Cardiomegaly* scores compared to upright Posterior-Anterior (PA) scans.
3. **Demographic & Scanner Domain Shift**:
   - Models trained on adult tertiary hospital cohorts (NIH / Stanford CheXpert) require local recalibration before evaluation on pediatric populations or non-standard detector exposure profiles.
4. **NLP-Extracted Label Noise**:
   - Automated report-mining labels in public chest X-ray datasets carry ~5-10% label noise on subtle overlapping findings such as *Infiltration*.
