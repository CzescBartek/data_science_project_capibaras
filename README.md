# Skin Lesion Classification

[![Project Status: Active](https://img.shields.io/badge/Project%20Status-Complete-green.svg)](#)
[![Data Science](https://img.shields.io/badge/Domain-Data%20Science-blue.svg)](#)
[![Year](https://img.shields.io/badge/Year-2026-orange.svg)](#)

An automated machine learning pipeline for binary classification of skin lesions into **cancerous** and **non-cancerous** classes using clinically meaningful, interpretable features based on the medical ABC criteria and advanced texture descriptors.

---

## 👥 Authors & Team
* **Group Name**: CapiBARA  
* **Course/Context**: Project in Data Science (2026)  

### Team Members:
* Laura Betev
* Jakob Daniel Viernes Bregenhoj
* Krzysztof Jacek Marcinkiewicz
* Bartosz Pliszka
* Utku Yoztyurk

---

## 📌 1. Introduction
Skin cancer is a critical public health concern where early detection significantly improves patient survival rates and treatment outcomes. Given the global shortage and limited access to clinical dermatologists in many regions, automated decision-support tools serve as vital screening aids.

This project investigates whether machine learning models can accurately distinguish between cancerous and non-cancerous skin lesions using highly interpretable, clinically relevant visual descriptors. Our pipeline extracts features mapped to the traditional dermatological **ABC criteria** (**A**symmetry, **B**order irregularity, and **C**olour variation) directly from segmented lesion images, alongside modern texture descriptors like Local Binary Patterns (LBP). 

---

## 📊 2. Dataset & Methodology

### 2.1 Dataset: PAD-UFES-20
The project utilizes the **PAD-UFES-20 dataset**, which consists of **2,103 clinical images** of skin lesions collected via smartphones, accompanied by detailed patient metadata. 

The original dataset spans six distinct diagnostic categories. For this study, the problem was framed as a **binary classification task** to optimize the detection of malignant potential:

| Original Diagnosis | Class Mapping | Binary Target | Sample Count |
| :--- | :--- | :---: | :---: |
| **Basal Cell Carcinoma (BCC)** | Cancerous | `1` | 815 |
| **Squamous Cell Carcinoma (SCC)** | Cancerous | `1` | 184 |
| **Malignant Melanoma (MEL)** | Cancerous | `1` | 49 |
| **Actinic Keratosis (ACK)** | Non-Cancerous | `0` | 634 |
| **Melanocytic Nevus (NEV)** | Non-Cancerous | `0` | 220 |
| **Seborrheic Keratosis (SEK)** | Non-Cancerous | `0` | 201 |

* **Total Cancerous Images**: 1,048  
* **Total Non-Cancerous Images**: 1,055  
* *Note: The overall binary dataset is nearly perfectly balanced, mitigating majority class prediction bias.*

### 2.2 Structured Engineering Pipeline
The end-to-end development followed an explicit, iterative approach:
1. **Metadata Inspection & Diagnostics Alignment**: Mapping multi-class outputs into structured binary targets.
2. **Mask Validation**: Filtering data based on the availability and correctness of segmentation masks (e.g., matching `PAT_XXXX.png` with `PAT_XXXX_mask.png`). Fully black or missing masks were filtered out.
3. **Artifact Preprocessing**: Dedicated routines to remove hair obstructions and clinical pen markings.
4. **Feature Extraction**: Computing shape-based geometry (asymmetry, compactness, convexity), HSV color distributions, and local textures.
5. **Model Optimization**: Training via 5-fold cross-validation on an 80% development set, with final evaluation on a 20% holdout set.

---

## 🧹 3. Preprocessing & Artifact Removal

Clinical skin images often contain structural and artificial noise that degrades feature extraction. Two primary noise sources were addressed:

### 3.1 Pen Mark Removal
Clinicians frequently draw blue ink boundaries around suspicious lesions before imaging. These markings introduce highly non-uniform artificial boundaries and deep blue color vectors. 
* **Approach**: Filtered pixels dynamically within a strict, predefined Hue range in the **HSV color space** corresponding to clinical ink.
* **Result**: Detected pixels are masked and inpainted prior to feature calculations, ensuring border and color metrics describe the tissue, not the pen.

### 3.2 Hair Feature Extraction & Inpainting
Hair occlusions introduce fake edges and artificial dark structures over the lesion area.
* **Hair Coverage Ratio**: Computed using a $5	imes5$ cross-shaped kernel via **Top-Hat** and **Black-Hat morphological transformations**. The intensity profile from the Black-Hat operation isolated dark hair structures against light skin. The hair coverage ratio is calculated as:
  $$\text{Hair Coverage Ratio} = \frac{\text{White Hair Pixels}}{\text{Total Pixels}}$$
* **Adaptive Inpainting Strategy**:
  * **Coverage < 0.005**: Hair removal skipped to preserve raw skin details.
  * **Coverage [0.005 - 0.035]**: Moderate filtering using a $12	imes12$ structural element.
  * **Coverage > 0.035**: Aggressive filtering using a $25	imes25$ element targeting thick hair structures.
* **Inpainting**: Executed using **OpenCV’s Telea algorithm** to smoothly blend hair pixels out based on surrounding healthy tissue values.

---

## 📐 4. Feature Engineering

### 4.1 Asymmetry (A)
Computed directly from the binary segmentation mask $M$. The mask is split along its horizontal and vertical centroids, mirrored, and evaluated using an Exclusive-OR (XOR) logic gate:
$$A = \frac{\sum \text{XOR}_H + \sum \text{XOR}_V}{2 \cdot \sum M}$$
A score of $0$ indicates flawless spatial symmetry, while values scaling toward $1$ imply high asymmetry, standard in malignant mutations.

### 4.2 Border Irregularity (B)
Extracted using two distinctive geometric measures:
1. **Compactness**: Measures perimeter circularity.
   $$\text{Compactness} = \frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2}$$
   Values close to $1.0$ indicate circular benign layouts; lower scores reflect highly irregular, jagged malignant boundaries.
2. **Convexity**: Evaluates localized concavity.
   $$\text{Convexity} = \frac{\text{Area}}{\text{Area}_{\text{hull}}}$$
   Where $\text{Area}_{\text{hull}}$ is the area bounding the convex hull. Low convexity exposes significant indentations.

### 4.3 Colour Variation (C)
To closely mirror the human visual system, pixels are transformed from RGB into the **HSV space**, explicitly isolating chromatic attributes from luminance.
* **Superpixel Segmentation**: Implemented using **SLIC (Simple Linear Iterative Clustering)** to partition the lesion into uniform, edge-adhering clusters.
* **Variance Extraction**: Calculated the statistical variances of **Hue (H)**, **Saturation (S)**, and **Value (V)** inside the lesion area. Malignant tissue yields significant Saturation and Value variances due to chaotic pigmentation dynamics.

### 4.4 Local Binary Patterns (LBP) Texture Features
To compensate for structural abnormalities missed by macro-geometric attributes (e.g., scaling, roughness, crusting), an **LBP descriptor** was added.
* Compares every individual pixel against 8 equidistant neighbors.
* Encodes regional variations into a 256-bin normalized histogram, adding 256 dense texture dimensions.

---

## 🤖 5. Models & Evaluation Results

The dataset was split using a fixed random seed (`seed=0`): **80% Development Set** and **20% Test Set**. Optimal hyperparameters were derived using **5-fold cross-validation** evaluating the Area Under the ROC Curve (ROC AUC).

### 5.1 Evaluated Classifiers
1. **K-Nearest Neighbours (KNN)**: Evaluated across various $k$ spaces. The optimal structure peaked at $k=50$, rendering a CV Mean AUC of **0.7057**.
2. **Random Forest (RF) Ensemble**: Optimized across tree counts (`n_estimators`) and maximum tree constraints (`max_depth`). 
   * **Optimal Tree Depth**: 13
   * **Optimal Estimators**: 200

### 5.2 Performance Metrics (Best Baseline: Random Forest)
The Random Forest model outperformed KNN significantly due to its robust subspace sampling architecture, mitigating high dimensionality issues introduced by the 256 LBP texture bins.

| Metric | Random Forest Value |
| :--- | :---: |
| **Accuracy** | 0.6857 |
| **Precision** | 0.6639 |
| **Recall (Sensitivity)** | 0.7524 |
| **F1-Score** | 0.7054 |
| **ROC AUC** | **0.7370** |

### 5.3 Confusion Matrix Summary
* **True Benign (0)**: 130 | **False Malignant (FP)**: 80
* **False Benign (FN)**: 52  | **True Malignant (1)**: 158

---

## 🔍 6. Model Interpretability & SHAP Analysis

To validate clinical alignment, global and local model predictions were explained using **SHAP (SHapley Additive exPlanations)**.

* **Global Feature Insights**: **Asymmetry (FEATURE_A)** emerged as the strongest predictive driver across the forest, followed closely by **Color Saturation (FEATURE_B_S)**. High saturation values shifted outputs toward a cancerous prediction.
* **LBP Significance**: Specific LBP bins (such as `LBP_54`, `LBP_221`, `LBP_39`) consistently ranked above raw border compactness, proving that local surface texture patterns add significant discriminative signals.
* **Local Waterfall Breakdown (Image 200)**: Exemplified how an individual prediction ($f(x) = 0.644$ vs baseline $E[f(X)] = 0.5$) is constructed. While a high symmetry profile (`FEATURE_A = 0.611`) pushed the probability down by $-0.05$, the strong color variance (`FEATURE_B_S = 0.71`) pushed it back up by $+0.03$, combined with cumulative positive texture updates ($+0.13$ across secondary features) resulting in a correct cancerous assignment.

---

## ⚠️ 7. Limitations & Future Outlook

While a solid foundational framework was achieved, several constraints limit production-level deployment:
1. **Segmentation Dependency**: Shape-based descriptors (Asymmetry, Compactness) heavily rely on mask clean-cut precision. Noisy or misaligned masks directly distort features.
2. **The Melanoma Scarcity Gap**: Though the overall binary classes are balanced (~1,050 samples each), the cancerous cohort is deeply skewed toward Basal Cell Carcinoma (815 samples) over **Malignant Melanoma (49 samples)**. Clinically, this is dangerous; the model can score a high overall AUC while systematically misclassifying lethal Melanomas due to lack of training representations.
3. **High-Dimensionality Redundancy**: Incorporating 256 LBP variables risks overfitting. Introducing **Principal Component Analysis (PCA)** to lower LBP space density would improve model robustness.
4. **Clinical Explanability Trade-off**: High-performing LBP bins lack immediate anatomical equivalents that can be easily communicated to patients compared to plain asymmetry metrics.

**Future Enhancements**: Transitioning toward automated mask generation networks (e.g., SAM or U-Net), expanding dataset distribution with highly targeted Melanoma images, and benchmarking against end-to-end Deep Learning architectures.

---

## 📚 References
1. Achanta, R. et al. "SLIC superpixels compared to state-of-the-art superpixel methods." *IEEE TPAMI*, 34(11), 2274-2282, 2012.
2. Ojala, T., Pietikäinen, M. & Mäenpää, T. "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns." *IEEE TPAMI*, 24(7), 971-987, 2002.
3. Pacheco, A. G. C. et al. "PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones." *Data in Brief*, 32:106221, 2020.
4. Rehman, M. et al. "Machine learning based skin lesion segmentation method with novel borders and hair removal techniques." *PMC*, 0275781, 2022.
5. Roky, A. H. et al. "Overview of skin cancer types and prevalence rates across continents." *Cancer Pathogenesis and Therapy*, 3, 2024.
6. Shalu, S. & Kamboj, A. "Melanoma detection using HSV and YCbCr color space." *IJCA*, 179(40), 22-25, 2018.
