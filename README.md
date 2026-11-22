# Pi-Sigma Neural Network Versus Classical Classifiers for Kecimen and Besni Raisin Variety Classification

[![Open In Colab](https://google.com)](https://google.com)
[![Dataset: UCI Raisin](https://shields.io)](https://uci.edu)

This repository contains the source code and generated figures accompanying the manuscript:

> **Pi-Sigma Neural Network Versus Classical Classifiers for Kecimen and Besni Raisin Variety Classification: A Leakage-Free Repeated Cross-Validation Study**  
> Barka Piyinkir Ndahi, Ayuba Audu Balami, Abraham Ayegba Alfa and  Muhammad Bashir Abdullahi

---

## Overview

Published machine-learning comparisons on the public UCI Raisin dataset often rely on differing validation protocols, making reported accuracies difficult to rank reliably. This study evaluates a compact **Pi-Sigma Neural Network (PSNN)** ($K = 2, 3, 4$ summing units) against six classical baseline configurations under a single, leakage-free **10 × repeated stratified 5-fold cross-validation** protocol (50 fold-runs, `seed = 42`).

Key findings:
* **Competitive accuracy with minimal parameters:** A 2-unit PSNN (18 trainable parameters) achieved **86.44 ± 1.90%** accuracy (Macro-F1: `0.864`, AUC: `0.924`, MCC: `0.731`), showing no statistically significant difference after Holm correction compared to RBF-SVM (`87.03%`), Logistic Regression (`86.67%`), Quadratic Logistic Regression (`86.56%`), or Random Forest (`85.91%`).
* **Fast inference:** PSNN inference required **0.0023 ms per grain**, matching Logistic Regression and running ~5× faster than RBF-SVM and ~50× faster than Random Forest.
* **Threshold-driven error profile:** Class-specific precision/recall trade-offs are governed by the decision threshold rather than inherent model bias.

---

## Repository Contents

| File | Description |
| :--- | :--- |
| `psnn_raisin_revised.py` | Main Python script implementing the NumPy-based PSNN, classical baselines, leakage-free 10 × 5-fold CV, corrected resampled *t*-tests with Holm adjustment, timing benchmarks, and figure generation. |
| `fig_accuracy_box.png` | **Figure 1:** Distribution of test accuracy across the 50 fold-runs for all nine evaluated model configurations. |
| `fig_roc.png` | **Figure 2:** Out-of-fold ROC curves and AUC scores from the first cross-validation repeat. |
| `fig_threshold.png` | **Figure 3:** Operating-point analysis illustrating precision and recall trade-offs for Kecimen and Besni across decision thresholds ($0.05$ to $0.95$). |

---

## Dataset

The study uses the public **Raisin Dataset** from the UCI Machine Learning Repository:
* **Source:** Çınar et al. (2020) — [UCI Machine Learning Repository (ID 850)](https://uci.edu)
* **DOI:** `10.24432/C5660T` (License: CC BY 4.0)
* **Composition:** 900 raisin grains (450 Kecimen, 450 Besni) described by 7 morphological features (`Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `Eccentricity`, `ConvexArea`, `Extent`).

---

## Quick Start & Reproducibility

### Option 1: Run in Browser (Google Colab)
Click the **Open in Colab** badge at the top of this page to inspect and run the full experiment directly in your browser without local installation.

### Option 2: Run Locally
1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd psnn-raisin
   ```

2. **Install dependencies:**
   The experiments were verified using Python 3.12 with the following core packages:
   ```bash
   pip install numpy scikit-learn scipy pandas matplotlib
   ```

3. **Execute the script:**
   ```bash
   python psnn_raisin_revised.py
   ```
   The script automatically loads the dataset (locally or from the UCI repository), executes all 50 fold-runs, prints the summary tables and statistical comparisons, and saves the three output figures.

---

## Citation

If you use this code or protocol in your research, please cite:

```bibtex
@article{ndahi2026psnn,
  title={Pi-Sigma Neural Network Versus Classical Classifiers for Kecimen and Besni Raisin Variety Classification: A Leakage-Free Repeated Cross-Validation Study},
  author={Ndahi, Barka Piyinkir and Salami, Fati Oiza and Alfa, Abraham Ayegba and Balami, Ayuba Audu},
  year={2026},
  note={Code available at: https://github.com}
}
```
