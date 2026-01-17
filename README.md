# Network Intrusion Detection System (NIDS) Optimization
![Python](https://img.shields.io/badge/Language-Python_3.12.12-blue?style=flat-square) ![Status](https://img.shields.io/badge/Status-Research_Complete-green?style=flat-square) ![Domain](https://img.shields.io/badge/Domain-Cybersecurity_%7C_Machine_Learning-orange?style=flat-square)

---

## Executive Summary

This repository hosts a comparative study on handling severe class imbalance in Network Intrusion Detection Systems (NIDS). The project develops two parallel machine learning pipelines to detect malicious network traffic. By contrasting **Data-Level Resampling** against **Algorithmic Cost-Sensitive Learning**, this project demonstrates that optimizing the model for the data's native distribution yields superior accuracy (99.14%) compared to synthetic data balancing.

**Dataset Access:** [Insert Link to Your Dataset Here]

---

## Technical Architecture

The project is divided into two distinct optimization strategies, each contained within its own dedicated notebook.

### Strategy A: Data-Level Optimization
**File:** `IBN_NIDS_Resampling_Optimization.ipynb`

* **Objective:** Eliminate class imbalance prior to model training.
* **Methodology:**
    * **Downsampling:** Reduction of majority class (Normal traffic).
    * **Upsampling:** Synthetic expansion of minority class (Attack traffic).
    * **Result:** A perfectly balanced 50/50 dataset.
* **Algorithm:** Standard Random Forest Classifier.

### Strategy B: Algorithmic-Level Optimization (Recommended)
**File:** `IBN_NIDS_Cost_Sensitive_Optimization.ipynb`

* **Objective:** Optimize the classifier to penalize misclassifications on the minority class without altering data distribution.
* **Methodology:**
    * **Imputation:** Random Forest regressor used to predict and fill missing categorical data.
    * **Class Weights:** Application of `class_weight='balanced'` to adjust cost functions.
    * **Tuning:** GridSearch Hyperparameter optimization (Entropy vs. Gini, Tree Depth).
* **Algorithm:** Cost-Sensitive Random Forest.

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/8c5aff50-8cce-45ba-a5be-357c2ba162f3" />

---

## Repository Structure

```text
├── IBN_NIDS_Cost_Sensitive_Optimization.ipynb   # Native distribution (Best Performance)
├── IBN_NIDS_Resampling_Optimization.ipynb       # Resampled distribution (Baseline)
├── README.md                                    # Project documentation
└── data/                                        # (User to create this directory)
```
## Benchmarking Results

The following table summarizes the performance metrics achieved by each approach on the test set.

| Metric | Data-Level Optimization (Resampling) | Algorithmic-Level Optimization (Cost-Sensitive) |
| :--- | :--- | :--- |
| **Accuracy** | 98.30% | **99.14%** |
| **Precision** | 98.0% | **99.3%** |
| **Recall** | 98.0% | **99.1%** |
| **ROC-AUC** | 98.0% | **99.1%** |

*Key Insight: The Cost-Sensitive approach (Algorithmic-Level) outperformed the Resampling method, demonstrating that preserving the original data distribution and using advanced imputation yields superior detection capabilities.*

## Installation and Usage

### Prerequisites
* Python 3.12.12
* Jupyter Notebook or Google Colab
* Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

### Instructions
1.  Clone this repository.
2.  Download the dataset from the link provided above.
3.  Update the `folder_path` variable in the notebooks to point to your local data directory.
4.  Run the notebooks in the order of interest (Resampling first for baseline, Cost-Sensitive second for advanced analysis).

## Future Work
* Implementation of Deep Learning models (CNN/LSTM) for feature extraction.
* Real-time traffic analysis integration.
* Expansion to multi-class classification to identify specific attack types.

## License
This project is open-source and available for educational and research purposes.
