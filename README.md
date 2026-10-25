# CSE 472 — Machine Learning Sessional

Machine learning coursework from **Bangladesh University of Engineering and Technology (BUET)**, covering data preprocessing, classification, ensemble learning, feedforward neural networks, dimensionality reduction, and expectation-maximization.

## Assignments

| Assignment | Topics | Dataset(s) | Code |
| --- | --- | --- | --- |
| 1 | Data preprocessing and feature selection | IBM HR Employee Attrition | [Notebook](DataPreProcessing/1905062.ipynb) |
| 2 | Logistic regression, bagging, and stacking | Telco Churn, Adult, Credit Card Fraud | [Notebook](LogisticRegression,stacking,bagging/1905062/1905062.ipynb) |
| 3 | Feedforward neural network from scratch | Fashion-MNIST | [Notebook](FNN/1905062/1905062.ipynb) |
| 4 | PCA and expectation-maximization | Assignment-provided datasets | [Notebook](PCM,EMalgorithm/1905062/1905062.ipynb) |

## Assignment 1 — Data Preprocessing

Preprocessing and feature analysis using the **IBM HR Analytics Employee Attrition** dataset.

- Explore data distributions, missing values, and duplicates.
- Clean data and encode categorical variables.
- Apply standardization and min-max scaling.
- Rank features by their correlation with employee attrition.
- Evaluate the selected features using logistic regression.

[Notebook](DataPreProcessing/1905062.ipynb) · [Assignment](DataPreProcessing/CSE472_Assignment_1.pdf) · [Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

## Assignment 2 — Logistic Regression and Ensemble Learning

Implementation and evaluation of classification methods on **Telco Customer Churn**, **UCI Adult**, and **Credit Card Fraud Detection** datasets.

- Implement logistic regression with NumPy, including sigmoid, gradient descent, and prediction.
- Train nine logistic regression learners on bootstrap samples for bagging.
- Experiment with voting and stacking using a logistic regression meta-classifier.
- Measure accuracy, sensitivity, specificity, precision, F1-score, AUROC, and AUPR.
- Compare classifier results across datasets.

[Notebook](LogisticRegression,stacking,bagging/1905062/1905062.ipynb) · [Report](LogisticRegression,stacking,bagging/1905062/1905062.pdf) · [Assignment](LogisticRegression,stacking,bagging/CSE472_Assignment_2.pdf)

Datasets: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) · [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult) · [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Assignment 3 — Feedforward Neural Network

A **NumPy-based feedforward neural network** for ten-class image classification on **Fashion-MNIST**. `torchvision` is used for dataset loading; the network components are implemented with NumPy.

- Implement fully connected layers, ReLU, and softmax.
- Implement forward and backward propagation.
- Use categorical cross-entropy loss and Adam optimization.
- Add batch normalization and dropout.
- Experiment with architectures and learning rates; analyze training curves and confusion matrices.

**Reported test results:** 89.13% accuracy and 0.8902 macro F1-score for the selected model.

[Notebook](FNN/1905062/1905062.ipynb) · [Report](FNN/1905062/Report_1905062.pdf) · [Saved model](FNN/1905062/1905062_model.pkl) · [Assignment](FNN/Assignment3-v2.0.pdf) · [Dataset](https://github.com/zalandoresearch/fashion-mnist)

## Assignment 4 — PCA and Expectation-Maximization

Two unsupervised learning experiments:

**Principal Component Analysis (PCA)**
- Standardize the input data.
- Compute the covariance matrix and its eigendecomposition using NumPy.
- Project high-dimensional data onto the first two principal components.
- Compare PCA visualization with t-SNE and UMAP.

**Expectation-Maximization (EM)**
- Implement EM for a two-component Poisson mixture.
- Compute component responsibilities in the E-step.
- Update Poisson parameters and mixture weights in the M-step.

[Notebook](PCM,EMalgorithm/1905062/1905062.ipynb) · [Report](PCM,EMalgorithm/1905062/1905062.pdf) · [Assignment](PCM,EMalgorithm/Assignment%204_%20PCA-EM.pdf)

## Repository Structure

```text
CSE472_MachineLearning/
├── DataPreProcessing/
│   └── 1905062.ipynb
├── LogisticRegression,stacking,bagging/
│   └── 1905062/
│       └── 1905062.ipynb
├── FNN/
│   └── 1905062/
│       ├── 1905062.ipynb
│       └── 1905062_model.pkl
└── PCM,EMalgorithm/
    └── 1905062/
        └── 1905062.ipynb
```

Each assignment directory also includes assignment instructions and/or reports.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/lara-62/CSE472_MachineLearning.git
cd CSE472_MachineLearning
```

Create a virtual environment and install the libraries used across the notebooks:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter torch torchvision umap-learn
```

Start Jupyter:

```bash
jupyter notebook
```

Open the notebook for the assignment you want to run. Some notebooks use locally referenced input files, so download or provide their datasets in the expected working directory:

- **Assignment 1:** `WA_Fn-UseC_-HR-Employee-Attrition.csv`
- **Assignment 2:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`, `adult.data` / `adult.test`, or `creditcard.csv`, depending on the selected experiment.
- **Assignment 3:** Fashion-MNIST is downloaded by `torchvision` on first use.
- **Assignment 4:** `pca_data.txt` and `em_data.txt` (not included in the repository).

The notebooks contain experiment-specific cells and settings that may need to be selected or adjusted before running.
