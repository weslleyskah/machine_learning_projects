# Machine Learning Projects

This repository contains machine learning projects that explore and analyze datasets to compare the performance of different ML algorithms.

---

## Food Project

Regression task to predict the protein values of thousands of food items based on major food nutrient values: carbohydrate, energy, water, fat and nitrogen.

The datasets were acquired from the U.S Department of Agriculture: fdc.nal.usda.gov/download-datasets

## Housing Project

Regression task, from the book "Hands on Machine Learning", to predict the median house price values of thousands of districts from the state of California, USA 1990.

The dataset was acquired from the author of the book: raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz

---

## Setup Instructions

Download, install dependencies, and run the project:

### 1. Prerequisites
- Install Python: [Download Python](https://www.python.org/downloads/).

### 2. Clone the Repository
```bash
git clone https://github.com/weslleyskah/machine_learning_projects.git
```

### 3. Navigate to the Repository
```bash
cd machine_learning_projects
```

### 4. Create and Activate a Virtual Environment
#### On Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 5. Install Required Python Packages
```bash
python -m pip install matplotlib numpy pandas scipy scikit-learn joblib
```

### 6. Verify Installation
```bash
python -c "import matplotlib, numpy, pandas, scipy, sklearn, joblib"
```

### 7. Navigate to the Project Code
```bash
cd food_project/code
```

### 8. Run the Python File
```bash
python food_project.py
```

This will process the dataset, generate comparisons between ML algorithms, and produce outputs such as reshaped dataframes, RMSE values, and predictions. It will also create the following folders:
- `datasets` – Contains dataset files.
- `models` – Stores trained machine learning models.
- `venv` – Virtual environment for dependency management.

### 9. Deactivate the Virtual Environment
```bash
cd ../..
.\venv\Scripts\deactivate
```

---

## Repository Structure

The repository is organized as follows:

```
datasets/          Contains source and reshaped .csv files from the datasets.
food_project/      Food dataset analysis using U.S. Department of Agriculture data.
housing_project/   Housing project from "Hands-On Machine Learning" by Aurélien Géron.
img/               Visualization images generated using Matplotlib.
models/            Trained machine learning models.
venv/              Python virtual environment for dependency management.
```

---
