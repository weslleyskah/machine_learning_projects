# Download and Run the python files from the projects

```batch
:: Install python: python.org/downloads/

:: Clone the repository
git clone "https://github.com/weslleyskah/machine_learning_projects.git"

:: Navigate to the directory of the repository
cd machine_learning_projects

:: Create a python virtual environment
python -m venv venv

:: Activate the virtual environment
.\venv\scripts\activate

:: Install python packages
python -m pip install matplotlib numpy pandas scipy scikit-learn joblib

:: Check if they were installed correctly
python -c "import matplotlib, numpy, pandas, scipy, sklearn, joblib"

:: Navigate to the code directory of the project
cd food_project\code

:: Run a python file from the project
python food_data_2.py

:: The python files will generally print information about the dataset to make comparisons between dfferent ML agorithms: reshaped dataframes from the original datasets, numerical and textual columns, linear correlations, standard deviations, RMSE values, prediction and label columns, etc.
:: This will create the "datasets", "models" and "venv" folders inside the directory of the repository

:: Navigate back to the directory of the repository
cd machine_learning_projects

:: Deactivate the virtual environment and leave
.\venv\scripts\deactivate
```

# Repository

```     
datasets          Dataset source and reshaped .csv files
food_project      Food project using data from the U.S Department of Agriculture
housing_project   Housing project from the book "Hands on Machine Learning"
img               Images created with matplotlib for data analysis 
models            Trained ML models
venv              Virtual Environment
```