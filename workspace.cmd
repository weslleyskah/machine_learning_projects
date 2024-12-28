:: Install and upgrade pip
pip --version
python -m pip install --upgrade pip

:: Create the virtual environment inside the ML directory
cd %ML%
python -m venv venv

:: Activate the environment
:: Every package will be installed in this environment while it is activated
cd %ML%
.\venv\Scripts\activate

set PATH=%ML%\venv\Scripts;%PATH%
where python
:: -> %ML%\venv\Scripts\python.exe

:: Install packages (env -> lib -> site-packages)
python -m pip install matplotlib numpy pandas scipy scikit-learn joblib
python -c "import matplotlib, numpy, pandas, scipy, sklearn, joblib"

:: Import packages inside a python file
:: On file.py inside VSCode: select the python interpreter from the virtual environment
:: Search >Python: Select Interpreter (C:\...\ML\venv\scripts\python.exe)

:: The virtual environmenet serves to install packages and run python files from there
:: The command "where python" lists the python.exe interpreters, the first one is being used

:: Quit the virtual environment
:: The terminal session returns to using the default python.exe from the system-wide installation
cd %ML%
.\venv\scripts\deactivate