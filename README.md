# Complilation

## Set up venv and run program
```bash
python -m venv .venv

# On Mac and Linux
source .venv/bin/activate
# On Windows
.venv\Scripts\activate

pip install -U pip
pip install -r requirements.txt
python main.py
```

## Compilation
Dans le venv
```bash
pip install pyinstaller
pyinstaller --noconfirm --clean --onefile --name ca_project --hidden-import PyQt6.QtCore --hidden-import PyQt6.QtGui --hidden-import PyQt6.QtWidgets --collect-all PyQt6 --collect-all pandas --collect-all openpyxl main.py
```
