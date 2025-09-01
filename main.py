# -*- coding: utf-8 -*-

import sys
try:
    from PyQt6.QtWidgets import QApplication
except Exception:
    from PyQt5.QtWidgets import QApplication

from ca_pipeline.gui import MainWindow

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
