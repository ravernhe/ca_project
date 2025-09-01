# -*- coding: utf-8 -*-

import sys
from PyQt6.QtWidgets import QApplication

from ca_pipeline.gui import MainWindow

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
