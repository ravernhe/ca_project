# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import List
import pandas as pd

# Compatibilité PyQt6
from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
        QListWidget, QFileDialog, QMessageBox, QFormLayout, QLineEdit, QLabel,
        QGroupBox, QGridLayout
    )
from PyQt6.QtCore import Qt

from .parsers import parse_file_with_origin
from .pipeline import (
    pipeline_concat, compute_facturation_from_external,
    compute_ca_attendu, normalize_to_target, compute_dates_repere,
    compute_progress_and_backlog, inject_excel_error_for_ss_missing_exec
)
from .helpers import export_excel


class DropList(QListWidget):
    """Zone de dépôt pour un type d'origine donné."""
    def __init__(self, origin_label: str, parent=None):
        super().__init__(parent)
        self.origin_label = origin_label
        self.setAcceptDrops(True)
        self.setMinimumHeight(120)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.exists() and p.suffix.lower() in (".xlsx", ".xls", ".csv"):
                self.addItem(str(p))

    def files(self) -> List[Path]:
        return [Path(self.item(i).text()) for i in range(self.count())]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CA Mensuel - Pipeline")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        root.addWidget(QLabel("1) Déposez / ajoutez vos fichiers par type"))

        # --- Grille des listes + boutons ---
        grid_box = QGroupBox("Fichiers par type")
        grid = QGridLayout(grid_box)

        # Listes
        self.list_intra        = DropList("intra")
        self.list_sans_session = DropList("sans_session")
        self.list_inter        = DropList("inter")
        self.list_fact_eur     = DropList("fact_eur")
        self.list_fact_hkd     = DropList("fact_hkd")

        # Titre des zones
        grid.addWidget(QLabel("INTRA"), 0, 0)
        grid.addWidget(QLabel("Sans session"), 0, 1)
        grid.addWidget(QLabel("INTER"), 0, 2)
        grid.addWidget(QLabel("Facturation EUR"), 2, 0)
        grid.addWidget(QLabel("Facturation HKD"), 2, 1)

        # List widgets
        grid.addWidget(self.list_intra,        1, 0)
        grid.addWidget(self.list_sans_session, 1, 1)
        grid.addWidget(self.list_inter,        1, 2)
        grid.addWidget(self.list_fact_eur,     3, 0)
        grid.addWidget(self.list_fact_hkd,     3, 1)

        # Boutons Add… + Clear pour chaque liste
        def make_buttons(target_list: DropList):
            row = QHBoxLayout()
            btn_add = QPushButton("Add…")
            btn_clear = QPushButton("Clear")
            # Add…
            def choose():
                files, _ = QFileDialog.getOpenFileNames(
                    self, "Sélectionner fichiers", str(Path.cwd()),
                    "Excel/CSV (*.xlsx *.xls *.csv);;Tous (*.*)"
                )
                for f in files:
                    target_list.addItem(f)
            btn_add.clicked.connect(choose)
            # Clear
            btn_clear.clicked.connect(target_list.clear)
            row.addStretch(1)
            row.addWidget(btn_add)
            row.addWidget(btn_clear)
            return row

        grid.addLayout(make_buttons(self.list_intra),        1, 0, alignment=Qt.AlignmentFlag.AlignBottom if hasattr(Qt, "AlignmentFlag") else Qt.AlignBottom)
        grid.addLayout(make_buttons(self.list_sans_session), 1, 1, alignment=Qt.AlignmentFlag.AlignBottom if hasattr(Qt, "AlignmentFlag") else Qt.AlignBottom)
        grid.addLayout(make_buttons(self.list_inter),        1, 2, alignment=Qt.AlignmentFlag.AlignBottom if hasattr(Qt, "AlignmentFlag") else Qt.AlignBottom)
        grid.addLayout(make_buttons(self.list_fact_eur),     3, 0, alignment=Qt.AlignmentFlag.AlignBottom if hasattr(Qt, "AlignmentFlag") else Qt.AlignBottom)
        grid.addLayout(make_buttons(self.list_fact_hkd),     3, 1, alignment=Qt.AlignmentFlag.AlignBottom if hasattr(Qt, "AlignmentFlag") else Qt.AlignBottom)

        root.addWidget(grid_box)

        # --- Paramètres ---
        root.addWidget(QLabel("2) Paramètres"))
        form = QFormLayout()
        self.in_mois = QLineEdit()
        self.in_annee = QLineEdit()
        self.in_date_clot = QLineEdit(); self.in_date_clot.setPlaceholderText("YYYY-MM-DD")
        self.in_debut_ex  = QLineEdit(); self.in_debut_ex.setPlaceholderText("YYYY-MM-DD")
        self.in_fin_ex    = QLineEdit(); self.in_fin_ex.setPlaceholderText("YYYY-MM-DD")
        self.in_hkd_rate  = QLineEdit(); self.in_hkd_rate.setPlaceholderText("e.g. 9")

        form.addRow("Mois de clôture", self.in_mois)
        form.addRow("Année de clôture", self.in_annee)
        form.addRow("Date de clôture", self.in_date_clot)
        form.addRow("Début d'exercice", self.in_debut_ex)
        form.addRow("Fin d'exercice", self.in_fin_ex)
        form.addRow("Taux HKD → EUR", self.in_hkd_rate)
        root.addLayout(form)

        # --- Output ---
        out_row = QHBoxLayout()
        self.out_path = QLineEdit(str(Path.cwd() / "CA_consolide.xlsx"))
        btn_browse_out = QPushButton("Save as…")
        btn_browse_out.clicked.connect(self.choose_out)
        out_row.addWidget(QLabel("Output (.xlsx)"))
        out_row.addWidget(self.out_path, 1)
        out_row.addWidget(btn_browse_out)
        root.addLayout(out_row)

        # --- Run ---
        run = QPushButton("Run")
        run.clicked.connect(self.run_pipeline)
        root.addWidget(run)

        self.resize(1100, 740)

    def choose_out(self):
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer sous", str(self.out_path.text()), "Excel (*.xlsx)")
        if path:
            if not path.lower().endswith(".xlsx"):
                path += ".xlsx"
            self.out_path.setText(path)

    def run_pipeline(self):
        try:
            data_frames, fact_eur_frames, fact_hkd_frames = [], [], []

            buckets = [
                (self.list_intra,        "intra"),
                (self.list_sans_session, "sans_session"),
                (self.list_inter,        "inter"),
                (self.list_fact_eur,     "fact_eur"),
                (self.list_fact_hkd,     "fact_hkd"),
            ]

            for lst, origin in buckets:
                for p in lst.files():
                    if origin in ("fact_eur", "fact_hkd"):
                        fact_df = parse_file_with_origin(p, origin)  # non concaténé au main
                        (fact_eur_frames if origin == "fact_eur" else fact_hkd_frames).append(fact_df)
                    else:
                        data_frames.append(parse_file_with_origin(p, origin))

            if not data_frames:
                raise RuntimeError("Ajoutez au moins un fichier INTRA / SANS_SESSION / INTER.")

            df = pipeline_concat(data_frames)

            fact_eur = pipeline_concat(fact_eur_frames) if fact_eur_frames else pd.DataFrame()
            fact_hkd = pipeline_concat(fact_hkd_frames) if fact_hkd_frames else pd.DataFrame()

            # Y, Y-1, Y-2 + total (and get closure year hint from facts)
            df, closure_year = compute_facturation_from_external(df, fact_eur, fact_hkd, self.in_hkd_rate.text())

            # CA attendu
            df = compute_ca_attendu(df)

            # Dates repère (execution date override)
            df = compute_dates_repere(df)

            # Progress / backlog / FAE / PCA
            df = compute_progress_and_backlog(
                df,
                date_cloture=self.in_date_clot.text(),
                debut_exercice=self.in_debut_ex.text(),
                fin_exercice=self.in_fin_ex.text(),
                closure_year_hint=closure_year
            )

            # Export ordering
            df = normalize_to_target(df)

            # Genere error
            df = inject_excel_error_for_ss_missing_exec(df)
            meta = {
                "Mois de clôture": self.in_mois.text(),
                "Année de clôture": self.in_annee.text(),
                "Date de clôture": self.in_date_clot.text(),
                "Début d'exercice": self.in_debut_ex.text(),
                "Fin d'exercice": self.in_fin_ex.text(),
                "Taux HKD->EUR": self.in_hkd_rate.text(),
            }
            export_excel(df, Path(self.out_path.text()), meta)

            QMessageBox.information(self, "Terminé", f"Consolidation OK :\n{self.out_path.text()}")

        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))
