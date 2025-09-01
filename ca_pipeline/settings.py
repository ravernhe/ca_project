# -*- coding: utf-8 -*-

DATE_OUTPUT_STYLE = "mdy_slash"  # export as m/d/yy (e.g., 6/14/23)

TARGET_COLUMNS = [
    "Origine rapport","Filiale First Finance",
    "Code analytique (cf session)","Code analytique (pdt)","Nom de l'opportunité","Session",
    "Date d'éxécution (produit sans session)","Date de début","Date de fin",
    "Prix Intra 1 standard (converti)","Prix total",
    "Axe (sessions)","Axe (produit d'opportunité)","Statut",
    "Facturation (convertie) N","Facturation (convertie) N-1",
    "Nombre de sessions prévisionnelles","Nombre de sessions réelles","Nombre de sessions annulées","Nombre de sessions facturées",
    "Quantité","Nbre de jours ou personnes par session",
    "Prix de vente (converti) Devise","Prix de vente (converti)",
    "Prix total (converti) Devise","Prix total (converti)",
    "Prix Intra 1 standard (converti) Devise","Dans les locaux de",
    "Date prévisionnelle de début de projet","Date de fin de projet",
    "Code analytique","StorageOpportuniteId","Nom du produit","Famille de produit",
    "Lien Vers le Produit d'opportunité","Type d'enregistrement","CA attendu",
    "Date de début repère", "Date de fin repère", "Facturation Y-2", "Facturation Y-1",
    "Facturation Y", "Facutration totale", "Avancement global", "Avancement EOY",
    "CA avancement", "CA YTD", "CA EOY (backlog)", "FAE", "PCA",
]

COLUMN_MAPPING = {
    "origine_rapport": "Origine rapport",
    "filiale first finance": "Filiale First Finance",
    "filiale": "Filiale First Finance",

    "code analytique": "Code analytique",
    "code analytique.1": "Code analytique.1",
    "code analytique (cf session)": "Code analytique (cf session)",
    "code analytique (pdt)": "Code analytique (pdt)",

    "storageopportuniteid": "StorageOpportuniteId",
    "opportunite": "Nom de l'opportunité",
    "nom de l'opportunité": "Nom de l'opportunité",
    "nom du produit": "Nom du produit",
    "famille de produit": "Famille de produit",
    "produit: titre du produit": "Nom du produit",

    "session": "Session",
    "type d'enregistrement": "Type d'enregistrement",
    "statut": "Statut",

    "date d'exécution (produit sans session)": "Date d'éxécution (produit sans session)",
    "date d'éxécution (produit sans session)": "Date d'éxécution (produit sans session)",
    "date de début": "Date de début",
    "date de fin": "Date de fin",
    "date prévisionnelle de début de projet": "Date prévisionnelle de début de projet",
    "date de fin de projet": "Date de fin de projet",

    "axe": "Axe",
    "axe (sessions)": "Axe (sessions)",
    "axe (produit d'opportunité)": "Axe (produit d'opportunité)",

    "nombre de sessions prévisionnelles": "Nombre de sessions prévisionnelles",
    "nombre de sessions réelles": "Nombre de sessions réelles",
    "nombre de sessions annulées": "Nombre de sessions annulées",
    "nombre de sessions facturées": "Nombre de sessions facturées",
    "quantité": "Quantité",
    "nbre de jours ou personnes par session": "Nbre de jours ou personnes par session",

    "prix de vente (converti) devise": "Prix de vente (converti) Devise",
    "prix de vente (converti)": "Prix de vente (converti)",
    "prix total (converti) devise": "Prix total (converti) Devise",
    "prix total (converti)": "Prix total (converti)",
    "prix intra 1 standard (converti) devise": "Prix Intra 1 standard (converti) Devise",
    "prix intra 1 standard (converti)": "Prix Intra 1 standard (converti)",
    "prix total devise": "Prix total (converti) Devise",
    "prix total": "Prix total",

    "ca session (converti) devise": "CA session (converti) Devise",
    "ca session (converti)": "CA session (converti)",

    "lien vers le produit d'opportunité": "Lien Vers le Produit d'opportunité",
    "propriétaire de l'opportunité: nom complet": "Propriétaire: Nom complet",
    "propriétaire: nom complet": "Propriétaire: Nom complet",
}

INTERMEDIATE_MONETARY_COLS = ["CA session (converti)"]

DATE_COLUMNS = [
    "Date d'éxécution (produit sans session)",
    "Date de début",
    "Date de fin",
    "Date prévisionnelle de début de projet",
    "Date de fin de projet",
]
