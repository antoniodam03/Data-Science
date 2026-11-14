import os
import pandas as pd
import numpy as np

FILE_PATH = "/Users/antonio/Desktop/Serie_Temporali/entsoe_data_2024_2025.csv"
OUTPUT_DIR = "/Users/antonio/Desktop/Serie_Temporali/output"
OUTPUT_PATH = f"{OUTPUT_DIR}/entsoe_data_processed.csv"
columns_of_interest = [
    "datetime",
    "country",
    "price",
    "solar",
    "wind_onshore",
    "wind_offshore",
    "hydro_total",
    "fossil_gas",
    "nuclear",
    "fossil_hard_coal",
    "fossil_brown_coal_lignite",
    "biomass",
    "waste",
]


def leggi_dataset(file_path=FILE_PATH):
    """Carica il dataset CSV e restituisce i dati come DataFrame."""
    dataset = pd.read_csv(file_path)
    # Conserva l'anno locale originale per riconoscere i periodi di discontinuità.
    dataset["anno_locale"] = dataset["datetime"].str[:4].astype(int)
    dataset["datetime"] = pd.to_datetime(dataset["datetime"], utc=True)
    return dataset


def seleziona_colonne(dataset):
    """Mantiene nel DataFrame solo le colonne di interesse."""
    colonne_da_eliminare = [
        colonna for colonna in dataset.columns
        if colonna not in columns_of_interest
    ]
    return dataset.drop(columns=colonne_da_eliminare)


def sostituisci_valori_mancanti(dataset):
    dataset = dataset[dataset["country"] != "IE_SEM"].copy()
    fonti = [
        c for c in columns_of_interest
        if c not in ("datetime", "country", "price") and c in dataset.columns
    ]

    # Nel 2025 i dati polacchi sono pubblicati nelle colonne *_actual_aggregated_,
    # mentre le colonne standard risultano a zero. Ricostruisce le fonti selezionate.
    colonne_polonia = {
        "solar": "_solar_actual_aggregated_",
        "wind_onshore": "_wind_onshore_actual_aggregated_",
        "fossil_gas": "_fossil_gas_actual_aggregated_",
        "fossil_hard_coal": "_fossil_hard_coal_actual_aggregated_",
        "fossil_brown_coal_lignite": "_fossil_brown_coal_lignite_actual_aggregated_",
        "biomass": "_biomass_actual_aggregated_",
    }
    polonia_2025 = dataset["country"].eq("PL") & dataset["anno_locale"].ge(2025)
    for colonna, sorgente in colonne_polonia.items():
        dataset.loc[polonia_2025, colonna] = dataset.loc[polonia_2025, sorgente]

    componenti_idro_polonia = [
        "_hydro_pumped_storage_actual_aggregated_",
        "_hydro_run_of_river_and_poundage_actual_aggregated_",
        "_hydro_water_reservoir_actual_aggregated_",
    ]
    dataset.loc[polonia_2025, "hydro_total"] = dataset.loc[
        polonia_2025, componenti_idro_polonia
    ].sum(axis=1, min_count=1)

    # In Svizzera idroelettrico e nucleare passano da valori positivi nel 2024
    # a zero per tutte le ore del periodo 2025: sono dati di generazione mancanti,
    # non produzione nulla. Li preserviamo come NaN invece di inventare zeri.
    svizzera_2025 = dataset["country"].eq("CH") & dataset["anno_locale"].ge(2025)
    dataset.loc[svizzera_2025, ["hydro_total", "nuclear"]] = np.nan

    # fonte mai prodotta in una zona -> 0
    for c in fonti:
        mai = dataset.groupby("country")[c].transform(lambda s: s.notna().sum() == 0)
        dataset.loc[mai, c] = 0
    dataset = dataset.sort_values(["country", "datetime"])
    # Interpola solo lacune brevi interne; lascia mancanti i periodi lunghi o ai bordi.
    cols = fonti + ["price"]
    dataset[cols] = dataset.groupby("country")[cols].transform(
        lambda s: s.interpolate(method="linear", limit=3, limit_area="inside")
    )
    return dataset.drop(columns="anno_locale")


def salva_dataset(dataset, file_path=OUTPUT_PATH):
    """Salva il dataset elaborato in formato CSV."""
    output_dir = file_path.rsplit("/", 1)[0]
    os.makedirs(output_dir, exist_ok=True)
    dataset.to_csv(file_path, index=False)


def stampa_info_dataset(dataset):
    """Stampa le informazioni del DataFrame."""
    dataset.info()
    valori_nulli = dataset.isna().sum()
    valori_nulli = valori_nulli[valori_nulli > 0]
    print("\n =============================================================================\n")
    print("\nValori nulli per colonna:")
    print("\n =============================================================================\n")

    print(valori_nulli.to_string())


if __name__ == "__main__":
    dati = leggi_dataset()
    stampa_info_dataset(dati)
    dati = sostituisci_valori_mancanti(dati)
    dati = seleziona_colonne(dati)
    salva_dataset(dati)
    stampa_info_dataset(dati)
