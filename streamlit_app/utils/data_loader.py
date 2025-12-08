# %%

# -*- coding: utf-8 -*-
"""
Module de chargement et traitement des données
"""

import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st


@st.cache_data
def load_power_data(nb_turbines: int = 104) -> pd.DataFrame:
    """
    Charge les données de production éolienne et nucléaire.

    Args:
        nb_turbines: Nombre d'éoliennes dans le parc

    Returns:
        DataFrame avec les colonnes WP, NP, et données dérivées
    """
    # Chemins possibles pour les données
    data_path = Path(__file__).parent.parent.parent / "data"

    if data_path is None:
        # Générer des données de démonstration
        st.warning("Fichier de données non trouvé. Utilisation de données de démonstration.")
        return generate_demo_data(nb_turbines)

    # Charger les données
    df = pd.read_csv(data_path / "data_2.csv", sep=";", decimal=",")

    df_final = pd.DataFrame()

    # Calculer la puissance éolienne totale
    df_final['WP'] = df['Wind power [kW]'] * nb_turbines

    # Puissance nucléaire
    df_final['NP'] = df['Nuclear power plant [kW]']

    # Ajouter l'index horaire
    df_final['hour'] = range(len(df))
    df_final['day'] = df_final['hour'] // 24
    df_final['month'] = (df_final['day'] // 30).clip(0, 11)

    # Calculer la conso
    df_final["demand"] = df["Demand [kWh]"]

    # Calculer les métriques dérivées
    grid_limit = 1319414  # kW
    df_final['total_power'] = df_final['WP'] + df_final['NP']
    df_final['excess_power'] = np.maximum(0, df_final['total_power'] - grid_limit)
    df_final['grid_limit'] = grid_limit

    return df_final

def generate_demo_data(nb_turbines: int = 104) -> pd.DataFrame:
    """
    Génère des données de démonstration pour le cas où les vraies données
    ne sont pas disponibles.
    """
    np.random.seed(42)
    hours = 8760

    # Simuler la production éolienne (pattern saisonnier + bruit)
    t = np.arange(hours)
    seasonal = 0.3 * np.sin(2 * np.pi * t / 8760 - np.pi/2) + 0.7
    daily = 0.1 * np.sin(2 * np.pi * t / 24)
    noise = np.random.weibull(2, hours) * 0.3
    wp_per_turbine = np.clip(3300 * seasonal * (1 + daily + noise), 0, 3300)

    # Simuler la production nucléaire (plus stable avec maintenance)
    np_base = 1450000 * 0.92  # Facteur de charge 92%
    maintenance = np.zeros(hours)
    # Simuler des arrêts de maintenance
    for _ in range(3):
        start = np.random.randint(0, hours - 500)
        maintenance[start:start + np.random.randint(200, 500)] = 1
    nuclear = np_base * (1 - maintenance * 0.8) + np.random.normal(0, 10000, hours)
    nuclear = np.clip(nuclear, 0, 1450000)

    df = pd.DataFrame({
        'WP_per_turbine': wp_per_turbine,
        'NP': nuclear,
        'WP': wp_per_turbine * nb_turbines,
        'hour': t,
        'day': t // 24,
        'month': (t // 730).clip(0, 11),
    })

    grid_limit = 1319414
    df['total_power'] = df['WP'] + df['NP']
    df['excess_power'] = np.maximum(0, df['total_power'] - grid_limit)
    df['grid_limit'] = grid_limit

    return df


@st.cache_data
def get_statistics(df: pd.DataFrame) -> dict:
    """
    Calcule les statistiques descriptives des données.

    Args:
        df: DataFrame avec les données de puissance

    Returns:
        Dictionnaire avec les statistiques
    """
    stats = {
        'wind': {
            'mean': df['WP'].mean(),
            'std': df['WP'].std(),
            'min': df['WP'].min(),
            'max': df['WP'].max(),
            'total_energy': df['WP'].sum() / 1000,  # MWh
            'capacity_factor': df['WP'].mean() / (df['WP'].max()) if df['WP'].max() > 0 else 0,
        },
        'nuclear': {
            'mean': df['NP'].mean(),
            'std': df['NP'].std(),
            'min': df['NP'].min(),
            'max': df['NP'].max(),
            'total_energy': df['NP'].sum() / 1000,  # MWh
            'capacity_factor': df['NP'].mean() / 1450000,
        },
        'excess': {
            'mean': df['excess_power'].mean(),
            'std': df['excess_power'].std(),
            'max': df['excess_power'].max(),
            'total_energy': df['excess_power'].sum() / 1000,  # MWh
            'hours_congestion': (df['excess_power'] > 0).sum(),
            'pct_congestion': (df['excess_power'] > 0).mean() * 100,
        },
        'demand': {
            'mean': df['demand'].mean(),
            'std': df['demand'].std(),
            'min': df['demand'].min(),
            'max': df['demand'].max(),
            'total_energy': df['demand'].sum() / 1000,  # MWh
        },
        'total': {
            'mean': df['total_power'].mean(),
            'max': df['total_power'].max(),
            'total_energy': df['total_power'].sum() / 1000,  # MWh
        }
    }

    return stats


def get_monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les statistiques mensuelles.
    """
    monthly = df.groupby('month').agg({
        'WP': ['mean', 'max', 'sum'],
        'NP': ['mean', 'max', 'sum'],
        'excess_power': ['mean', 'max', 'sum'],
        'total_power': ['mean', 'max'],
    }).round(2)

    monthly.columns = ['_'.join(col).strip() for col in monthly.columns]
    monthly.index = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin',
                     'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']

    return monthly


def get_hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le profil horaire moyen.
    """
    df_copy = df.copy()
    df_copy['hour_of_day'] = df_copy['hour'] % 24

    hourly = df_copy.groupby('hour_of_day').agg({
        'WP': 'mean',
        'NP': 'mean',
        'excess_power': 'mean',
        'total_power': 'mean',
    }).round(2)

    return hourly

# %%
