# -*- coding: utf-8 -*-
"""
Configuration de l'application Streamlit - Centrale Hydrogène
"""

import os
from pathlib import Path

# Chemins de base
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR
IMAGES_DIR = BASE_DIR.parent / "images"

# Paramètres du système
SYSTEM_PARAMS = {
    "duration": 8760,  # Heures par an
    "grid_limit": 1319414,  # Limite réseau [kW]
    "LHV_kg": 33.3,  # kWh/kg
    "LHV_NM3": 3,  # kWh/Nm3
    "truck_capacity": 29.36,  # Capacité camion [m³]
    "unavailable_hours": 3,  # Heures d'indisponibilité par trajet
    "H2_price": 2.7,  # Prix hydrogène [€/kg]
    "eta_F_characteristic": 0.04409448818,  # Caractéristique électrolyseur
}

# Paramètres du modèle de gaz
GAS_MODEL = {
    "T_op": 80,  # Température opération [°C]
    "P_op": 15,  # Pression opération [bar]
    "M": 2e-3,  # Masse molaire H2 [kg/mol]
    "R": 8.314,  # Constante gaz parfait
    "T_c": -240,  # Température critique [°C]
    "P_c": 130,  # Pression critique [bar]
    "n": 1.4,  # cp/cv
    "P_out": 250,  # Pression sortie [bar]
}

# Paramètres économiques
ECONOMICS = {
    "install_fee": 1800,  # CAPEX électrolyseur [€/kW]
    "OPEX_PEM": 54,  # OPEX PEM [€/kW/an]
    "water_price": 3e-3,  # Prix eau [€/kg]
    "water_consumption": 9,  # Consommation eau [kg H2O/kg H2]
    "price_H2": 2.7,  # Prix H2 [€/kg]
    "storage_capex": 490,  # CAPEX stockage [€/kg]
    "truck_capex": 610000,  # CAPEX camion [€]
    "truck_opex": 30500,  # OPEX camion [€/an]
    "compressor_opex": 4665,  # OPEX compresseur [€/an]
    "selling_capex_fixed": 93296,  # CAPEX vente fixe [€]
    "wacc": 0.05,  # WACC
    "lifetime": 30,  # Durée de vie [ans]
    "degradation": 0.003,  # Dégradation annuelle
}

# Paramètres de l'algorithme génétique (défaut)
GA_DEFAULT_PARAMS = {
    "population_size": 20,
    "n_generations": 20,
    "crossover_prob": 0.5,
    "mutation_prob": 0.1,
    "n_improvements": 4,
    "n_best_children": 5,
    "elite_ratio": 0.05,
}

# Bornes des variables d'optimisation
OPTIMIZATION_BOUNDS = {
    "C_min": 100,  # Capacité min électrolyseur [kW]
    "C_max": 100000,  # Capacité max électrolyseur [kW]
    "S_min": 10,  # Stockage min [m³]
    "S_max": 10000,  # Stockage max [m³]
    "N_min": 1,  # Nombre min camions
    "N_max": 50,  # Nombre max camions
    "T_min": 0.1,  # Seuil min
    "T_max": 0.95,  # Seuil max
}

# Design optimal trouvé
OPTIMAL_DESIGN = {
    "electrolyzer_capacity": 49161,  # [kW]
    "storage_capacity": 326,  # [m³]
    "number_of_trucks": 11,
    "threshold": 0.65,
    "LCOH": 0.165,  # [€/kWh]
    "H2_annual": 2978162,  # [kg/an]
    "wasted_power": 0.692,
    "wasted_hydrogen": 0.02,
}

# Palette de couleurs
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#2ca02c",
    "accent": "#ff7f0e",
    "danger": "#d62728",
    "info": "#17becf",
    "dark": "#2c3e50",
    "light": "#ecf0f1",
    "hydrogen": "#00b4d8",
    "nuclear": "#e76f51",
    "wind": "#52b788",
    "grid": "#6c757d",
}

# Configuration des graphiques Plotly
PLOTLY_CONFIG = {
    "displayModeBar": True,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

PLOTLY_LAYOUT = {
    "template": "plotly_white",
    "font": {"family": "Arial, sans-serif", "size": 12},
    "hoverlabel": {"font_size": 12},
    "margin": {"l": 60, "r": 30, "t": 50, "b": 60},
}
