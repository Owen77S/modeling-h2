# -*- coding: utf-8 -*-
"""
Modèle de la centrale hydrogène - Version optimisée pour Streamlit
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
import streamlit as st


@dataclass
class H2PlantModel:
    """
    Modèle de centrale de production d'hydrogène.
    Version optimisée avec numpy pour performance.
    """

    # Paramètres système
    duration: int = 8760
    grid_limit: float = 1319414  # kW

    # Paramètres électrolyseur
    eta_F_characteristic: float = 0.04409448818
    aux_loss: float = 0.03  # 3% pertes auxiliaires

    # Paramètres gaz
    T_op: float = 80  # °C
    P_op: float = 15  # bar
    P_out: float = 250  # bar
    M: float = 2e-3  # kg/mol
    R: float = 8.314  # J/(mol·K)
    n: float = 1.4  # cp/cv

    # Paramètres économiques
    LHV_kg: float = 33.3  # kWh/kg
    truck_capacity: float = 29.36  # m³
    unavailable_hours: int = 3
    H2_price: float = 2.7  # €/kg

    # CAPEX
    install_fee: float = 1800  # €/kW
    storage_capex: float = 490  # €/kg
    truck_capex: float = 610000  # €
    selling_capex_fixed: float = 93296  # €

    # OPEX
    OPEX_PEM: float = 54  # €/kW/an
    water_price: float = 3e-3  # €/L
    water_consumption: float = 9  # L/kg H2
    compressor_opex: float = 4665  # €/an
    truck_opex: float = 30500  # €/an

    # Données
    WP: np.ndarray = field(default_factory=lambda: np.zeros(8760))
    NP: np.ndarray = field(default_factory=lambda: np.zeros(8760))

    def __post_init__(self):
        """Initialise les tableaux de résultats."""
        self._init_arrays()

    def _init_arrays(self):
        """Initialise les tableaux numpy pour les calculs."""
        n = self.duration
        self.excess_power = np.zeros(n)
        self.supply_power = np.zeros(n)
        self.mass_H2 = np.zeros(n)
        self.H2 = np.zeros(n)
        self.H2_compressed = np.zeros(n)
        self.stored = np.zeros(n)
        self.wasted = np.zeros(n)
        self.sold = np.zeros(n)

    def load_data(self, WP: np.ndarray, NP: np.ndarray):
        """Charge les données de puissance."""
        self.WP = np.array(WP)
        self.NP = np.array(NP)

    def compute_excess_power(self):
        """Calcule la puissance excédentaire (vectorisé)."""
        total = self.WP + self.NP
        self.excess_power = np.maximum(0, total - self.grid_limit)

    def faraday_efficiency(self, capacity: float, supply: np.ndarray) -> np.ndarray:
        """
        Calcule l'efficacité faradique (vectorisé).

        η_F = 1 - exp(-(P_supply / C) / 0.04409)
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = supply / capacity
            eta = 1 - np.exp(-ratio / self.eta_F_characteristic)
            eta = np.where(supply > 0, eta, 0)
        return eta

    def compute_electrolyzer_production(self, capacity: float):
        """
        Calcule la production horaire d'hydrogène (vectorisé).

        Paramètres:
            capacity: Capacité de l'électrolyseur [kW]
        """
        # Puissance fournie à l'électrolyseur (limitée par la capacité)
        self.supply_power = np.minimum(self.excess_power, capacity)

        # Efficacité faradique
        eta_F = self.faraday_efficiency(capacity, self.supply_power)

        # Pertes auxiliaires
        n_aux = 1 - self.aux_loss

        # Production de masse H2 [kg]
        self.mass_H2 = n_aux * eta_F * self.supply_power / self.LHV_kg

        # Volume H2 (loi gaz parfait) [m³]
        T_K = self.T_op + 273.15
        self.H2 = self.mass_H2 * self.R * T_K / (self.M * self.P_op * 1e5)

        # Volume après compression [m³]
        self.H2_compressed = self.H2 * (self.P_op / self.P_out) ** (1 / self.n)

    def compute_hydrogen_management(self, storage_capacity: float,
                                    n_trucks: int, threshold: float):
        """
        Gère le stockage et la vente d'hydrogène.

        Paramètres:
            storage_capacity: Capacité de stockage [m³]
            n_trucks: Nombre de camions
            threshold: Seuil de vente (0-1)
        """
        # Réinitialiser
        self.stored = np.zeros(self.duration)
        self.wasted = np.zeros(self.duration)
        self.sold = np.zeros(self.duration)

        # État des camions
        trucks_available = n_trucks
        unavailabilities = []  # [(nb_trucks, remaining_hours), ...]

        # Première heure
        self.stored[0] = min(self.H2_compressed[0], storage_capacity)

        for t in range(1, self.duration):
            # Disponibilité dans le stockage
            available_space = storage_capacity - self.stored[t - 1]
            h2_produced = self.H2_compressed[t]

            # Stockage
            if h2_produced > available_space:
                self.wasted[t] = h2_produced - available_space
                stored_now = available_space
            else:
                stored_now = h2_produced

            self.stored[t] = self.stored[t - 1] + stored_now

            # Vente si seuil atteint et camions disponibles
            if self.stored[t] >= storage_capacity * threshold and trucks_available > 0:
                # Nombre de camions à utiliser
                n_trucks_used = min(
                    int(self.stored[t] // self.truck_capacity),
                    trucks_available
                )
                if n_trucks_used > 0:
                    quantity_sold = n_trucks_used * self.truck_capacity
                    trucks_available -= n_trucks_used
                    self.sold[t] = quantity_sold
                    self.stored[t] -= quantity_sold
                    unavailabilities.append([n_trucks_used, self.unavailable_hours])

            # Gestion retour des camions
            available_next = 0
            for i in range(len(unavailabilities)):
                unavailabilities[i][1] -= 1
                if unavailabilities[i][1] == 0:
                    available_next = unavailabilities[i][0]

            if available_next > 0:
                trucks_available += available_next
                unavailabilities = [u for u in unavailabilities if u[1] > 0]

    def compute_KPIs(self, capacity: float, storage_capacity: float,
                    n_trucks: int) -> Dict[str, float]:
        """
        Calcule les KPIs du système.

        Returns:
            Dictionnaire avec LCOH, H2 produit, pertes, etc.
        """
        # Production totale H2
        H2_total_kg = np.sum(self.mass_H2)

        # H2 vendu (conversion m³ → kg)
        T_K = self.T_op + 273.15
        H2_sold_kg = (self.P_out * 1e5 * np.sum(self.sold) * self.M) / (self.R * T_K)

        # Stockage en kg
        storage_kg = (self.P_out * 1e5 * storage_capacity * self.M) / (self.R * T_K)

        # CAPEX
        CAPEX_PEM = self.install_fee * capacity
        CAPEX_storage = self.storage_capex * storage_kg
        CAPEX_selling = self.selling_capex_fixed + n_trucks * self.truck_capex
        CAPEX_total = CAPEX_PEM + CAPEX_storage + CAPEX_selling

        # OPEX annuel
        OPEX_PEM = self.OPEX_PEM * capacity
        OPEX_water = self.water_price * self.water_consumption * H2_total_kg
        OPEX_selling = self.truck_opex * n_trucks
        OPEX_total = OPEX_PEM + OPEX_water + self.compressor_opex + OPEX_selling

        # Énergie dans H2 vendu
        energy_H2 = H2_sold_kg * self.LHV_kg  # kWh

        # LCOH
        if energy_H2 > 0:
            LCOH = (CAPEX_total + OPEX_total) / energy_H2
        else:
            LCOH = float('inf')

        # Autres KPIs
        total_excess = np.sum(self.excess_power)
        total_supply = np.sum(self.supply_power)
        wasted_power = 1 - (total_supply / total_excess) if total_excess > 0 else 0

        total_H2_compressed = np.sum(self.H2_compressed)
        total_wasted = np.sum(self.wasted)
        wasted_hydrogen = total_wasted / total_H2_compressed if total_H2_compressed > 0 else 0

        time_full = np.sum(self.stored >= storage_capacity * 0.99) / self.duration

        return {
            'LCOH': LCOH,
            'LCOH_kg': LCOH * self.LHV_kg,  # €/kg
            'H2_produced': H2_total_kg,
            'H2_sold': H2_sold_kg,
            'wasted_power': wasted_power,
            'wasted_hydrogen': wasted_hydrogen,
            'time_storage_full': time_full,
            'CAPEX_total': CAPEX_total,
            'CAPEX_PEM': CAPEX_PEM,
            'CAPEX_storage': CAPEX_storage,
            'CAPEX_selling': CAPEX_selling,
            'OPEX_total': OPEX_total,
            'revenue': H2_sold_kg * self.H2_price,
            'energy_H2': energy_H2,
            'capacity_factor': total_supply / (capacity * self.duration) if capacity > 0 else 0,
        }

    def objective(self, C: float, S: float, N: int, T: float) -> float:
        """
        Fonction objectif pour l'optimisation: minimiser LCOH.

        Paramètres:
            C: Capacité électrolyseur [kW]
            S: Capacité stockage [m³]
            N: Nombre de camions
            T: Seuil de vente

        Returns:
            LCOH [€/kWh]
        """
        self.compute_excess_power()
        self.compute_electrolyzer_production(C)
        self.compute_hydrogen_management(S, int(N), T)
        kpis = self.compute_KPIs(C, S, int(N))
        return kpis['LCOH']

    def evaluate(self, C: float, S: float, N: int, T: float) -> Dict[str, float]:
        """
        Évalue complètement une configuration.

        Returns:
            Dictionnaire avec tous les KPIs
        """
        self.compute_excess_power()
        self.compute_electrolyzer_production(C)
        self.compute_hydrogen_management(S, int(N), T)
        return self.compute_KPIs(C, S, int(N))

    def check_constraints(self, max_wasted_power: float = 0.8,
                         max_wasted_hydrogen: float = 0.8) -> Tuple[bool, bool]:
        """
        Vérifie les contraintes.

        Returns:
            (contrainte_power_ok, contrainte_h2_ok)
        """
        total_excess = np.sum(self.excess_power)
        total_supply = np.sum(self.supply_power)
        wasted_power = 1 - (total_supply / total_excess) if total_excess > 0 else 0

        total_H2 = np.sum(self.H2_compressed)
        total_wasted = np.sum(self.wasted)
        wasted_h2 = total_wasted / total_H2 if total_H2 > 0 else 0

        return (wasted_power < max_wasted_power, wasted_h2 < max_wasted_hydrogen)

    def get_time_series(self) -> pd.DataFrame:
        """
        Retourne toutes les séries temporelles dans un DataFrame.
        """
        return pd.DataFrame({
            'hour': range(self.duration),
            'WP': self.WP,
            'NP': self.NP,
            'excess_power': self.excess_power,
            'supply_power': self.supply_power,
            'mass_H2': self.mass_H2,
            'H2_compressed': self.H2_compressed,
            'stored': self.stored,
            'wasted': self.wasted,
            'sold': self.sold,
        })


@st.cache_resource
def create_plant(WP: np.ndarray, NP: np.ndarray) -> H2PlantModel:
    """
    Crée une instance de centrale avec les données chargées.
    Mise en cache pour éviter de recréer l'objet.
    """
    plant = H2PlantModel()
    plant.load_data(WP, NP)
    return plant


def quick_evaluate(plant: H2PlantModel, C: float, S: float,
                   N: int, T: float) -> Dict[str, float]:
    """
    Évaluation rapide sans modifier l'état du plant original.
    """
    # Créer une copie légère
    temp_plant = H2PlantModel()
    temp_plant.WP = plant.WP
    temp_plant.NP = plant.NP
    return temp_plant.evaluate(C, S, N, T)
