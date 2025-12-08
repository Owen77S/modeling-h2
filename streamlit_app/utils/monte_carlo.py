# -*- coding: utf-8 -*-
"""
Moteur d'analyse Monte Carlo pour quantification d'incertitude
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial
import sys
from pathlib import Path

# Ajouter le chemin parent pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.distributions import ParameterDistribution, latin_hypercube_sampling, get_default_distributions
from utils.model import H2PlantModel
from utils.data_loader import load_power_data


@dataclass
class MonteCarloResults:
    """
    Classe contenant les résultats d'une analyse Monte Carlo.

    Attributes:
        n_samples: Nombre d'échantillons simulés
        parameter_samples: DataFrame avec les valeurs échantillonnées de chaque paramètre
        kpi_results: DataFrame avec les KPIs calculés pour chaque échantillon
        base_case: Dictionnaire des valeurs nominales des paramètres
        base_case_kpis: Dictionnaire des KPIs pour le cas nominal
        correlations_pearson: Matrice de corrélation de Pearson
        correlations_spearman: Matrice de corrélation de Spearman
        statistics: Dictionnaire des statistiques (mean, std, percentiles, etc.)
    """
    n_samples: int
    parameter_samples: pd.DataFrame
    kpi_results: pd.DataFrame
    base_case: Dict[str, float]
    base_case_kpis: Dict[str, float]
    correlations_pearson: pd.DataFrame
    correlations_spearman: pd.DataFrame
    statistics: Dict[str, Dict]


class MonteCarloAnalyzer:
    """
    Analyseur Monte Carlo pour quantifier l'impact des incertitudes
    sur les performances de la centrale H2.
    """

    def __init__(self,
                 design_config: Dict[str, float],
                 distributions: Optional[Dict[str, ParameterDistribution]] = None,
                 power_data: Optional[pd.DataFrame] = None):
        """
        Initialise l'analyseur Monte Carlo.

        Args:
            design_config: Configuration de design {'C': capacité, 'S': stockage, 'N': camions, 'T': threshold}
            distributions: Distributions des paramètres (si None, utilise les valeurs par défaut)
            power_data: Données de puissance (WP, NP). Si None, charge depuis load_power_data()
        """
        self.design_config = design_config
        self.distributions = distributions or get_default_distributions()

        # Charger les données de puissance
        if power_data is None:
            df = load_power_data(104)
            self.wp_data = df['WP'].values
            self.np_data = df['NP'].values
        else:
            self.wp_data = power_data['WP'].values
            self.np_data = power_data['NP'].values

    def run_single_simulation(self,
                             sample_params: Dict[str, float],
                             design_config: Dict[str, float]) -> Dict[str, float]:
        """
        Exécute une simulation unique avec un jeu de paramètres donné.

        Args:
            sample_params: Dictionnaire des valeurs de paramètres échantillonnés
            design_config: Configuration de design (C, S, N, T)

        Returns:
            Dictionnaire des KPIs calculés
        """
        try:
            # Créer une instance de la centrale
            plant = H2PlantModel()

            # Appliquer les paramètres économiques échantillonnés
            if 'install_fee' in sample_params:
                plant.install_fee = sample_params['install_fee']
            if 'OPEX_PEM' in sample_params:
                plant.OPEX_PEM = sample_params['OPEX_PEM']
            if 'price_H2' in sample_params:
                plant.H2_price = sample_params['price_H2']
            if 'water_price' in sample_params:
                plant.water_price = sample_params['water_price']
            if 'storage_capex' in sample_params:
                plant.storage_capex = sample_params['storage_capex']
            if 'truck_capex' in sample_params:
                plant.truck_capex = sample_params['truck_capex']

            # Appliquer les paramètres techniques échantillonnés
            if 'eta_F_characteristic' in sample_params:
                plant.eta_F_characteristic = sample_params['eta_F_characteristic']
            if 'grid_limit' in sample_params:
                plant.grid_limit = sample_params['grid_limit']

            # Charger les données
            plant.load_data(self.wp_data, self.np_data)

            # Évaluer la configuration
            kpis = plant.evaluate(
                C=design_config['C'],
                S=design_config['S'],
                N=design_config['N'],
                T=design_config['T']
            )

            return kpis

        except Exception as e:
            print(f"Erreur dans la simulation: {e}")
            # Retourner des valeurs par défaut en cas d'erreur
            return {
                'LCOH': np.nan,
                'H2': 0,
                'H2_waste': 0,
                'power_waste': 0,
                'CAPEX_total': 0,
                'OPEX_total': 0
            }

    def run_parallel_simulation(self,
                               sample_idx: int,
                               param_samples: Dict[str, np.ndarray],
                               design_config: Dict[str, float]) -> Dict[str, float]:
        """
        Wrapper pour exécution parallèle d'une simulation.

        Args:
            sample_idx: Index de l'échantillon
            param_samples: Dictionnaire des arrays d'échantillons de paramètres
            design_config: Configuration de design

        Returns:
            Dictionnaire des KPIs avec l'index ajouté
        """
        # Extraire les valeurs pour cet échantillon
        sample_params = {key: values[sample_idx] for key, values in param_samples.items()}

        # Exécuter la simulation
        kpis = self.run_single_simulation(sample_params, design_config)
        kpis['sample_idx'] = sample_idx

        return kpis

    def run_monte_carlo(self,
                       n_samples: int = 1000,
                       sampling_method: str = 'lhs',
                       n_processes: Optional[int] = None,
                       seed: Optional[int] = 42) -> MonteCarloResults:
        """
        Exécute l'analyse Monte Carlo complète.

        Args:
            n_samples: Nombre d'échantillons à simuler
            sampling_method: Méthode d'échantillonnage ('lhs' ou 'random')
            n_processes: Nombre de processus parallèles (None = auto)
            seed: Graine aléatoire pour reproductibilité

        Returns:
            MonteCarloResults avec tous les résultats
        """
        print(f"Démarrage analyse Monte Carlo avec {n_samples} échantillons...")

        # 1. Échantillonnage des paramètres
        if sampling_method == 'lhs':
            param_samples = latin_hypercube_sampling(self.distributions, n_samples, seed)
        else:
            param_samples = {}
            for param_name, dist in self.distributions.items():
                param_samples[param_name] = dist.sample(n_samples, seed)

        # Créer DataFrame des échantillons
        param_df = pd.DataFrame(param_samples)

        # 2. Cas de base (nominal)
        base_case = {param: dist.nominal for param, dist in self.distributions.items()}
        base_case_kpis = self.run_single_simulation(base_case, self.design_config)

        # 3. Simulations Monte Carlo
        print("Exécution des simulations...")

        # Déterminer le nombre de processus
        if n_processes is None:
            n_processes = max(1, mp.cpu_count() - 1)

        # Fonction partielle avec les paramètres fixes
        simulation_func = partial(
            self.run_parallel_simulation,
            param_samples=param_samples,
            design_config=self.design_config
        )

        # Exécution parallèle
        if n_processes > 1:
            with mp.Pool(processes=n_processes) as pool:
                results = pool.map(simulation_func, range(n_samples))
        else:
            # Exécution séquentielle (utile pour le débogage)
            results = [simulation_func(i) for i in range(n_samples)]

        # Créer DataFrame des résultats
        kpi_df = pd.DataFrame(results)
        kpi_df = kpi_df.set_index('sample_idx')

        # 4. Calcul des corrélations
        print("Calcul des corrélations...")
        combined_df = pd.concat([param_df, kpi_df], axis=1)

        # Corrélation de Pearson (linéaire)
        corr_pearson = combined_df.corr(method='pearson')

        # Corrélation de Spearman (monotone)
        corr_spearman = combined_df.corr(method='spearman')

        # 5. Calcul des statistiques
        print("Calcul des statistiques...")
        statistics = self._compute_statistics(kpi_df, base_case_kpis)

        # 6. Créer l'objet résultat
        results = MonteCarloResults(
            n_samples=n_samples,
            parameter_samples=param_df,
            kpi_results=kpi_df,
            base_case=base_case,
            base_case_kpis=base_case_kpis,
            correlations_pearson=corr_pearson,
            correlations_spearman=corr_spearman,
            statistics=statistics
        )

        print("Analyse Monte Carlo terminée!")
        return results

    def _compute_statistics(self,
                           kpi_df: pd.DataFrame,
                           base_case_kpis: Dict[str, float]) -> Dict[str, Dict]:
        """
        Calcule les statistiques descriptives pour chaque KPI.

        Args:
            kpi_df: DataFrame des KPIs
            base_case_kpis: KPIs du cas nominal

        Returns:
            Dictionnaire {kpi_name: {statistiques}}
        """
        statistics = {}

        for kpi_name in kpi_df.columns:
            values = kpi_df[kpi_name].dropna()

            if len(values) == 0:
                continue

            stats = {
                'mean': float(values.mean()),
                'median': float(values.median()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'p5': float(values.quantile(0.05)),
                'p10': float(values.quantile(0.10)),
                'p25': float(values.quantile(0.25)),
                'p75': float(values.quantile(0.75)),
                'p90': float(values.quantile(0.90)),
                'p95': float(values.quantile(0.95)),
                'base_case': base_case_kpis.get(kpi_name, np.nan),
                'cv': float(values.std() / values.mean()) if values.mean() != 0 else np.nan  # Coefficient de variation
            }

            statistics[kpi_name] = stats

        return statistics

    def compute_tornado_data(self,
                            kpi_name: str = 'LCOH',
                            variation_pct: float = 0.20) -> pd.DataFrame:
        """
        Calcule les données pour un diagramme tornado (analyse de sensibilité).

        Pour chaque paramètre, évalue l'impact d'une variation de ±variation_pct
        sur le KPI cible.

        Args:
            kpi_name: Nom du KPI à analyser (défaut: 'LCOH')
            variation_pct: Pourcentage de variation (défaut: 20%)

        Returns:
            DataFrame avec les impacts de chaque paramètre
        """
        tornado_data = []

        base_case = {param: dist.nominal for param, dist in self.distributions.items()}
        base_kpi = self.run_single_simulation(base_case, self.design_config)[kpi_name]

        for param_name, dist in self.distributions.items():
            # Variation basse
            params_low = base_case.copy()
            params_low[param_name] = dist.nominal * (1 - variation_pct)
            kpi_low = self.run_single_simulation(params_low, self.design_config)[kpi_name]

            # Variation haute
            params_high = base_case.copy()
            params_high[param_name] = dist.nominal * (1 + variation_pct)
            kpi_high = self.run_single_simulation(params_high, self.design_config)[kpi_name]

            # Calculer l'impact
            impact_low = abs(kpi_low - base_kpi)
            impact_high = abs(kpi_high - base_kpi)
            impact_total = impact_low + impact_high

            tornado_data.append({
                'parameter': param_name,
                'base_value': dist.nominal,
                'kpi_base': base_kpi,
                'kpi_low': kpi_low,
                'kpi_high': kpi_high,
                'impact_low': impact_low,
                'impact_high': impact_high,
                'impact_total': impact_total,
                'impact_pct': (impact_total / base_kpi) * 100 if base_kpi != 0 else 0
            })

        # Trier par impact total décroissant
        tornado_df = pd.DataFrame(tornado_data)
        tornado_df = tornado_df.sort_values('impact_total', ascending=False)

        return tornado_df

    def export_results(self, results: MonteCarloResults, output_dir: str = '.'):
        """
        Exporte les résultats de l'analyse Monte Carlo vers des fichiers CSV.

        Args:
            results: Résultats Monte Carlo
            output_dir: Répertoire de sortie
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Exporter les échantillons de paramètres
        results.parameter_samples.to_csv(output_path / 'mc_parameter_samples.csv')

        # Exporter les résultats KPI
        results.kpi_results.to_csv(output_path / 'mc_kpi_results.csv')

        # Exporter les corrélations
        results.correlations_pearson.to_csv(output_path / 'mc_correlations_pearson.csv')
        results.correlations_spearman.to_csv(output_path / 'mc_correlations_spearman.csv')

        # Exporter les statistiques
        stats_df = pd.DataFrame(results.statistics).T
        stats_df.to_csv(output_path / 'mc_statistics.csv')

        print(f"Résultats exportés dans {output_path}")
