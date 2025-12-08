# -*- coding: utf-8 -*-
"""
Définitions des distributions de probabilité pour l'analyse Monte Carlo
"""

import numpy as np
from typing import Dict, Tuple, Optional, Literal
from dataclasses import dataclass


@dataclass
class ParameterDistribution:
    """
    Classe représentant la distribution d'un paramètre incertain.

    Attributes:
        name: Nom du paramètre
        nominal: Valeur nominale (base case)
        dist_type: Type de distribution ('normal', 'uniform', 'triangular', 'lognormal')
        params: Paramètres de la distribution (dépend du type)
        unit: Unité du paramètre
        description: Description du paramètre
    """
    name: str
    nominal: float
    dist_type: Literal['normal', 'uniform', 'triangular', 'lognormal']
    params: Dict
    unit: str = ""
    description: str = ""

    def sample(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Génère n_samples échantillons selon la distribution définie.

        Args:
            n_samples: Nombre d'échantillons à générer
            seed: Graine aléatoire pour la reproductibilité

        Returns:
            Array numpy de taille n_samples
        """
        if seed is not None:
            np.random.seed(seed)

        if self.dist_type == 'normal':
            # Paramètres: mean, std
            mean = self.params.get('mean', self.nominal)
            std = self.params.get('std', 0)
            return np.random.normal(mean, std, n_samples)

        elif self.dist_type == 'uniform':
            # Paramètres: low, high
            low = self.params.get('low', self.nominal * 0.9)
            high = self.params.get('high', self.nominal * 1.1)
            return np.random.uniform(low, high, n_samples)

        elif self.dist_type == 'triangular':
            # Paramètres: low, mode, high
            low = self.params.get('low', self.nominal * 0.9)
            mode = self.params.get('mode', self.nominal)
            high = self.params.get('high', self.nominal * 1.1)
            return np.random.triangular(low, mode, high, n_samples)

        elif self.dist_type == 'lognormal':
            # Paramètres: mean, sigma (en échelle log)
            mean = self.params.get('mean', np.log(self.nominal))
            sigma = self.params.get('sigma', 0.1)
            return np.random.lognormal(mean, sigma, n_samples)

        else:
            raise ValueError(f"Type de distribution non supporté: {self.dist_type}")

    def get_bounds(self) -> Tuple[float, float]:
        """Retourne les bornes (min, max) de la distribution."""
        if self.dist_type == 'normal':
            mean = self.params.get('mean', self.nominal)
            std = self.params.get('std', 0)
            return (mean - 3*std, mean + 3*std)

        elif self.dist_type == 'uniform':
            return (self.params['low'], self.params['high'])

        elif self.dist_type == 'triangular':
            return (self.params['low'], self.params['high'])

        elif self.dist_type == 'lognormal':
            mean = self.params.get('mean', np.log(self.nominal))
            sigma = self.params.get('sigma', 0.1)
            return (np.exp(mean - 3*sigma), np.exp(mean + 3*sigma))

        return (self.nominal, self.nominal)


def get_default_distributions() -> Dict[str, ParameterDistribution]:
    """
    Retourne les distributions par défaut pour tous les paramètres incertains.

    Returns:
        Dictionnaire {nom_paramètre: ParameterDistribution}
    """
    distributions = {}

    # Paramètres économiques

    # 1. CAPEX PEM (install_fee)
    distributions['install_fee'] = ParameterDistribution(
        name='install_fee',
        nominal=1800,
        dist_type='triangular',
        params={
            'low': 1800 * 0.80,   # -20%
            'mode': 1800,
            'high': 1800 * 1.20   # +20%
        },
        unit='€/kW',
        description='Coût d\'installation de l\'électrolyseur PEM'
    )

    # 2. OPEX PEM
    distributions['OPEX_PEM'] = ParameterDistribution(
        name='OPEX_PEM',
        nominal=54,
        dist_type='triangular',
        params={
            'low': 54 * 0.85,    # -15%
            'mode': 54,
            'high': 54 * 1.15    # +15%
        },
        unit='€/kW/an',
        description='Coûts d\'exploitation de l\'électrolyseur'
    )

    # 3. Prix de vente H2
    distributions['price_H2'] = ParameterDistribution(
        name='price_H2',
        nominal=2.7,
        dist_type='triangular',
        params={
            'low': 2.7 * 0.70,   # -30%
            'mode': 2.7,
            'high': 2.7 * 1.30   # +30%
        },
        unit='€/kg',
        description='Prix de vente de l\'hydrogène'
    )

    # 4. Prix de l'eau
    distributions['water_price'] = ParameterDistribution(
        name='water_price',
        nominal=0.003,
        dist_type='uniform',
        params={
            'low': 0.003 * 0.60,   # -40%
            'high': 0.003 * 1.40   # +40%
        },
        unit='€/kg',
        description='Prix de l\'eau'
    )

    # 5. CAPEX stockage
    distributions['storage_capex'] = ParameterDistribution(
        name='storage_capex',
        nominal=490,
        dist_type='triangular',
        params={
            'low': 490 * 0.80,   # -20%
            'mode': 490,
            'high': 490 * 1.20   # +20%
        },
        unit='€/kg',
        description='Coût du stockage d\'hydrogène'
    )

    # 6. CAPEX camions
    distributions['truck_capex'] = ParameterDistribution(
        name='truck_capex',
        nominal=610000,
        dist_type='triangular',
        params={
            'low': 610000 * 0.90,   # -10%
            'mode': 610000,
            'high': 610000 * 1.10   # +10%
        },
        unit='€/truck',
        description='Coût d\'un camion de transport'
    )

    # Paramètres techniques

    # 7. Efficacité électrolyseur
    distributions['eta_F_characteristic'] = ParameterDistribution(
        name='eta_F_characteristic',
        nominal=0.0441,
        dist_type='normal',
        params={
            'mean': 0.0441,
            'std': 0.0441 * 0.05 / 3  # ±5% correspond à ±3σ
        },
        unit='',
        description='Efficacité caractéristique de l\'électrolyseur'
    )

    # 8. Limite du réseau
    distributions['grid_limit'] = ParameterDistribution(
        name='grid_limit',
        nominal=1319414,
        dist_type='normal',
        params={
            'mean': 1319414,
            'std': 1319414 * 0.05 / 3  # ±5% correspond à ±3σ
        },
        unit='kW',
        description='Limite de puissance du réseau'
    )

    return distributions


def latin_hypercube_sampling(distributions: Dict[str, ParameterDistribution],
                             n_samples: int,
                             seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Effectue un échantillonnage par hypercube latin pour tous les paramètres.

    Le LHS garantit une meilleure couverture de l'espace des paramètres
    qu'un échantillonnage aléatoire simple.

    Args:
        distributions: Dictionnaire des distributions de paramètres
        n_samples: Nombre d'échantillons à générer
        seed: Graine aléatoire pour la reproductibilité

    Returns:
        Dictionnaire {nom_paramètre: array d'échantillons}
    """
    if seed is not None:
        np.random.seed(seed)

    samples = {}
    n_params = len(distributions)

    # Générer les indices LHS
    lhs_indices = np.zeros((n_samples, n_params))
    for i in range(n_params):
        # Diviser [0,1] en n_samples intervalles et prendre un point aléatoire dans chaque
        intervals = np.arange(n_samples) / n_samples
        random_offsets = np.random.uniform(0, 1/n_samples, n_samples)
        lhs_indices[:, i] = intervals + random_offsets
        # Permuter aléatoirement
        np.random.shuffle(lhs_indices[:, i])

    # Transformer les indices LHS en échantillons selon chaque distribution
    for idx, (param_name, dist) in enumerate(distributions.items()):
        uniform_samples = lhs_indices[:, idx]

        # Transformer les échantillons uniformes selon la distribution cible
        if dist.dist_type == 'normal':
            mean = dist.params.get('mean', dist.nominal)
            std = dist.params.get('std', 0)
            # Transformation inverse de la fonction de répartition normale
            from scipy.stats import norm
            samples[param_name] = norm.ppf(uniform_samples, loc=mean, scale=std)

        elif dist.dist_type == 'uniform':
            low = dist.params['low']
            high = dist.params['high']
            samples[param_name] = low + uniform_samples * (high - low)

        elif dist.dist_type == 'triangular':
            low = dist.params['low']
            mode = dist.params['mode']
            high = dist.params['high']
            # Transformation inverse pour distribution triangulaire
            fc = (mode - low) / (high - low)
            mask = uniform_samples < fc
            samples[param_name] = np.where(
                mask,
                low + np.sqrt(uniform_samples * (high - low) * (mode - low)),
                high - np.sqrt((1 - uniform_samples) * (high - low) * (high - mode))
            )

        elif dist.dist_type == 'lognormal':
            mean = dist.params.get('mean', np.log(dist.nominal))
            sigma = dist.params.get('sigma', 0.1)
            from scipy.stats import lognorm
            samples[param_name] = lognorm.ppf(uniform_samples, s=sigma, scale=np.exp(mean))

    return samples


def create_scenario_samples(scenario: Literal['best', 'base', 'worst']) -> Dict[str, float]:
    """
    Crée un échantillon pour un scénario spécifique (best/base/worst case).

    Args:
        scenario: Type de scénario ('best', 'base', 'worst')

    Returns:
        Dictionnaire {nom_paramètre: valeur}
    """
    distributions = get_default_distributions()
    samples = {}

    for param_name, dist in distributions.items():
        if scenario == 'base':
            samples[param_name] = dist.nominal
        elif scenario == 'best':
            # Meilleur cas: favorable pour réduire le LCOH
            if param_name in ['install_fee', 'OPEX_PEM', 'water_price', 'storage_capex', 'truck_capex']:
                # Coûts bas
                bounds = dist.get_bounds()
                samples[param_name] = bounds[0]
            elif param_name == 'price_H2':
                # Prix de vente élevé
                bounds = dist.get_bounds()
                samples[param_name] = bounds[1]
            elif param_name == 'eta_F_characteristic':
                # Efficacité élevée
                bounds = dist.get_bounds()
                samples[param_name] = bounds[1]
            else:
                samples[param_name] = dist.nominal
        elif scenario == 'worst':
            # Pire cas: défavorable pour le LCOH
            if param_name in ['install_fee', 'OPEX_PEM', 'water_price', 'storage_capex', 'truck_capex']:
                # Coûts élevés
                bounds = dist.get_bounds()
                samples[param_name] = bounds[1]
            elif param_name == 'price_H2':
                # Prix de vente bas
                bounds = dist.get_bounds()
                samples[param_name] = bounds[0]
            elif param_name == 'eta_F_characteristic':
                # Efficacité basse
                bounds = dist.get_bounds()
                samples[param_name] = bounds[0]
            else:
                samples[param_name] = dist.nominal

    return samples
