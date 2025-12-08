# -*- coding: utf-8 -*-
"""
Algorithme Génétique pour l'optimisation de la centrale hydrogène
Version adaptée pour Streamlit avec callbacks temps réel
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Callable, Optional
import time


@dataclass
class Individual:
    """Représente un individu (solution) dans la population."""
    genes: List[float]  # [C, S, N, T]
    fitness: float = float('inf')

    def __post_init__(self):
        self.genes = list(self.genes)

    @property
    def C(self) -> float:
        return self.genes[0]

    @property
    def S(self) -> float:
        return self.genes[1]

    @property
    def N(self) -> int:
        return int(self.genes[2])

    @property
    def T(self) -> float:
        return self.genes[3]

    def to_dict(self) -> Dict:
        return {
            'electrolyzer_capacity': self.C,
            'storage_capacity': self.S,
            'number_of_trucks': self.N,
            'threshold': self.T,
            'LCOH': self.fitness
        }


@dataclass
class GACallback:
    """Callbacks pour suivre la progression de l'AG."""
    on_generation: Optional[Callable] = None
    on_evaluation: Optional[Callable] = None
    on_improvement: Optional[Callable] = None
    should_stop: Optional[Callable] = None


@dataclass
class GAHistory:
    """Historique de l'optimisation."""
    generations: List[int] = field(default_factory=list)
    best_fitness: List[float] = field(default_factory=list)
    mean_fitness: List[float] = field(default_factory=list)
    std_fitness: List[float] = field(default_factory=list)
    best_individuals: List[Dict] = field(default_factory=list)
    population_diversity: List[float] = field(default_factory=list)
    execution_times: List[float] = field(default_factory=list)

    def add_generation(self, gen: int, population: List[Individual], exec_time: float):
        """Ajoute les statistiques d'une génération."""
        fitnesses = [ind.fitness for ind in population if ind.fitness < float('inf')]

        if fitnesses:
            self.generations.append(gen)
            self.best_fitness.append(min(fitnesses))
            self.mean_fitness.append(np.mean(fitnesses))
            self.std_fitness.append(np.std(fitnesses))
            self.best_individuals.append(population[0].to_dict())
            self.execution_times.append(exec_time)

            # Diversité génétique (écart-type normalisé des gènes)
            genes_array = np.array([ind.genes for ind in population])
            diversity = np.mean(np.std(genes_array, axis=0) / (np.mean(genes_array, axis=0) + 1e-10))
            self.population_diversity.append(diversity)

    def to_dict(self) -> Dict:
        return {
            'generations': self.generations,
            'best_fitness': self.best_fitness,
            'mean_fitness': self.mean_fitness,
            'std_fitness': self.std_fitness,
            'best_individuals': self.best_individuals,
            'population_diversity': self.population_diversity,
            'execution_times': self.execution_times,
        }


class GeneticAlgorithm:
    """
    Algorithme génétique pour optimiser les paramètres de la centrale H2.
    """

    def __init__(
        self,
        plant,
        bounds: Dict[str, Tuple[float, float]],
        population_size: int = 50,
        n_generations: int = 30,
        crossover_prob: float = 0.95,
        mutation_prob: float = 0.75,
        elite_ratio: float = 0.05,
        tournament_size: int = 3,
        seed: Optional[int] = None
    ):
        """
        Initialise l'algorithme génétique.

        Args:
            plant: Instance de H2PlantModel
            bounds: Dictionnaire des bornes {param: (min, max)}
            population_size: Taille de la population
            n_generations: Nombre de générations
            crossover_prob: Probabilité de crossover
            mutation_prob: Probabilité de mutation
            elite_ratio: Ratio d'élites conservées
            tournament_size: Taille du tournoi de sélection
            seed: Graine aléatoire pour reproductibilité
        """
        self.plant = plant
        self.bounds = bounds
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elite_ratio = elite_ratio
        self.tournament_size = tournament_size

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # État
        self.population: List[Individual] = []
        self.history = GAHistory()
        self.best_individual: Optional[Individual] = None
        self.current_generation = 0
        self.is_running = False
        self.is_paused = False

        # Contraintes
        self.max_wasted_power = 0.80
        self.max_wasted_hydrogen = 0.80

        # Stagnation detection
        self.stagnation_count = 0
        self.stagnation_threshold = 5
        self.last_best_fitness = float('inf')

    def _random_individual(self) -> Individual:
        """Crée un individu aléatoire."""
        genes = [
            random.uniform(self.bounds['C'][0], self.bounds['C'][1]),
            random.uniform(self.bounds['S'][0], self.bounds['S'][1]),
            random.randint(int(self.bounds['N'][0]), int(self.bounds['N'][1])),
            random.uniform(self.bounds['T'][0], self.bounds['T'][1]),
        ]
        return Individual(genes)

    def _evaluate(self, individual: Individual) -> float:
        """Évalue un individu et retourne son fitness (LCOH)."""
        try:
            # Assurer que N est entier et T est dans [0,1]
            C = max(self.bounds['C'][0], min(self.bounds['C'][1], individual.C))
            S = max(self.bounds['S'][0], min(self.bounds['S'][1], individual.S))
            N = max(int(self.bounds['N'][0]), min(int(self.bounds['N'][1]), individual.N))
            T = max(0.1, min(0.99, individual.T))

            # Évaluer
            lcoh = self.plant.objective(C, S, N, T)

            # Vérifier contraintes
            wp_ok, wh_ok = self.plant.check_constraints(
                self.max_wasted_power,
                self.max_wasted_hydrogen
            )

            # Pénalité si contraintes violées
            if not wp_ok or not wh_ok:
                lcoh = float('inf')

            return lcoh

        except Exception:
            return float('inf')

    def _initialize_population(self, callback: Optional[GACallback] = None) -> None:
        """Initialise la population avec des individus valides."""
        self.population = []
        attempts = 0
        max_attempts = self.population_size * 20

        while len(self.population) < self.population_size and attempts < max_attempts:
            ind = self._random_individual()
            ind.fitness = self._evaluate(ind)

            if ind.fitness < float('inf'):
                self.population.append(ind)
                if callback and callback.on_evaluation:
                    callback.on_evaluation(len(self.population), self.population_size)

            attempts += 1

        # Compléter avec des individus même invalides si nécessaire
        while len(self.population) < self.population_size:
            ind = self._random_individual()
            ind.fitness = self._evaluate(ind)
            self.population.append(ind)

        self._sort_population()

    def _sort_population(self) -> None:
        """Trie la population par fitness."""
        self.population.sort(key=lambda x: x.fitness)
        if self.population and (self.best_individual is None or
                                self.population[0].fitness < self.best_individual.fitness):
            self.best_individual = Individual(
                self.population[0].genes.copy(),
                self.population[0].fitness
            )

    def _tournament_selection(self) -> Individual:
        """Sélection par tournoi."""
        candidates = random.sample(self.population, min(self.tournament_size, len(self.population)))
        return min(candidates, key=lambda x: x.fitness)

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Crossover arithmétique."""
        if random.random() > self.crossover_prob:
            return Individual(parent1.genes.copy() if parent1.fitness < parent2.fitness
                            else parent2.genes.copy())

        child_genes = []
        for i in range(4):
            alpha = random.random()
            gene = alpha * parent1.genes[i] + (1 - alpha) * parent2.genes[i]
            child_genes.append(gene)

        # Assurer N entier
        child_genes[2] = round(child_genes[2])

        return Individual(child_genes)

    def _mutate(self, individual: Individual) -> Individual:
        """Mutation gaussienne adaptative."""
        if random.random() > self.mutation_prob:
            return individual

        genes = individual.genes.copy()

        # Choisir un gène à muter
        i = random.randint(0, 3)

        # Amplitude de mutation adaptative
        bound_range = self.bounds[['C', 'S', 'N', 'T'][i]][1] - self.bounds[['C', 'S', 'N', 'T'][i]][0]
        mutation_strength = 0.1 * bound_range

        # Mutation gaussienne
        genes[i] += random.gauss(0, mutation_strength)

        # Contraindre aux bornes
        bounds_key = ['C', 'S', 'N', 'T'][i]
        genes[i] = max(self.bounds[bounds_key][0], min(self.bounds[bounds_key][1], genes[i]))

        # N doit être entier
        genes[2] = round(genes[2])

        return Individual(genes)

    def _crazy_mutation(self, individual: Individual) -> Individual:
        """Mutation forte pour échapper aux optima locaux."""
        genes = individual.genes.copy()

        # Muter plusieurs gènes fortement
        for i in range(4):
            if random.random() < 0.5:
                bounds_key = ['C', 'S', 'N', 'T'][i]
                genes[i] = random.uniform(self.bounds[bounds_key][0], self.bounds[bounds_key][1])

        genes[2] = round(genes[2])
        return Individual(genes)

    def _local_search(self, individual: Individual, n_steps: int = 3) -> Individual:
        """Recherche locale pour améliorer un individu."""
        best = individual

        for _ in range(n_steps):
            for i in range(4):
                bounds_key = ['C', 'S', 'N', 'T'][i]
                step = 0.01 * (self.bounds[bounds_key][1] - self.bounds[bounds_key][0])

                for direction in [-1, 1]:
                    new_genes = best.genes.copy()
                    new_genes[i] += direction * step
                    new_genes[i] = max(self.bounds[bounds_key][0],
                                      min(self.bounds[bounds_key][1], new_genes[i]))
                    new_genes[2] = round(new_genes[2])

                    candidate = Individual(new_genes)
                    candidate.fitness = self._evaluate(candidate)

                    if candidate.fitness < best.fitness:
                        best = candidate

        return best

    def step(self, callback: Optional[GACallback] = None) -> bool:
        """
        Exécute une génération de l'AG.

        Returns:
            True si l'algorithme continue, False si terminé
        """
        if self.current_generation >= self.n_generations:
            return False

        if callback and callback.should_stop and callback.should_stop():
            return False

        start_time = time.time()

        # Élitisme
        n_elite = max(1, int(self.elite_ratio * self.population_size))
        new_population = [Individual(ind.genes.copy(), ind.fitness)
                         for ind in self.population[:n_elite]]

        # Améliorer les élites par recherche locale
        for i in range(min(2, n_elite)):
            improved = self._local_search(new_population[i])
            if improved.fitness < new_population[i].fitness:
                new_population[i] = improved

        # Détecter stagnation
        if self.population[0].fitness >= self.last_best_fitness * 0.9999:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
            self.last_best_fitness = self.population[0].fitness

        # Générer nouveaux individus
        while len(new_population) < self.population_size:
            # Sélection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            # Crossover
            child = self._crossover(parent1, parent2)

            # Mutation
            if self.stagnation_count >= self.stagnation_threshold:
                # Mutation forte si stagnation
                child = self._crazy_mutation(child)
            else:
                child = self._mutate(child)

            # Évaluation
            child.fitness = self._evaluate(child)
            new_population.append(child)

        # Ajouter quelques individus aléatoires pour diversité
        n_random = max(1, int(0.05 * self.population_size))
        for _ in range(n_random):
            if len(new_population) >= self.population_size:
                # Remplacer un mauvais individu
                idx = random.randint(self.population_size // 2, self.population_size - 1)
                new_ind = self._random_individual()
                new_ind.fitness = self._evaluate(new_ind)
                new_population[idx] = new_ind

        # Mettre à jour la population
        self.population = new_population
        self._sort_population()

        # Reset stagnation si amélioration
        if self.stagnation_count >= self.stagnation_threshold:
            self.stagnation_count = 0

        # Enregistrer historique
        exec_time = time.time() - start_time
        self.history.add_generation(self.current_generation, self.population, exec_time)

        # Callback
        if callback and callback.on_generation:
            callback.on_generation(
                self.current_generation,
                self.population[0],
                self.history
            )

        self.current_generation += 1
        return self.current_generation < self.n_generations

    def run(self, callback: Optional[GACallback] = None) -> Tuple[Individual, GAHistory]:
        """
        Exécute l'algorithme complet.

        Returns:
            (meilleur individu, historique)
        """
        self.is_running = True
        self.current_generation = 0
        self.history = GAHistory()
        self.stagnation_count = 0
        self.last_best_fitness = float('inf')

        # Initialisation
        self._initialize_population(callback)

        # Évolution
        while self.step(callback):
            if self.is_paused:
                while self.is_paused and self.is_running:
                    time.sleep(0.1)

            if not self.is_running:
                break

        self.is_running = False
        return self.best_individual, self.history

    def pause(self):
        """Met en pause l'algorithme."""
        self.is_paused = True

    def resume(self):
        """Reprend l'algorithme."""
        self.is_paused = False

    def stop(self):
        """Arrête l'algorithme."""
        self.is_running = False
        self.is_paused = False

    def get_top_n(self, n: int = 10) -> List[Dict]:
        """Retourne les N meilleurs individus."""
        return [ind.to_dict() for ind in self.population[:n]]

    def get_statistics(self) -> Dict:
        """Retourne les statistiques actuelles."""
        if not self.population:
            return {}

        fitnesses = [ind.fitness for ind in self.population if ind.fitness < float('inf')]

        return {
            'generation': self.current_generation,
            'best_fitness': min(fitnesses) if fitnesses else float('inf'),
            'mean_fitness': np.mean(fitnesses) if fitnesses else float('inf'),
            'std_fitness': np.std(fitnesses) if fitnesses else 0,
            'valid_individuals': len(fitnesses),
            'total_individuals': len(self.population),
            'stagnation_count': self.stagnation_count,
        }
