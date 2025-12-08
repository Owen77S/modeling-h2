# 🔋 Modélisation et Optimisation d'une Centrale Hydrogène

**Une application Streamlit interactive pour la modélisation énergétique, l'optimisation et l'analyse de sensibilité**

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red)
![License](https://img.shields.io/badge/license-MIT-green)

## 📋 Table des Matières

- [Description](#description)
- [Démarrage Rapide](#démarrage-rapide)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Modèle Physique](#modèle-physique)
- [Optimisation](#optimisation)
- [Technologies](#technologies)
- [Documentation](#documentation)
- [Résultats](#résultats)
- [FAQ](#faq)

## 📖 Description

Cette application démontre un projet complet de modélisation et d'optimisation d'une centrale de production d'hydrogène vert alimentée par les surplus d'électricité d'un système éolien-nucléaire.

### Objectif Principal

**Minimiser le coût de production d'hydrogène (LCOH)** en optimisant simultanément:
- La capacité de l'électrolyseur PEM
- La capacité du stockage haute pression
- Le nombre de camions de transport
- Le seuil de déclenchement de la vente

### Contexte

L'intégration massive des énergies renouvelables crée des situations de congestion réseau. Cette solution valorise les surplus énergétiques par la production d'hydrogène vert, une filière stratégique pour la transition énergétique.

## 🚀 Démarrage Rapide

### Installation

```bash
# Cloner ou accéder au dossier
cd modeling-h2/streamlit_app

# Installer les dépendances
pip install -r requirements.txt
```

### Lancer l'Application

**Windows:**
```bash
run.bat
```

**Linux/Mac:**
```bash
bash run.sh
```

**Ou manuellement:**
```bash
streamlit run app.py
```

L'application s'ouvre automatiquement à `http://localhost:8501`

Pour plus de détails, voir [QUICKSTART.md](QUICKSTART.md)

## ✨ Fonctionnalités

### 1. 📊 Dashboard Exploratoire
- Visualisation interactive des productions éolienne et nucléaire
- Analyse des excédents de puissance
- Statistiques descriptives complètes
- Export des données en CSV

### 2. 🧬 Optimisation par Algorithme Génétique
- Exécution en **temps réel** avec visualisation
- Contrôles play/pause/stop/reset
- Configuration dynamique des paramètres génétiques
- Suivi de la convergence génération par génération
- Export des résultats (solutions, historique)

### 3. 📐 Modèle Physique Complet
- Électrolyseur PEM avec efficacité faradique
- Compression et stockage haute pression (250 bar)
- Gestion logistique du transport (camions)
- Calcul complet du LCOH (Levelized Cost of Hydrogen)

### 4. 📈 Analyses de Sensibilité
- Impact de chaque paramètre individuellement
- Heatmaps bi-variées
- Sensibilité à la limite réseau
- Sensibilité à la capacité éolienne

### 5. 📚 Documentation Interactive
- Explications détaillées du modèle
- Équations mathématiques formatées
- Code documenté et accessible
- Visualisations explicatives

## 🏗️ Architecture

### Structure du Projet

```
streamlit_app/
├── app.py                      # Point d'entrée principal
├── config.py                   # Configuration globale
├── requirements.txt            # Dépendances
├── QUICKSTART.md              # Guide de démarrage rapide
├── run.bat                     # Script Windows
├── run.sh                      # Script Linux/Mac
├── .streamlit/
│   └── config.toml            # Configuration Streamlit
├── pages/                      # Pages de navigation
│   ├── 1_Introduction.py
│   ├── 2_Modele_Equations.py
│   ├── 3_Implementation_Code.py
│   ├── 4_Dashboard_Donnees.py
│   ├── 5_Optimisation_AG.py
│   ├── 6_Analyse_Sensibilite.py
│   └── 7_Conclusions.py
└── utils/                      # Modules utilitaires
    ├── data_loader.py         # Chargement des données
    ├── model.py               # Modèle H2PlantModel
    ├── genetic_algorithm.py   # Algorithme génétique
    ├── visualizations.py      # Graphiques Plotly
    └── image_loader.py        # Gestion des images
```

### Flux de Données

```
data_2.xlsx (Données éoliennes/nucléaires)
    ↓
data_loader.py (Chargement & traitement)
    ↓
H2PlantModel (Simulation horaire)
    ↓
GeneticAlgorithm (Optimisation)
    ↓
visualizations.py (Graphiques interactifs)
    ↓
Streamlit UI (Interface utilisateur)
```

## 🔬 Modèle Physique

### Composantes

#### 1. Production d'Électricité
- Parc éolien: 104 turbines × 3.3 MW (343 MW total)
- Centrale nucléaire: 1450 MW (facteur de charge 92%)
- Simulation horaire: 8760 heures/année

#### 2. Gestion du Réseau
```
P_excess(t) = max(0, P_wind(t) + P_nuclear(t) - P_grid_limit)
```
- Limite réseau: 1,319,414 kW (configurable)
- Cet excédent alimente l'électrolyseur

#### 3. Électrolyse PEM
**Efficacité Faradique:**
```
η_F = 1 - exp(-(P_supply/C) / 0.04409)
```
- Rendement dépendant de la charge
- Pertes auxiliaires: 3%

**Production d'H2:**
```
ṁ_H2 = η_aux × η_F × P_supply / LHV_H2
```

#### 4. Compression et Stockage
- Compression polytropique: 15 bar → 250 bar
- Stockage haute pression
- Pertes évaporatives négligées

#### 5. Transport et Vente
- Flotte de camions (capacité: 29.36 m³ chacun)
- Temps aller-retour: 3 heures
- Seuil de déclenchement configurable

#### 6. Coûts (LCOH)
```
LCOH = (CAPEX + OPEX) / E_H2_vendu
```

**CAPEX:**
- Électrolyseur: 1,800 €/kW
- Stockage: 490 €/kg
- Transport: 610,000 €/camion + 93,296 € fixe

**OPEX annuel:**
- Maintenance PEM: 54 €/kW
- Eau: 0.003 € × 9 L/kg H2
- Transport: 30,500 €/camion

## 🧬 Optimisation

### Algorithme Génétique

**Individu:** `[C, S, N, T]`
- C: Capacité électrolyseur [kW]
- S: Capacité stockage [m³]
- N: Nombre de camions
- T: Seuil de vente (0-1)

**Opérateurs:**
| Opérateur | Description |
|-----------|-------------|
| Sélection | Tournoi (k=3) |
| Crossover | Arithmétique (α aléatoire) |
| Mutation | Gaussienne adaptative |
| Élitisme | Top 5% conservé |
| Diversité | 5% aléatoires/génération |

**Gestion de la Stagnation:**
- Détection après 5 générations sans amélioration
- Mutations "folles" pour s'échapper des optima locaux

**Contraintes:**
- Puissance perdue < 80%
- Hydrogène perdu < 80%

### Résultats Typiques

| Paramètre | Valeur | Unité |
|-----------|--------|-------|
| **LCOH** | 0.165 | €/kWh |
| **LCOH** | 5.49 | €/kg |
| **H2 Annuel** | 2,978,162 | kg |
| **Électrolyseur** | 49,161 | kW |
| **Stockage** | 326 | m³ |
| **Camions** | 11 | - |
| **Seuil** | 0.65 | - |

## 🛠️ Technologies

### Frontend
- **Streamlit** (1.28+) - Interface web interactive
- **Plotly** (5.18+) - Visualisations interactives

### Backend
- **Python** (3.9+) - Langage principal
- **NumPy** (1.24+) - Calculs numériques
- **Pandas** (2.0+) - Manipulation de données
- **SciPy** (1.11+) - Fonctions scientifiques

### Infrastructure
- **Pillow** - Gestion des images
- **openpyxl** - Lecture Excel

## 📚 Documentation

### Pages Principales

1. **Introduction** - Contexte, hypothèses, résultats clés
2. **Modèle et Équations** - Physique complète du système
3. **Implémentation et Code** - Architecture technique
4. **Dashboard Données** - Exploration interactive
5. **Optimisation AG** ⭐ - Section principale avec exécution en temps réel
6. **Analyse de Sensibilité** - Impact des paramètres
7. **Conclusions** - Synthèse et perspectives

### Fichiers Clés

| Fichier | Rôle |
|---------|------|
| `model.py` | Simulation horaire du système |
| `genetic_algorithm.py` | Optimisation multi-objective |
| `visualizations.py` | 15+ graphiques Plotly |
| `data_loader.py` | Gestion des données |

## 📊 Résultats

### Configuration Optimale Trouvée

```
Électrolyseur:    49,161 kW (49 MW)
Stockage:         326 m³
Camions:          11 unités
Seuil de vente:   65%
```

### Performances

```
LCOH:                   0.165 €/kWh (5.49 €/kg)
H2 produit annuel:      2.98 Mt
H2 vendu:               2.92 Mt (98% valorisé)
Pertes puissance:       69.2% (contrainte: <80%)
Pertes H2:              2.0% (contrainte: <80%)
```

## ❓ FAQ

### Q: Puis-je modifier les données d'entrée?
**R:** Oui, remplacez `data_2.xlsx` dans le dossier parent avec vos données au même format.

### Q: Comment exporter les résultats?
**R:** Utilisez les boutons **"Télécharger CSV"** sur la page d'optimisation.

### Q: Puis-je ajouter de nouveaux paramètres?
**R:** Oui, modifiez `model.py` et `config.py`, puis adaptez les pages.

### Q: L'optimisation est lente?
**R:** Réduisez la population (50→30) et les générations (30→20). Ou parallélisez le code.

### Q: Port 8501 déjà utilisé?
```bash
streamlit run app.py --server.port 8502
```

### Q: Erreur "data_2.xlsx non trouvé"?
**R:** L'application génère des données de démonstration automatiquement.

## 🎓 Compétences Démontrées

✅ Modélisation énergétique
✅ Optimisation (algorithmes évolutionnaires)
✅ Programmation Python avancée (OOP, dataclasses, type hints)
✅ Visualisation de données (Plotly, Streamlit)
✅ Analyse scientifique (NumPy, Pandas, SciPy)
✅ Développement d'applications web
✅ Documentation technique

## 📝 Notes

- Cache Streamlit activé pour optimiser les performances
- AG est mono-thread (peut être parallélisé)
- Données simulées si fichiers réels non trouvés
- Configuration de thème dans `.streamlit/config.toml`

## 🤝 Contribution

Les suggestions et améliorations sont bienvenues!

## 📞 Support

Pour toute question:
1. Consultez [QUICKSTART.md](QUICKSTART.md)
2. Voir la documentation intégrée dans chaque page
3. Vérifiez la section FAQ ci-dessus

## 📄 License

MIT - Libre d'utilisation et de modification

---

**Développé avec ❤️ par [Votre Nom]**

*Portfolio technique en modélisation énergétique et optimisation*
