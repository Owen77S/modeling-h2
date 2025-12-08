# 🚀 QUICKSTART - Centrale Hydrogène

## Démarrage Rapide

### Prérequis
- Python 3.9 ou supérieur
- pip (gestionnaire de paquets Python)
- Git (optionnel)

### Installation

#### 1. Cloner le projet ou accéder au dossier

```bash
cd modeling-h2/streamlit_app
```

#### 2. Installer les dépendances

```bash
# Windows
pip install -r requirements.txt

# Linux/Mac
pip install -r requirements.txt
```

### Lancer l'application

#### Option A: Utiliser le script de démarrage (Recommandé)

**Windows:**
```bash
run.bat
```

**Linux/Mac:**
```bash
bash run.sh
```

#### Option B: Commande manuelle

```bash
streamlit run app.py
```

### Accéder à l'application

L'application s'ouvre automatiquement dans votre navigateur à:
```
http://localhost:8501
```

Si ce n'est pas le cas, ouvrez manuellement cette URL dans votre navigateur.

---

## 🎯 Guide de Navigation

### 1️⃣ **Page d'Accueil**
- Vue d'ensemble du projet
- Présentation des résultats clés
- Navigation vers les différentes sections

### 2️⃣ **Introduction**
- Contexte de la problématique
- Hypothèses du système
- Paramètres clés

### 3️⃣ **Modèle et Équations**
- Explication physique du système
- Équations principales
- Démonstration interactive

### 4️⃣ **Implémentation et Code**
- Architecture technique
- Choix de conception
- Code documenté

### 5️⃣ **Dashboard Données**
- Exploration interactive des données
- Statistiques descriptives
- Visualisations des productions
- Export en CSV

### 6️⃣ **Optimisation AG** ⭐ (Plus Important)
**C'est la section clé:**
1. Configurez les paramètres dans la barre latérale
2. Cliquez sur **"Démarrer"**
3. Observez l'optimisation en temps réel
4. Consultez les résultats détaillés

### 7️⃣ **Analyse de Sensibilité**
- Impact des paramètres individuels
- Heatmaps bi-variées
- Analyse limite réseau
- Sensibilité capacité éolienne

### 8️⃣ **Conclusions**
- Synthèse des résultats
- Limitations du modèle
- Perspectives futures

---

## ⚙️ Configuration

### Paramètres de l'Application

Modifiables dans **sidebar de chaque page**:

- **Population** : 20-200 individus (défaut: 50)
- **Générations** : 10-100 (défaut: 30)
- **Crossover** : 0.5-1.0 (défaut: 0.95)
- **Mutation** : 0.1-1.0 (défaut: 0.75)

### Bornes d'Optimisation

Configurables dans le **sidebar** (page Optimisation):
- Capacité électrolyseur: 1,000 - 200,000 kW
- Stockage: 50 - 2,000 m³
- Camions: 1 - 30 unités
- Seuil de vente: 30% - 95%

---

## 📊 Résultats Typiques

Avec la configuration optimale trouvée:

| Métrique | Valeur | Unité |
|----------|--------|-------|
| **LCOH** | 0.165 | €/kWh |
| **LCOH** | 5.49 | €/kg |
| **H2 Annuel** | 2,978,162 | kg |
| **Capacité Électrolyseur** | 49,161 | kW |
| **Stockage** | 326 | m³ |
| **Camions** | 11 | - |

---

## 🐛 Dépannage

### L'application ne démarre pas

```bash
# Vérifiez que Python est installé
python --version

# Vérifiez les dépendances
pip list

# Réinstallez les dépendances
pip install -r requirements.txt --upgrade
```

### Port 8501 déjà utilisé

```bash
streamlit run app.py --server.port 8502
```

### Données non trouvées

L'application génère des données de démonstration si `data_2.xlsx` est absent.
Pour utiliser vos propres données, placez `data_2.xlsx` dans le dossier parent.

### Problèmes de performance

- Réduisez la **taille de la population**
- Diminuez le **nombre de générations**
- Réduisez la **résolution de la grille** en analyse de sensibilité

---

## 📚 Documentation Complète

### Fichiers Importants

| Fichier | Description |
|---------|-------------|
| `app.py` | Point d'entrée principal |
| `config.py` | Configuration globale (paramètres, couleurs) |
| `utils/model.py` | Classe H2PlantModel (cœur du modèle) |
| `utils/genetic_algorithm.py` | Implémentation de l'AG |
| `utils/visualizations.py` | Graphiques Plotly |
| `utils/data_loader.py` | Chargement et traitement des données |

### Modèle Physique

Le système simule:
1. **Production d'électricité** : Éolien + Nucléaire
2. **Gestion réseau** : Limite de capacité
3. **Électrolyse** : Rendement faradique
4. **Compression** : Loi des gaz parfaits
5. **Stockage** : Gestion du buffer H2
6. **Transport** : Logistique par camions
7. **Économie** : Calcul du LCOH

### Variables Optimisées

L'algorithme génétique optimise 4 paramètres:
- **C** : Capacité électrolyseur [kW]
- **S** : Capacité stockage [m³]
- **N** : Nombre de camions
- **T** : Seuil de déclenchement vente (0-1)

**Objectif**: Minimiser le LCOH (Levelized Cost of Hydrogen)

**Contraintes**:
- Puissance perdue < 80%
- Hydrogène perdu < 80%

---

## 🔗 Ressources

### Données d'Entrée

- **Renewable Ninja** : Profils éoliens
- **ENTSO-E** : Données réseau
- **Base de données nucléaires** : Productions nucléaires

### Technologies

- **Streamlit** : Interface web interactive
- **Plotly** : Visualisations interactives
- **NumPy/Pandas** : Calculs scientifiques
- **Python 3.10+** : Langage de programmation

### Références Scientifiques

- IEA Hydrogen Reports
- IRENA Green Hydrogen Cost
- EU Hydrogen Strategy 2020
- IEEE Transactions on Energy Conversion

---

## 💡 Conseils d'Utilisation

### Pour Commencer
1. Lancez l'application
2. Explorez la page **Introduction** pour comprendre le contexte
3. Consultez le **Dashboard Données** pour voir les profils de production
4. Étudiez le **Modèle** pour comprendre les équations
5. Lancez une **Optimisation** simple (30 générations, 50 individus)

### Pour Approfondir
1. Augmentez les **générations** à 50-100
2. Augmentez la **population** à 100-200
3. Explorez les **analyses de sensibilité**
4. Testez différentes **bornes d'optimisation**
5. Modifiez les **seuils de contraintes**

### Pour la Recherche
1. Compilez les **historiques d'AG** en CSV
2. Utilisez les **données exportées** pour analyse externe
3. Adaptez le **modèle** à votre cas d'usage
4. Intégrez des **données réelles** via APIs

---

## 📞 Support

### Questions Fréquentes

**Q: Puis-je modifier les données d'entrée?**
R: Oui, remplacez `data_2.xlsx` par vos propres données dans le format exact.

**Q: Comment exporter les résultats?**
R: Utilisez les boutons **"Télécharger CSV"** sur la page d'optimisation.

**Q: Puis-je ajouter de nouveaux paramètres?**
R: Oui, modifiez les classes dans `utils/model.py` et `config.py`.

**Q: Comment paralléliser l'optimisation?**
R: Vous pouvez adapter `utils/genetic_algorithm.py` pour utiliser le multiprocessing.

---

## 🎓 Apprentissage

Cette application démontre:

✅ **Modélisation énergétique** - Physique complète d'un système H2
✅ **Optimisation** - Algorithmes génétiques robustes
✅ **Data Science** - Analyse et visualisation
✅ **Web Development** - Interface Streamlit professionnelle
✅ **Software Engineering** - Code structuré et documenté
✅ **Python avancé** - OOP, dataclasses, type hints

---

## 📝 Notes

- L'application utilise le cache Streamlit pour optimiser les performances
- Les calculs de l'AG sont mono-thread (peut être parallélisé)
- Les données sont simulées si les vrais fichiers ne sont pas trouvés
- La configuration de thème se fait dans `.streamlit/config.toml`

---

**Bon usage de l'application! 🚀**

Pour toute question technique, consultez la documentation complète dans chaque page.
