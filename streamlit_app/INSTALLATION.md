# 📦 Guide d'Installation Complet

## ✅ Checklist Pré-Installation

Avant de commencer, vérifiez:

- [ ] Vous avez accès à Internet (pour télécharger les paquets)
- [ ] Vous avez au moins 500 MB d'espace disque libre
- [ ] Vous êtes utilisateur non-administrateur autorisé (recommandé)

---

## 🖥️ Installation par Système

### Windows

#### 1. Vérifier Python

```bash
# Ouvrir PowerShell ou Cmd
python --version
```

**Attendu:** `Python 3.9.x` ou supérieur

**Si non trouvé:**
- Télécharger depuis https://www.python.org/downloads/
- **IMPORTANT:** Cocher "Add Python to PATH" lors de l'installation
- Redémarrer l'ordinateur

#### 2. Accéder au Dossier

```bash
cd modeling-energy-systems\modeling-h2\streamlit_app
```

#### 3. Créer un Environnement Virtuel (Optionnel mais recommandé)

```bash
python -m venv venv
venv\Scripts\activate
```

#### 4. Installer les Dépendances

```bash
pip install -r requirements.txt
```

**Attendu:** Pas d'erreur, tous les paquets installés

#### 5. Vérifier l'Installation

```bash
python -m streamlit --version
python -c "import numpy; print('NumPy OK')"
python -c "import pandas; print('Pandas OK')"
python -c "import plotly; print('Plotly OK')"
```

#### 6. Lancer l'Application

**Option A (Recommandée):**
```bash
run.bat
```

**Option B (Manuel):**
```bash
streamlit run app.py
```

---

### Linux (Ubuntu/Debian)

#### 1. Vérifier Python

```bash
python3 --version
```

**Attendu:** `Python 3.9` ou supérieur

**Si non trouvé:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

#### 2. Accéder au Dossier

```bash
cd modeling-energy-systems/modeling-h2/streamlit_app
```

#### 3. Créer un Environnement Virtuel

```bash
python3 -m venv venv
source venv/bin/activate
```

#### 4. Installer les Dépendances

```bash
pip install -r requirements.txt
```

#### 5. Vérifier l'Installation

```bash
streamlit --version
python -c "import numpy; print('NumPy OK')"
python -c "import pandas; print('Pandas OK')"
```

#### 6. Lancer l'Application

**Option A (Recommandée):**
```bash
bash run.sh
```

**Option B (Manuel):**
```bash
streamlit run app.py
```

---

### macOS

#### 1. Vérifier Python

```bash
python3 --version
```

**Attendu:** `Python 3.9` ou supérieur

**Si non trouvé (avec Homebrew):**
```bash
brew install python3
```

#### 2. Accéder au Dossier

```bash
cd modeling-energy-systems/modeling-h2/streamlit_app
```

#### 3. Créer un Environnement Virtuel

```bash
python3 -m venv venv
source venv/bin/activate
```

#### 4. Installer les Dépendances

```bash
pip install -r requirements.txt
```

#### 5. Vérifier l'Installation

```bash
streamlit --version
python -c "import numpy; print('NumPy OK')"
```

#### 6. Lancer l'Application

```bash
bash run.sh
```

---

## 🔧 Dépannage d'Installation

### Erreur: "Python command not found"

**Windows:**
```bash
# Utiliser python3 au lieu de python
python3 --version

# Ou ajouter Python au PATH manuellement
```

**Linux/Mac:**
```bash
# Vérifier l'installation
which python3

# Créer un alias si nécessaire
alias python=python3
```

### Erreur: "pip command not found"

```bash
# Réinstaller pip
python -m ensurepip --upgrade

# Ou
python -m pip install --upgrade pip
```

### Erreur: "No module named streamlit"

```bash
# Vérifier l'activation du venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Puis réinstaller
pip install streamlit==1.28.0
```

### Erreur: "Port 8501 already in use"

```bash
# Utiliser un port différent
streamlit run app.py --server.port 8502

# Ou tuer le processus (Linux/Mac)
lsof -ti:8501 | xargs kill -9
```

### Erreur lors de l'installation des paquets

```bash
# Mettre à jour pip d'abord
python -m pip install --upgrade pip

# Puis réessayer
pip install -r requirements.txt

# Si ça échoue, installer individuellement
pip install streamlit==1.28.0
pip install pandas==2.0.0
pip install numpy==1.24.0
pip install plotly==5.18.0
pip install openpyxl==3.1.0
pip install pillow==10.0.0
```

### Erreur de mémoire

Si l'application consomme trop de RAM:

```bash
# Avec moins de cache
streamlit run app.py --client.caching_enabled=false

# Ou limiter la taille de cache
streamlit run app.py --client.caching_max_size=1
```

---

## 📊 Vérification Post-Installation

Après installation, vérifier que tout fonctionne:

### 1. Tester les Imports

```python
python
>>> import streamlit as st
>>> import pandas as pd
>>> import numpy as np
>>> import plotly.graph_objects as go
>>> from PIL import Image
>>> print("Tous les imports OK!")
```

### 2. Tester les Modules

```bash
# Depuis le dossier streamlit_app
python -c "from utils.model import H2PlantModel; print('Model OK')"
python -c "from utils.genetic_algorithm import GeneticAlgorithm; print('GA OK')"
python -c "from utils.visualizations import create_power_chart; print('Viz OK')"
```

### 3. Lancer l'Application

```bash
streamlit run app.py
```

Vérifier que:
- [ ] L'application démarre sans erreur
- [ ] La page d'accueil s'affiche
- [ ] Les graphiques se chargent
- [ ] La navigation entre pages fonctionne

---

## 🌐 Accès à l'Application

### Local
```
http://localhost:8501
```

### Depuis une autre machine (même réseau)
```
http://<votre-ip>:8501
```

Trouver votre IP:
- **Windows:** `ipconfig` → IPv4 Address
- **Linux/Mac:** `ifconfig` → inet

### Public (avec tunneling)
```bash
streamlit run app.py --logger.level=debug --client.remoteWebsocketUrl=<url>
```

---

## 📦 Mise à Jour

### Mettre à jour les paquets

```bash
pip install --upgrade -r requirements.txt
```

### Mettre à jour Streamlit spécifiquement

```bash
pip install --upgrade streamlit
```

### Vérifier les versions

```bash
pip list | grep -E "streamlit|pandas|numpy|plotly"
```

---

## 🧹 Nettoyage

### Supprimer l'environnement virtuel

**Windows:**
```bash
rmdir /s venv
```

**Linux/Mac:**
```bash
rm -rf venv
```

### Vider le cache Streamlit

**Windows:**
```bash
rmdir /s %USERPROFILE%\.streamlit\cache
```

**Linux/Mac:**
```bash
rm -rf ~/.streamlit/cache
```

---

## 🔄 Réinstallation Complète

Si quelque chose ne fonctionne pas:

```bash
# 1. Supprimer l'environnement virtuel
# Windows: rmdir /s venv
# Linux/Mac: rm -rf venv

# 2. Supprimer le cache
# Windows: rmdir /s %USERPROFILE%\.streamlit\cache
# Linux/Mac: rm -rf ~/.streamlit/cache

# 3. Réinstaller
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate

# 4. Réinstaller les dépendances
pip install -r requirements.txt

# 5. Lancer
streamlit run app.py
```

---

## 📚 Ressources

### Documentation Officielle
- [Python.org](https://www.python.org/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Pandas Docs](https://pandas.pydata.org/)
- [NumPy Docs](https://numpy.org/)
- [Plotly Docs](https://plotly.com/)

### Tutoriels
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)
- [Streamlit Getting Started](https://docs.streamlit.io/library/get-started)
- [Pip Documentation](https://pip.pypa.io/)

---

## ✨ Configuration Avancée

### Variables d'Environnement

```bash
# Windows (PowerShell)
$env:STREAMLIT_SERVER_PORT = 8502

# Linux/Mac
export STREAMLIT_SERVER_PORT=8502
```

### Fichier de Configuration

Créer `~/.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"

[server]
port = 8501
headless = true
runOnSave = true
```

### Logs Détaillés

```bash
streamlit run app.py --logger.level=debug
```

---

## 📞 Support

### En Cas de Problème

1. **Vérifiez Python:** `python --version` (3.9+)
2. **Vérifiez pip:** `pip --version`
3. **Réinstallez les dépendances:** `pip install -r requirements.txt`
4. **Consultez QUICKSTART.md** pour FAQ
5. **Consultez README.md** pour documentation

### Commandes Utiles

```bash
# Vérifier toutes les dépendances
pip check

# Voir les paquets installés
pip list

# Mettre à jour tous les paquets
pip install --upgrade pip setuptools wheel

# Créer un fichier des dépendances actuelles
pip freeze > requirements.txt
```

---

**Installation complète! 🎉 Prêt à utiliser l'application.**

Rendez-vous dans [QUICKSTART.md](QUICKSTART.md) pour commencer!
