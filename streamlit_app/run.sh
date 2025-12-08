#!/bin/bash
# Script de démarrage de l'application Streamlit - Linux/Mac
# Usage: ./run.sh ou bash run.sh

echo ""
echo "===================================================="
echo "  Démarrage de l'application - Centrale Hydrogène"
echo "===================================================="
echo ""

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "ERREUR: Python 3 n'est pas installé."
    echo "Veuillez installer Python 3.9+ depuis https://www.python.org"
    echo ""
    exit 1
fi

# Afficher la version de Python
echo "Python détecté:"
python3 --version
echo ""

# Créer un environnement virtuel (optionnel mais recommandé)
if [ ! -d "venv" ]; then
    echo "Création d'un environnement virtuel..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERREUR: Impossible de créer l'environnement virtuel."
        exit 1
    fi
fi

# Activer l'environnement virtuel
echo "Activation de l'environnement virtuel..."
source venv/bin/activate

# Vérifier si streamlit est installé
if ! python -m streamlit --version &> /dev/null; then
    echo "Installation des dépendances..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERREUR: L'installation des dépendances a échoué."
        echo "Veuillez vérifier votre connexion Internet."
        exit 1
    fi
fi

echo "Dépendances vérifiées ✓"
echo ""
echo "Lancement de l'application..."
echo ""
echo "L'application s'ouvrira dans votre navigateur à: http://localhost:8501"
echo ""
echo "Appuyez sur Ctrl+C pour arrêter l'application"
echo ""

# Lancer l'application
python -m streamlit run app.py

# Vérifier le code de sortie
if [ $? -ne 0 ]; then
    echo ""
    echo "ERREUR: L'application n'a pas pu démarrer."
    echo "Pour plus d'informations, consultez la documentation."
    exit 1
fi
