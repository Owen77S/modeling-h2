# -*- coding: utf-8 -*-
"""
Application Streamlit - Modélisation et Optimisation d'une Centrale Hydrogène
Point d'entrée principal de l'application

Auteur: Owen Sogbadji
Description: Démonstration de compétences en modélisation énergétique,
             optimisation par algorithme génétique et développement d'applications.
"""

import streamlit as st

# Configuration de la page principale
st.set_page_config(
    page_title="Centrale Hydrogène - Portfolio",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """
        ## Modélisation d'une Centrale Hydrogène

        Application de démonstration technique présentant:
        - Modélisation physique d'un système de production H2
        - Optimisation par algorithme génétique
        - Analyse de sensibilité multi-paramètres
        - Visualisations interactives

        Développé avec Streamlit, Python, NumPy et Plotly.
        """
    }
)

# Style CSS personnalisé
st.markdown("""
<style>
    /* Style général */
    .main {
        padding: 1rem;
    }

    /* Titres */
    h1 {
        color: #1f77b4;
        padding-bottom: 10px;
    }

    h2 {
        color: #2c3e50;
    }

    /* Métriques */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #1f77b4;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }

    [data-testid="stSidebar"] h1 {
        font-size: 1.5rem;
        color: #2c3e50;
    }

    /* Boutons */
    .stButton > button {
        border-radius: 5px;
        font-weight: 500;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #2c3e50;
    }

    /* Tables */
    .dataframe {
        font-size: 0.9rem;
    }

    /* Cards effect */
    .element-container {
        transition: transform 0.2s;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - Navigation et présentation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/hydrogen.png", width=80)
    st.title("Centrale H2")
    st.markdown("---")

    st.markdown("""
    ### Navigation

    Utilisez les pages ci-dessous pour explorer le projet:

    1. **Introduction** - Contexte et objectifs
    2. **Modèle** - Équations et physique
    3. **Code** - Implémentation technique
    4. **Données** - Dashboard exploratoire
    5. **Optimisation** - AG en temps réel
    6. **Sensibilité** - Analyses paramétriques
    7. **Conclusions** - Synthèse et perspectives
    """)

    st.markdown("---")

    st.markdown("""
    ### À propos

    Cette application démontre des compétences en:
    - Modélisation énergétique
    - Algorithmes d'optimisation
    - Data science & visualisation
    - Développement d'applications
    """)

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8rem;'>
        Portfolio Technique<br>
        Modélisation Énergétique
    </div>
    """, unsafe_allow_html=True)

# Page d'accueil principale
st.title("Analyse techno-économique d'une centrale à hydrogène optimisée pour réduire les risques de congestion")

st.markdown("""
<p style='font-size: 1.1rem;'>
        Cette application interactive présente un projet complet de modélisation
        et d'optimisation d'une centrale de production d'hydrogène vert pour limiter les risques de congestion.
    </p>
""", unsafe_allow_html=True)

# Vue d'ensemble
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='background: #e3f2fd; padding: 20px; border-radius: 10px; '>
        <h3>Objectif</h3>
        <p>Minimiser le coût de production d'hydrogène (LCOH) en optimisant
        le dimensionnement d'une centrale utilisée pour réduire les risques de congestion.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background: #e8f5e9; padding: 20px; border-radius: 10px;'>
        <h3>Méthode</h3>
        <p>Modélisation Python et optimisation avec algorithme génétique de 4 variables de décision:
        capacité électrolyseur, stockage, transport et seuil de vente.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='background: #fff3e0; padding: 20px; border-radius: 10px;'>
        <h3>Résultats</h3>
        <p>LCOH optimal de 0.165 €/kWh avec une valorisation de 98%
        de l'hydrogène produit et respect des contraintes.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Aperçu des fonctionnalités
st.header("Aperçu des fonctionnalités")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Exploration des données")
    st.markdown("""
    - Visualisation des productions éolienne et nucléaire
    - Analyse des excédents de puissance
    - Statistiques descriptives complètes
    """)

    st.subheader("Optimisation interactive")
    st.markdown("""
    - Lancement de l'algorithme génétique (AG) en temps réel
    - Suivi de la convergence génération par génération
    - Configuration des paramètres génétiques
    - Visualisation de la population
    """)

with col2:
    st.subheader("Modèle physique")
    st.markdown("""
    - Électrolyseur PEM avec efficacité faradique
    - Compression et stockage haute pression
    - Gestion logistique du transport
    - Calcul du LCOH complet
    """)

    st.subheader("Analyses des résultats")
    st.markdown("""
    - Synthèse des résultats
    - Sensibilité à la limite réseau
    - Sensibilité à la capacité éolienne
    - Analyse de Monte-Carlo
    """)

st.markdown("---")

# # Guide de démarrage
# st.header("Pour commencer")

# st.markdown("""
# 1. **Explorez les données** dans l'onglet "Dashboard Données" pour comprendre les profils de production
# 2. **Étudiez le modèle** dans "Modèle et Équations" pour comprendre la physique du système
# 3. **Lancez une optimisation** dans "Optimisation AG" pour voir l'algorithme en action
# 4. **Analysez la sensibilité** pour comprendre l'impact des différents paramètres
# 5. **Consultez les conclusions** pour une synthèse complète du projet

# 👈 **Utilisez la barre latérale** pour naviguer entre les différentes sections.
# """)

# Technologies utilisées
st.header("Technologies utilisées")

tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)

with tech_col1:
    st.markdown("""
    **Python**
    - NumPy
    - Pandas
    - SciPy
    """)

with tech_col2:
    st.markdown("""
    **Visualisation**
    - Plotly
    - Streamlit
    - Matplotlib
    """)

with tech_col3:
    st.markdown("""
    **Optimisation**
    - AG from sratch
    """)

with tech_col4:
    st.markdown("""
    **Data**
    - Excel/CSV
    - Renewable Ninja
    - Données réelles
    """)

# Footer
st.markdown("---")
st.markdown("""
<div class='footer'>
    <p><strong>Portfolio Technique - Modélisation Énergétique</strong></p>
    <p>Développé avec Streamlit | Python | Plotly</p>
    <p style='font-size: 0.8rem;'>
        Démonstration de compétences en modélisation, optimisation et développement d'applications
    </p>
</div>
""", unsafe_allow_html=True)
