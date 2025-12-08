# -*- coding: utf-8 -*-
"""
Page 1: Introduction et contexte
Présentation du projet de modélisation de centrale hydrogène
"""

import streamlit as st
import sys
from pathlib import Path

# Ajouter le chemin parent pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_loader import display_image, get_section_images
from config import SYSTEM_PARAMS, ECONOMICS, OPTIMAL_DESIGN, COLORS

# Configuration de la page
st.set_page_config(
    page_title="Introduction - Centrale H2",
    page_icon="🔋",
    layout="wide"
)


# Introduction
st.markdown("""
# Contexte du projet

Ce projet adresse un défi majeur de la transition énergétique : **la gestion des surplus
d'électricité renouvelable** lorsque la production dépasse la capacité du réseau.

### Problématique

L'intégration massive des énergies renouvelables (éolien, solaire) dans le mix énergétique
crée des situations de **congestion du réseau électrique**. Lorsque la production dépasse
la demande et la capacité de transport, l'électricité excédentaire doit être stockée. Sinon elle devra être perdue.

### Solution proposée

La **production d'hydrogène vert** par électrolyse de l'eau représente une solution
prometteuse pour valoriser ces surplus énergétiques. L'hydrogène produit peut être ensuite revalorisé dans l'industrie ou 
en tant que carburant.
""")

st.markdown("")

# Afficher l'image du layout en pleine largeur
if not display_image("layout.png", caption="Architecture du système", use_column_width=True):
    if not display_image("system_layout.png", caption="Architecture du système", use_column_width=True):
        st.info("💡 Architecture: Éolien + Nucléaire → Réseau + H2")

st.markdown("---")

# Architecture du système
st.header("Architecture du système étudié")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    ### Parc éolien
    - **104 turbines** Nordex N131
    - Puissance unitaire: **3.3 MW**
    - Capacité totale: **343 MW**
    - Données: Renewable Ninja
    """)

with col2:
    st.markdown("""
    ### Centrale nucléaire
    - Modèle: **Oskarshamn 3**
    - Capacité: **1450 MW**
    - Profil: Données réelles 2021
    """)

with col3:
    st.markdown("""
    ### Réseau électrique
    - Risque de congestion
    - Injection prioritaire
    - Excédent → Électrolyseur
    """)

with col4:
    st.markdown("""
    ### Centrale hydrogène
    - Électrolyseurs PEM
    - Stockage haute pression
    - Transport par camions
    - Optimisation multi-paramètres
    """)

st.markdown("---")

st.header("Techniques")

col1, col2 = st.columns(2) 

with col1:
    st.markdown("###### Modélisation technique et économique des centrales et réseau électrique")
    st.markdown("###### Analyse de sensitivité")

with col2:
    st.markdown("###### Optimisation multi-paramètres sous contraintes avec algorithme génétique")
    st.markdown("###### Analyse de Monte-Carlo pour estimation d'incertitudes")

st.markdown("---")

# Objectifs du projet
st.header("Objectifs du Projet")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Projet de Modélisation Énergétique | Centrale Hydrogène</p>
    <p>Navigation: Utilisez le menu latéral pour explorer les différentes sections</p>
</div>
""", unsafe_allow_html=True)
