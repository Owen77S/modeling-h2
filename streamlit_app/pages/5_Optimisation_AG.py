# -*- coding: utf-8 -*-
"""
Page 5: Optimisation par Algorithme Génétique
Interface interactive avec visualisation en temps réel
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_power_data
from utils.model import H2PlantModel
from utils.genetic_algorithm import GeneticAlgorithm, GACallback, GAHistory
from utils.visualizations import (
    create_ga_convergence_chart, create_ga_live_chart,
    create_population_scatter, create_kpi_comparison_chart,
    create_capex_breakdown_chart, create_h2_production_chart,
    create_h2_management_chart, create_3d_population_scatter, COLORS
)
from utils.image_loader import display_image, get_section_images
from config import OPTIMIZATION_BOUNDS, GA_DEFAULT_PARAMS, OPTIMAL_DESIGN

# Import pour la visualisation 3D
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.optimization import algo_with_tracking, empty_plant

st.set_page_config(
    page_title="Optimisation - Centrale H2",
    page_icon="🧬",
    layout="wide"
)

st.title("Optimisation par algorithme génétique")
st.markdown("---")

# Initialisation du session state
if 'ga_running' not in st.session_state:
    st.session_state.ga_running = False
if 'ga_paused' not in st.session_state:
    st.session_state.ga_paused = False
if 'ga_history' not in st.session_state:
    st.session_state.ga_history = None
if 'best_solution' not in st.session_state:
    st.session_state.best_solution = None
if 'population' not in st.session_state:
    st.session_state.population = None
if 'current_gen' not in st.session_state:
    st.session_state.current_gen = 0
if 'total_gens' not in st.session_state:
    st.session_state.total_gens = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'kpis_before' not in st.session_state:
    st.session_state.kpis_before = None
if 'kpis_after' not in st.session_state:
    st.session_state.kpis_after = None

# Session state pour la visualisation 3D
if 'ga_3d_results' not in st.session_state:
    st.session_state.ga_3d_results = None
if 'current_3d_iteration' not in st.session_state:
    st.session_state.current_3d_iteration = 0
if 'ga_3d_lcoh_history' not in st.session_state:
    st.session_state.ga_3d_lcoh_history = None
if 'ga_3d_best_config' not in st.session_state:
    st.session_state.ga_3d_best_config = None
if 'ga_3d_best_lcoh' not in st.session_state:
    st.session_state.ga_3d_best_lcoh = None

# Sidebar - Configuration de l'AG
with st.sidebar:
    st.header("Paramètres de l'AG")

    st.subheader("Population")
    population_size = st.slider(
        "Taille de la population",
        min_value=20, max_value=200, value=GA_DEFAULT_PARAMS['population_size'],
        help="Nombre d'individus par génération"
    )

    n_generations = st.slider(
        "Nombre de générations",
        min_value=10, max_value=100, value=GA_DEFAULT_PARAMS['n_generations'],
        help="Nombre d'itérations de l'algorithme"
    )



    st.subheader("Bornes des variables")

    with st.expander("Capacité électrolyseurs [kW]"):
        c_min = st.number_input("Min", value=OPTIMIZATION_BOUNDS['C_min'], key='c_min')
        c_max = st.number_input("Max", value=OPTIMIZATION_BOUNDS['C_max'], key='c_max')

    with st.expander("Capacité stockage [m³]"):
        s_min = st.number_input("Min", value=OPTIMIZATION_BOUNDS['S_min'], key='s_min')
        s_max = st.number_input("Max", value=OPTIMIZATION_BOUNDS['S_max'], key='s_max')

    with st.expander("Nombre de camions"):
        n_min = st.number_input("Min", value=OPTIMIZATION_BOUNDS['N_min'], key='n_min')
        n_max = st.number_input("Max", value=OPTIMIZATION_BOUNDS['N_max'], key='n_max')

# Charger les données et créer le modèle
@st.cache_resource
def get_plant():
    df = load_power_data(104)
    plant = H2PlantModel()
    plant.load_data(df['WP'].values, df['NP'].values)
    plant.compute_excess_power()
    return plant

plant = get_plant()

st.subheader("Problème d'optimisaton")

st.markdown("""
On optimise l'entiereté de la centrale à hydrogène pour que son design soit le plus adapté aux centrales électriques et à la localisation de la centrale.

##### Fonction objetif : 
- LCOH : fournit une mesure complète de la viabilité économique de notre système énergétique.

##### Paramètres à optimiser
- Capacité de l'électrolyseurs [kW] : principal coût,
- Capacité stockage [m³]
- Nombre de camions

##### Contraintes (pour fournir une solution durable d'un point de vue environnemental)
- pourcentage d'électricité perdue < 20%,
- pourcentage d'hydrogène perdu < 20%     
""")
display_image("OPTI KPI 2.png", caption="Problème d'optimisation", use_column_width=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(""" 
     """)

with col2:
    display_image("Geneticalgo.png", caption="Principe de l'algorithme génétique")

# Interface principale de visualisation 3D
st.markdown("---")
st.header("🔬 Visualisation 3D de l'Algorithme Génétique")

st.markdown("""
Cette interface permet de visualiser l'évolution de la population de l'algorithme génétique
dans un espace 3D, où chaque point représente un design possible de la centrale H2.

- **Axe X** : Capacité de l'électrolyseurs [kW]
- **Axe Y** : Capacité de stockage [m³]
- **Axe Z** : Nombre de camions
- **Couleur** : LCOH (rouge = mauvais, vert = bon)

Le formulaire à gauche permet de modifier les paramètres de l'algorithme génétique. **La taille de la population et le
nombre d'itérations maximum ont été réduit pour favoriser un rendu rapide.**
""")

# Boutons de contrôle principaux
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("▶️ Lancer l'optimisation", key='play_3d', width='stretch'):
        with st.spinner("Optimisation en cours..."):
            plant = empty_plant()
            best_config, best_lcoh, lcoh_history, final_pop, all_iterations = \
                algo_with_tracking(
                    plant, population_size, n_generations,
                    c_min, c_max, s_min, s_max, n_min, n_max,
                    track_population=True, printed=0
                )
            st.session_state.ga_3d_results = all_iterations
            st.session_state.ga_3d_lcoh_history = lcoh_history
            st.session_state.ga_3d_best_config = best_config
            st.session_state.ga_3d_best_lcoh = best_lcoh
            st.session_state.current_3d_iteration = 0
        st.success("✅ Optimisation terminée!")
        st.rerun()

with col_btn2:
    if st.button("🔄 Réinitialiser", key='reset_3d', disabled=st.session_state.ga_3d_results is None, width='stretch'):
        st.session_state.ga_3d_results = None
        st.session_state.ga_3d_lcoh_history = None
        st.session_state.ga_3d_best_config = None
        st.session_state.ga_3d_best_lcoh = None
        st.session_state.current_3d_iteration = 0
        st.rerun()

if st.session_state.ga_3d_results is not None and len(st.session_state.ga_3d_results) > 0:
    st.subheader("Choix de l'itération avec slider")

    # Afficher le slider seulement s'il y a plus d'une itération
    if len(st.session_state.ga_3d_results) > 1:
        # Le slider met à jour current_3d_iteration via son callback
        st.slider(
            "Itération",
            0,
            len(st.session_state.ga_3d_results) - 1,
            st.session_state.current_3d_iteration,
            key='iteration_slider_3d',
            on_change=lambda: setattr(st.session_state, 'current_3d_iteration', st.session_state.iteration_slider_3d)
        )
    else:
        # S'il n'y a qu'une seule itération, afficher simplement l'information
        st.info("Une seule itération disponible")
        st.session_state.current_3d_iteration = 0

if st.session_state.ga_3d_results is not None and len(st.session_state.ga_3d_results) > 0:
    st.markdown("---")

    total = len(st.session_state.ga_3d_results)
    current = st.session_state.current_3d_iteration
    current_data = st.session_state.ga_3d_results[current]

    # Visualisation 3D et status sur deux colonnes
    col_3d, col_status = st.columns([3, 1])

    with col_3d:
        st.subheader("📊 Population 3D")
        # Graphe 3D avec échelles fixées aux bornes des variables
        fig_3d = create_3d_population_scatter(
            st.session_state.ga_3d_results[current],
            bounds={'C': (c_min, c_max), 'S': (s_min, s_max), 'N': (n_min, n_max)}
        )
        st.plotly_chart(fig_3d, width='stretch', key=f'3d_plot_{current}')

    with col_status:
        st.subheader("📈 Status")

        # Métriques de status
        st.metric("Itération", f"{current + 1} / {total}")

        # Boutons de navigation
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("⏮️ Précédent", key='prev_3d', disabled=current == 0, width='stretch'):
                st.session_state.current_3d_iteration -= 1
                st.rerun()
        with col_next:
            if st.button("⏭️ Suivant", key='next_3d', disabled=current >= total - 1, width='stretch'):
                current += 1
                st.session_state.current_3d_iteration = current
                st.rerun()

        st.markdown("---")
        st.markdown("##### Configuration optimale à l'itération courante")
        st.metric("LCOH [€/kWh]", f"{current_data['best_lcoh']:.4f}")
        st.metric("C [kW]", f"{current_data['best_member'][0]:,.0f}")
        st.metric("S [m³]", f"{current_data['best_member'][1]:.0f}")
        st.metric("N", current_data['best_member'][2])

else:
    st.info("Cliquez sur 'Lancer l'optimisation' pour démarrer la visualisation")

st.markdown("---")

# Évolution des paramètres
st.subheader("Évolution des paramètres")

st.markdown('''**IMPORTANT** : l'algorithme risque de ne pas converger vers la solution optimale car la taille de la population et le nombre de générations sont faibles
pour favoriser une expérience utilisateur agréable.''')
# Vérifier que les données existent
if st.session_state.ga_3d_results is not None and len(st.session_state.ga_3d_results) > 0:
    # Extraire l'historique des meilleurs membres
    C_history = [it['best_member'][0] for it in st.session_state.ga_3d_results]
    S_history = [it['best_member'][1] for it in st.session_state.ga_3d_results]
    N_history = [it['best_member'][2] for it in st.session_state.ga_3d_results]

    col1, col2 = st.columns(2)

    # Graphe d'évolution du LCOH
    fig_lcoh = go.Figure()
    fig_lcoh.add_trace(go.Scatter(
        x=list(range(len(st.session_state.ga_3d_lcoh_history))),
        y=st.session_state.ga_3d_lcoh_history,
        mode='lines+markers',
        name='LCOH',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=3)
    ))
    fig_lcoh.add_trace(go.Scatter(
        x=[current],
        y=[st.session_state.ga_3d_lcoh_history[current]],
        mode='markers',
        marker=dict(size=8, color='red', symbol='diamond')
    ))
    fig_lcoh.update_layout(
        xaxis_title="Itération",
        yaxis_title="LCOH [€/kWh]",
        template="plotly_white",
        height=200,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=30)
    )
    with col1:
        st.plotly_chart(fig_lcoh, width='stretch')

    # Graphe d'évolution de C
    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(
        x=list(range(len(C_history))),
        y=C_history,
        mode='lines+markers',
        line=dict(color='#ff7f0e', width=2),
        marker=dict(size=3)
    ))
    fig_c.add_trace(go.Scatter(
        x=[current],
        y=[C_history[current]],
        mode='markers',
        marker=dict(size=8, color='red', symbol='diamond')
    ))
    fig_c.update_layout(
        xaxis_title="Itération",
        yaxis_title="C [kW]",
        template="plotly_white",
        height=200,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=30)
    )
    with col1:
        st.plotly_chart(fig_c, width='stretch')

    # Graphe d'évolution de S
    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(
        x=list(range(len(S_history))),
        y=S_history,
        mode='lines+markers',
        line=dict(color='#2ca02c', width=2),
        marker=dict(size=3)
    ))
    fig_s.add_trace(go.Scatter(
        x=[current],
        y=[S_history[current]],
        mode='markers',
        marker=dict(size=8, color='red', symbol='diamond')
    ))
    fig_s.update_layout(
        xaxis_title="Itération",
        yaxis_title="S [m³]",
        template="plotly_white",
        height=200,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=30)
    )
    with col2:
        st.plotly_chart(fig_s, width='stretch')

    # Graphe d'évolution de N
    fig_n = go.Figure()
    fig_n.add_trace(go.Scatter(
        x=list(range(len(N_history))),
        y=N_history,
        mode='lines+markers',
        line=dict(color='#d62728', width=2),
        marker=dict(size=3)
    ))
    fig_n.add_trace(go.Scatter(
        x=[current],
        y=[N_history[current]],
        mode='markers',
        marker=dict(size=8, color='red', symbol='diamond')
    ))
    fig_n.update_layout(
        xaxis_title="Itération",
        yaxis_title="N",
        template="plotly_white",
        height=200,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=30)
    )
    with col2:
        st.plotly_chart(fig_n, width='stretch')

    st.markdown("---")

    st.session_state.ga_3d_best_config = [49161, 326, 11]
else:
    st.info("Lancez l'optimisation pour voir l'évolution des paramètres")