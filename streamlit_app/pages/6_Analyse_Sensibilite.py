# -*- coding: utf-8 -*-
"""
Page 6: Analyse de Sensibilité
Exploration de l'impact des paramètres sur les performances
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_power_data
from utils.model import H2PlantModel
from utils.visualizations import COLORS, create_sensitivity_heatmap
from utils.image_loader import display_image, get_section_images
from config import OPTIMAL_DESIGN

st.set_page_config(
    page_title="Analyse de résultats - Centrale H2",
    page_icon="📊",
    layout="wide"
)

st.title("Analyse de résultats")


# Charger les données
@st.cache_resource
def get_plant():
    df = load_power_data(104)
    plant = H2PlantModel()
    plant.load_data(df['WP'].values, df['NP'].values)
    plant.compute_excess_power()
    return plant

plant = get_plant()

# Configuration de base (design optimal)
base_config = {
    'C': OPTIMAL_DESIGN['electrolyzer_capacity'],
    'S': OPTIMAL_DESIGN['storage_capacity'],
    'N': OPTIMAL_DESIGN['number_of_trucks'],
    'T': OPTIMAL_DESIGN['threshold']
}

base_C = 49161
base_S = 326
base_N = 11
base_T = 0.9

# Mettre à jour la config de base
base_config = {'C': base_C, 'S': base_S, 'N': base_N, 'T': base_T}

# KPIs de référence
@st.cache_data
def get_base_kpis(C, S, N, T):
    temp_plant = H2PlantModel()
    df = load_power_data(104)
    temp_plant.load_data(df['WP'].values, df['NP'].values)
    return temp_plant.evaluate(C, S, N, T)

base_kpis = get_base_kpis(base_C, base_S, base_N, base_T)

# Afficher les KPIs de référencest.markdown("---")

st.subheader("Configuration optimale")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Capacité électrolyseurs [kW]", "49161")
with col2:
    st.metric("Capacité stockage [m³]", "326")
with col3:
    st.metric("Nombre de camions", "11")
with col4:
    st.metric("LCOH [€/kWh]", "0.165")

st.markdown('---')

st.subheader("KPIs")
col2, col3, col4 = st.columns(3)
with col2:
    st.metric("H2 produit", "3,0 kt/an")
with col3:
    st.metric("Electricité perdue", f"69.2%")
with col4:
    st.metric("Pertes H2", f"2%")

st.markdown('---')

st.subheader("Coûts")
st.markdown("""La technologie PEM est la partie la plus coûteuse de l'usine à hydrogène. Par conséquent, si
le prix de la technologie PEM diminue à mesure que la technologie mûrit, cela pourrait réduire considérablement le LCOE de la
production d'hydrogène.""")
# Section CAPEX et OPEX Breakdown
from utils.optimization import define_plant, show_management
from utils.visualizations import create_capex_breakdown_chart, create_opex_breakdown_chart

# Utiliser la configuration optimale
optimal_plant = define_plant(C=49161, S=326, N=11, T=1.0)

# Obtenir tous les CAPEX et OPEX détaillés
kpis = optimal_plant.get_KPI_2()

# Afficher les totaux
col1, col2 = st.columns(2)
with col1:
    st.metric("CAPEX Total", f"{kpis['CAPEX_total']/1e6:.2f} M€")
with col2:
    st.metric("OPEX Total", f"{kpis['OPEX_total']/1e6:.2f} M€/an")

# Afficher les camemberts CAPEX et OPEX
col1, col2 = st.columns(2)

with col1:
    fig_capex = create_capex_breakdown_chart(kpis)
    st.plotly_chart(fig_capex, width='stretch', key='capex_breakdown_sensibilite')

with col2:
    fig_opex = create_opex_breakdown_chart(kpis)
    st.plotly_chart(fig_opex, width='stretch', key='opex_breakdown_sensibilite')

st.markdown("---")

st.subheader("Gestion de l'hydrogène")
fig1, fig2 = show_management(optimal_plant)

# Afficher le premier graphique
st.plotly_chart(fig1, width='stretch', key='management_fig1')

# Afficher la légende entre les deux graphiques
st.markdown("""
<div style='text-align: center; padding: 10px;'>
    <span style='color: #1f77b4; font-size: 14px; margin-right: 20px;'>━━ Hydrogen produced</span>
    <span style='color: #ff7f0e; font-size: 14px; margin-right: 20px;'>━━ Hydrogen stored</span>
    <span style='color: red; font-size: 14px; margin-right: 20px;'>- - Storage capacity</span>
    <span style='color: #2ca02c; font-size: 14px; margin-right: 20px;'>━━ Amount of hydrogen produced</span>
    <span style='color: #d62728; font-size: 14px;'>━━ Amount of hydrogen wasted</span>
</div>
""", unsafe_allow_html=True)

# Afficher le deuxième graphique
st.plotly_chart(fig2, width='stretch', key='management_fig2')

st.markdown("---")

st.subheader("Analyse de sensibilité")

display_image()