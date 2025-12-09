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

st.markdown("---")


# Section 1: KPIs
st.header("1. Performance - Atteinte des KPIs")

st.markdown("""
Le système de production d'hydrogène présente d'**excellentes performances** au regard des KPIs définis.
Le tableau ci-dessous résume les indicateurs clés de performance de la centrale H2 optimisée.
""")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("LCOH", "0.145 €/kWh", help="Soit 5.494 €/kg")
with col2:
    st.metric("H2 produit", "2 978 t/an", help="2 978 162 kg/an")
with col3:
    st.metric("Électricité gaspillée", "69.2%", delta="-30.8%", delta_color="inverse")
with col4:
    st.metric("H2 gaspillé", "2%", delta="-98%", delta_color="inverse")

st.markdown('---')

# Section 3: Performances économiques
st.header("2. Performances économiques")

st.markdown("""
Selon les résultats de l'optimisation, le projet présente les caractéristiques économiques suivantes :
- **CAPEX total** : 102.29 M€
- **OPEX total** : 3.13 M€/an
- **LCOH** : 0.145 €/kWh

**Analyse :**

La technologie PEM est la **partie la plus coûteuse** de la centrale à hydrogène. Par conséquent,
si le prix de la technologie PEM diminue à mesure que la technologie mûrit, cela pourrait
**réduire considérablement le LCOH** de la production d'hydrogène.
""")
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

st.subheader("Rentabilité du projet")

st.markdown("""
**Analyse de rentabilité selon le prix de vente de l'hydrogène :**

Dans le modèle économique avec période de retour sur investissement (PBP) et valeur actuelle nette (NPV) :

**Scénario actuel (2.7 €/kg en Suède) :**
- **PBP simple** : environ 21-22 ans
- **NPV** : < 0 (projet non rentable)

**Seuil de rentabilité :**
- Pour rendre le projet rentable et obtenir une NPV > 0, le **prix de vente minimal** de l'hydrogène doit être de **3.6 €/kg**
- Le PBP serait alors de 31-32 ans

**Comparaison avec le marché européen :**
- La plupart du marché européen vend l'hydrogène entre **3-6 €/kg**
- Le prix de 3.6 €/kg reste donc **compétitif**

**Facteur clé de réduction du LCOH :**

La technologie PEM représentant la part la plus importante du CAPEX, une **baisse du prix de la technologie PEM**
avec la maturité technologique pourrait **réduire significativement le LCOH** de la production d'hydrogène.
""")

st.markdown("---")

# Section 4: Production et gestion de l'hydrogène
st.header("3. Production et gestion de l'hydrogène")

st.markdown("""
### 3.1 Production horaire sur une année

La **puissance excédentaire, la puissance fournie et la production d'hydrogène**
de la centrale optimisée au cours d'une année sont présentées ci-dessous.

**Observations saisonnières :**

- **En été** : puissance excédentaire très limitée en raison de la faible production des centrales
  nucléaire et éolienne => production d'hydrogène instable.

- **En hiver** : puissance excédentaire élevée => puissance fournie et production d'hydrogène constantes.

**Conclusion :** L'optimisation de la taille de l'électrolyseur doit prendre en compte à la fois les saisons de forte
et de faible puissance excédentaire, car elle affecte grandement **l'efficacité du système et les
performances économiques**.
""")

fig1, fig2 = show_management(optimal_plant)

# Afficher le premier graphique
st.plotly_chart(fig1, width='stretch', key='management_fig1')

# Afficher la légende entre les deux graphiques
st.markdown("""
<div style='text-align: center; padding: 10px;'>
    <span style='color: #1f77b4; font-size: 14px; margin-right: 20px;'>━━ Hydrogen produced</span>
    <span style='color: #ff7f0e; font-size: 14px; margin-right: 20px;'>━━ Hydrogen stored</span>
    <span style='color: red; font-size: 14px; margin-right: 20px;'>- - Storage capacity</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
### 4.2 Production de l'hydrogène

L'**hydrogène produit et perdu** tout au long de l'année sont présentés ci-dessous.

**Observations clés :**

1. **Gaspillage minimal** : Tout au long de l'année, l'hydrogène gaspillé reste à un niveau très faible,
   représentant seulement **2% de l'hydrogène total produit**. Cela signifie que notre système optimisé peut
   **utiliser efficacement l'hydrogène produit**.

2. **Dimensionnement optimal** : Le système optimisé prend soigneusement en compte l'interaction entre la capacité
   de l'électrolyseur, la capacité de stockage et le nombre de camions, ce qui conduit à des **performances
   système satisfaisantes**.
""")

# Afficher la légende entre les deux graphiques
st.markdown("""
<div style='text-align: center; padding: 10px;'>
        <span style='color: #2ca02c; font-size: 14px; margin-right: 20px;'>━━ Amount of hydrogen produced</span>
    <span style='color: #d62728; font-size: 14px;'>━━ Amount of hydrogen wasted</span>
</div>
""", unsafe_allow_html=True)

# Afficher le deuxième graphique
st.plotly_chart(fig2, width='stretch', key='management_fig2')

st.markdown("---")

st.subheader("Analyse de sensibilité")

st.markdown("""
Pour évaluer l'impact des variables et différents paramètres du modèle sur les performances de la centrale H2.
""")

# 1. Sensibilité à la limite du réseau
st.markdown("### 1. Limitation du réseau (Grid Limit)")

st.markdown("""
La valeur de la limite du réseau est le **paramètre principal** de la modélisation, car la puissance fournie
à l'électrolyseur en dépend principalement.

**Impact sur le système :**
- Une limite de réseau plus élevée réduit la puissance excédentaire disponible
- Cela affecte directement la capacité de production d'hydrogène
- Le dimensionnement optimal de la centrale dépend fortement de cette contrainte
""")

display_image('grid limit/1.png', caption="Résultats de l'analyse de sensibilité sur la limite du réseau")

st.markdown("---")

# 2. Sensibilité à la capacité éolienne
st.markdown("### 2. Capacité de la centrale éolienne")

st.markdown("""
La centrale éolienne est le **second paramètre principal** de la modélisation. Sa production horaire
génère ou non des congestions, car le nœud à la limite du réseau devrait être conçu pour pouvoir
transporter toute la puissance produite par la centrale nucléaire.

**Méthodologie :**
- Seule la capacité de la centrale éolienne a été modifiée lors de l'analyse de sensibilité
- Elle est définie comme une fraction de la capacité de la centrale nucléaire
- Le même profil de puissance est conservé tout au long de l'analyse

**Observations clés :**
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    **1. Impact sur les variables de design**

    Plus la capacité éolienne est élevée, plus les différentes variables augmentent :
    - Une capacité éolienne élevée génère une puissance excédentaire moyenne plus importante
    - La capacité de l'électrolyseur augmente pour récupérer autant de puissance que possible, sans être surdimensionnée
    - Cela conduit à plus d'hydrogène produit
    - Le stockage s'agrandit et le nombre de camions augmente
    """)

with col2:
    st.markdown("""
    **2. Impact sur le LCOH**

    Plus la capacité éolienne est élevée, plus le LCOH diminue :
    - Cela traduit l'**effet d'échelle**
    - Plus on produit d'hydrogène, moins il coûte cher
    - La dilution des coûts fixes (CAPEX) sur un plus grand volume de production
    - Meilleure utilisation de l'infrastructure existante
    """)

display_image("wind capacity/sensi.png", caption="Résultats de l'analyse de sensibilité sur la capacité éolienne")

