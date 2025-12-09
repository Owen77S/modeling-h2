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

    /* Titres */
    .main-title {
        text-align: center;
        color: #1E88E5;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 20px;
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

# Page d'accueil
st.markdown('<h1 class="main-title">Analyse techno-économique d\'une centrale H2 pour réduire les risques de congestion</h1>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin: 2rem 0;">
    <p style="font-size: 1.2em; color: #555;">
        Cette application interactive présente un projet complet de modélisation
        et d'optimisation d'une centrale de production d'hydrogène vert pour limiter les risques de congestion.
    </p>
</div>
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
        <p>LCOH optimal de 0.148 €/kWh avec une valorisation de 98%
        de l'hydrogène produit et respect des contraintes.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Aperçu des fonctionnalités
st.header("Aperçu des fonctionnalités du site")

# 1. Modèle physique
st.subheader("1. Modélisations")
st.markdown("""
- Visualisation des productions éolienne et nucléaire
- Électrolyseur PEM avec efficacité faradique
- Compression et stockage haute pression
- Gestion logistique du transport
- Modèle économique complet
""")

# Afficher l'image de méthodologie
try:
    from PIL import Image
    method_img = Image.open("method.png")
    st.image(method_img, caption="Méthodologie de modélisation et optimisation")
except Exception as e:
    st.warning(f"Image de méthodologie non disponible: {e}")

st.markdown("---")

# 2. Optimisation interactive
st.subheader("2. Optimisation interactive")
st.markdown("""
- Lancement de l'algorithme génétique (AG) en temps réel
- Suivi de la convergence génération par génération
- Configuration des paramètres génétiques
- Visualisation 3D de la population dans l'espace des solutions (disponible dans l'onglet Optimisation GA)
""")

# Visualisation 3D avec données en dur pour performance
import numpy as np
import plotly.graph_objects as go

# Données d'une itération réelle stockées en dur (pour performance)
# Population de 20 individus d'une itération de l'AG
C_pop = np.array([49161, 45230, 52340, 38900, 61200, 43100, 55670, 47890, 50120, 41500,
                  53400, 46780, 58900, 44300, 51200, 39800, 48500, 56200, 42900, 54100])
S_pop = np.array([326, 410, 280, 520, 195, 450, 310, 385, 298, 475,
                  265, 390, 340, 430, 305, 490, 360, 270, 455, 315])
N_pop = np.array([11, 13, 9, 15, 7, 14, 10, 12, 11, 13,
                  8, 12, 10, 14, 9, 15, 11, 8, 14, 10])
LCOH_pop = np.array([0.165, 0.172, 0.168, 0.195, 0.178, 0.185, 0.171, 0.169, 0.166, 0.182,
                     0.174, 0.170, 0.177, 0.188, 0.173, 0.192, 0.175, 0.179, 0.186, 0.176])

# Point optimal (meilleur de la population)
idx_best = np.argmin(LCOH_pop)
C_opt, S_opt, N_opt = C_pop[idx_best], S_pop[idx_best], N_pop[idx_best]
LCOH_opt = LCOH_pop[idx_best]

fig_ag_3d = go.Figure()

# Population
fig_ag_3d.add_trace(go.Scatter3d(
    x=C_pop,
    y=S_pop,
    z=N_pop,
    mode='markers',
    marker=dict(
        size=8,
        color=LCOH_pop,
        colorscale='RdYlGn_r',
        showscale=True,
        colorbar=dict(
            title="LCOH<br>[€/kWh]",
            x=-0.15,
            len=0.7
        ),
        line=dict(color='white', width=1),
        cmin=0.16,
        cmax=0.20
    ),
    text=[f'C: {c:.0f} kW<br>S: {s:.0f} m³<br>N: {n}<br>LCOH: {lcoh:.3f}'
          for c, s, n, lcoh in zip(C_pop, S_pop, N_pop, LCOH_pop)],
    hovertemplate='%{text}<extra></extra>',
    name='Population AG'
))

# Point optimal
fig_ag_3d.add_trace(go.Scatter3d(
    x=[C_opt],
    y=[S_opt],
    z=[N_opt],
    mode='markers',
    marker=dict(
        size=18,
        color='red',
        symbol='diamond',
        line=dict(color='darkred', width=3)
    ),
    text=f'<b>Solution optimale</b><br>C: {C_opt:.0f} kW<br>S: {S_opt:.0f} m³<br>N: {N_opt}<br>LCOH: {LCOH_opt:.3f} €/kWh',
    hovertemplate='%{text}<extra></extra>',
    name='Optimal',
    showlegend=True
))

fig_ag_3d.update_layout(
    scene=dict(
        xaxis=dict(title='Capacité électrolyseur [kW]', range=[35000, 65000]),
        yaxis=dict(title='Capacité stockage [m³]', range=[150, 550]),
        zaxis=dict(title='Nombre de camions', range=[6, 16]),
        camera=dict(
            eye=dict(x=1.4, y=1.4, z=1.2)
        )
    ),
    template='plotly_white',
    height=550,
    showlegend=True,
    legend=dict(x=0.65, y=0.95, bgcolor='rgba(255,255,255,0.8)'),
    margin=dict(l=0, r=0, t=30, b=0),
    title=dict(
        text="Population d'une itération de l'algorithme génétique",
        font=dict(size=14)
    )
)

st.plotly_chart(fig_ag_3d, use_container_width=True, key='ag_3d_overview')

st.caption("Visualisation 3D d'une population de l'algorithme génétique. Chaque point représente une solution candidate.")

st.markdown("---")

# 3. Analyses des résultats
st.subheader("3. Analyses des résultats")
st.markdown("""
- Synthèse des KPIs de la solution optimale
- Analyse de sensibilité paramétrique (C, S, N)
- Sensibilité à la limite réseau
- Sensibilité à la capacité éolienne
- Analyse de Monte-Carlo pour quantification d'incertitude
""")

st.markdown("---")


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
st.markdown("""
<div class='footer'>
    <p><strong>Portfolio Technique - Modélisation Énergétique</strong></p>
    <p>Développé avec Streamlit | Python | Plotly</p>
    <p style='font-size: 0.8rem;'>
        Démonstration de compétences en modélisation, optimisation et développement d'applications
    </p>
</div>
""", unsafe_allow_html=True)
