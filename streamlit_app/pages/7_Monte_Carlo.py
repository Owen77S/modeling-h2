# -*- coding: utf-8 -*-
"""
Page 7: Analyse Monte Carlo
Quantification de l'incertitude sur les performances de la centrale H2
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.monte_carlo import MonteCarloAnalyzer, MonteCarloResults
from utils.distributions import get_default_distributions, create_scenario_samples
from utils.monte_carlo_viz import (
    create_histogram_with_stats,
    create_cdf_plot,
    create_tornado_chart,
    create_correlation_heatmap,
    create_parameter_impact_comparison,
    create_sensitivity_bars,
    create_box_plots,
    create_percentile_bands
)
from utils.data_loader import load_power_data
from config import OPTIMAL_DESIGN

st.set_page_config(
    page_title="Analyse Monte Carlo - Centrale H2",
    page_icon="🎲",
    layout="wide"
)

st.title("🎲 Analyse Monte Carlo")
st.markdown("""
L'analyse Monte Carlo permet de quantifier l'impact des incertitudes sur les paramètres économiques
et techniques sur les performances de la centrale H2 (LCOH, production, pertes, etc.).
""")

# Configuration optimale par défaut
default_config = {
    'C': 49161,  # kW
    'S': 326,    # m³
    'N': 11,     # camions
    'T': 0.9     # threshold
}

# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

st.sidebar.subheader("Design de la centrale")
C = st.sidebar.number_input("Capacité électrolyseur [kW]", value=default_config['C'], step=1000)
S = st.sidebar.number_input("Capacité stockage [m³]", value=default_config['S'], step=10)
N = st.sidebar.number_input("Nombre de camions", value=default_config['N'], step=1, min_value=1)
T = st.sidebar.number_input("Threshold", value=default_config['T'], step=0.1, min_value=0.0, max_value=1.0)

design_config = {'C': C, 'S': S, 'N': N, 'T': T}

st.sidebar.markdown("---")

st.sidebar.subheader("Paramètres Monte Carlo")
n_samples = st.sidebar.slider("Nombre d'échantillons", 100, 5000, 1000, 100)
sampling_method = st.sidebar.selectbox("Méthode d'échantillonnage", ['lhs', 'random'])
n_processes = st.sidebar.slider("Processus parallèles", 1, 8, 4)
seed = st.sidebar.number_input("Graine aléatoire", value=42, step=1)

# Bouton pour lancer l'analyse
run_analysis = st.sidebar.button("🚀 Lancer l'analyse Monte Carlo", type="primary")

st.sidebar.markdown("---")
st.sidebar.info("""
**Méthode LHS** (Latin Hypercube Sampling):
Meilleure couverture de l'espace des paramètres avec moins d'échantillons.

**Méthode Random**:
Échantillonnage aléatoire simple.
""")

# Initialiser session state
if 'mc_results' not in st.session_state:
    st.session_state.mc_results = None
if 'mc_analyzer' not in st.session_state:
    st.session_state.mc_analyzer = None

# Section 1: Distributions des paramètres
st.header("1. Distributions des paramètres incertains")

distributions = get_default_distributions()

# Afficher les distributions sous forme de tableau
dist_data = []
for param_name, dist in distributions.items():
    bounds = dist.get_bounds()
    dist_data.append({
        'Paramètre': param_name,
        'Valeur nominale': f"{dist.nominal:.4g}",
        'Type distribution': dist.dist_type,
        'Borne min': f"{bounds[0]:.4g}",
        'Borne max': f"{bounds[1]:.4g}",
        'Unité': dist.unit,
        'Description': dist.description
    })

df_distributions = pd.DataFrame(dist_data)
st.dataframe(df_distributions, use_container_width=True)

st.markdown("---")

# Analyse Monte Carlo
if run_analysis:
    with st.spinner(f"⏳ Exécution de l'analyse Monte Carlo avec {n_samples} échantillons..."):
        try:
            # Créer l'analyseur
            analyzer = MonteCarloAnalyzer(
                design_config=design_config,
                distributions=distributions,
                power_data=None  # Charge automatiquement
            )

            # Exécuter l'analyse
            results = analyzer.run_monte_carlo(
                n_samples=n_samples,
                sampling_method=sampling_method,
                n_processes=n_processes,
                seed=seed
            )

            # Sauvegarder dans session state
            st.session_state.mc_results = results
            st.session_state.mc_analyzer = analyzer

            st.success(f"✅ Analyse terminée avec succès! {n_samples} simulations réalisées.")

        except Exception as e:
            st.error(f"❌ Erreur lors de l'analyse: {e}")
            st.exception(e)

# Afficher les résultats si disponibles
if st.session_state.mc_results is not None:
    results = st.session_state.mc_results
    analyzer = st.session_state.mc_analyzer

    st.markdown("---")
    st.header("2. Résultats de l'analyse")

    # Section 2.1: Statistiques descriptives
    st.subheader("2.1 Statistiques descriptives")

    kpi_choice = st.selectbox(
        "Sélectionner le KPI à analyser",
        options=['LCOH', 'H2', 'H2_waste', 'power_waste', 'CAPEX_total', 'OPEX_total'],
        index=0
    )

    # Afficher les statistiques
    if kpi_choice in results.statistics:
        stats = results.statistics[kpi_choice]

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Cas nominal", f"{stats['base_case']:.4f}")
        with col2:
            st.metric("Moyenne", f"{stats['mean']:.4f}")
        with col3:
            st.metric("Médiane", f"{stats['median']:.4f}")
        with col4:
            st.metric("Écart-type", f"{stats['std']:.4f}")
        with col5:
            st.metric("CV", f"{stats['cv']*100:.1f}%")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min", f"{stats['min']:.4f}")
            st.metric("P5", f"{stats['p5']:.4f}")
        with col2:
            st.metric("P25", f"{stats['p25']:.4f}")
            st.metric("P75", f"{stats['p75']:.4f}")
        with col3:
            st.metric("P95", f"{stats['p95']:.4f}")
            st.metric("Max", f"{stats['max']:.4f}")

    # Section 2.2: Histogramme et CDF
    st.subheader("2.2 Distribution du KPI")

    col1, col2 = st.columns(2)

    with col1:
        fig_hist = create_histogram_with_stats(results, kpi_choice)
        st.plotly_chart(fig_hist, width='stretch', key='mc_histogram')

    with col2:
        fig_cdf = create_cdf_plot(results, kpi_choice)
        st.plotly_chart(fig_cdf, width='stretch', key='mc_cdf')

    # Bandes de percentiles
    fig_bands = create_percentile_bands(results, kpi_choice)
    st.plotly_chart(fig_bands, width='stretch', key='mc_bands')

    st.markdown("---")

    # Section 3: Analyse de sensibilité
    st.header("3. Analyse de sensibilité")

    st.subheader("3.1 Diagramme Tornado")

    with st.spinner("Calcul du diagramme tornado..."):
        tornado_df = analyzer.compute_tornado_data(kpi_name=kpi_choice, variation_pct=0.20)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig_tornado = create_tornado_chart(tornado_df, kpi_choice)
        st.plotly_chart(fig_tornado, width='stretch', key='mc_tornado')

    with col2:
        fig_sens_bars = create_sensitivity_bars(tornado_df, kpi_choice, top_n=10)
        st.plotly_chart(fig_sens_bars, width='stretch', key='mc_sens_bars')

    # Afficher le tableau des sensibilités
    st.dataframe(tornado_df[['parameter', 'base_value', 'kpi_base', 'kpi_low', 'kpi_high', 'impact_total', 'impact_pct']],
                use_container_width=True)

    st.markdown("---")

    # Section 4: Corrélations
    st.header("4. Corrélations")

    st.subheader("4.1 Impact des paramètres")
    fig_impact = create_parameter_impact_comparison(results, kpi_choice)
    st.plotly_chart(fig_impact, width='stretch', key='mc_impact')

    st.subheader("4.2 Matrices de corrélation")

    col1, col2 = st.columns(2)

    with col1:
        fig_corr_pearson = create_correlation_heatmap(results, method='pearson')
        st.plotly_chart(fig_corr_pearson, width='stretch', key='mc_corr_pearson')

    with col2:
        fig_corr_spearman = create_correlation_heatmap(results, method='spearman')
        st.plotly_chart(fig_corr_spearman, width='stretch', key='mc_corr_spearman')

    st.markdown("---")

    # Section 5: Distributions des KPIs
    st.header("5. Distributions de tous les KPIs")

    fig_boxes = create_box_plots(results, kpi_list=['LCOH', 'H2', 'H2_waste', 'power_waste'])
    st.plotly_chart(fig_boxes, width='stretch', key='mc_boxes')

    st.markdown("---")

    # Section 6: Export des résultats
    st.header("6. Export des résultats")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Export échantillons de paramètres
        csv_params = results.parameter_samples.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger échantillons paramètres (CSV)",
            data=csv_params,
            file_name="mc_parameter_samples.csv",
            mime="text/csv"
        )

    with col2:
        # Export résultats KPIs
        csv_kpis = results.kpi_results.to_csv(index=True)
        st.download_button(
            label="📥 Télécharger résultats KPIs (CSV)",
            data=csv_kpis,
            file_name="mc_kpi_results.csv",
            mime="text/csv"
        )

    with col3:
        # Export statistiques
        stats_df = pd.DataFrame(results.statistics).T
        csv_stats = stats_df.to_csv(index=True)
        st.download_button(
            label="📥 Télécharger statistiques (CSV)",
            data=csv_stats,
            file_name="mc_statistics.csv",
            mime="text/csv"
        )

    st.markdown("---")

    # Section 7: Analyse de scénarios
    st.header("7. Analyse de scénarios")

    st.markdown("""
    Comparaison de trois scénarios :
    - **Best case**: Paramètres favorables (coûts bas, efficacité élevée)
    - **Base case**: Valeurs nominales
    - **Worst case**: Paramètres défavorables (coûts élevés, efficacité basse)
    """)

    scenario_results = []

    for scenario in ['best', 'base', 'worst']:
        scenario_params = create_scenario_samples(scenario)
        scenario_kpis = analyzer.run_single_simulation(scenario_params, design_config)
        scenario_kpis['Scénario'] = scenario
        scenario_results.append(scenario_kpis)

    df_scenarios = pd.DataFrame(scenario_results)
    df_scenarios = df_scenarios.set_index('Scénario')

    st.dataframe(df_scenarios[['LCOH', 'H2', 'H2_waste', 'power_waste', 'CAPEX_total', 'OPEX_total']],
                use_container_width=True)

    # Visualisation des scénarios
    import plotly.graph_objects as go

    fig_scenarios = go.Figure()

    scenarios = ['best', 'base', 'worst']
    colors = {'best': 'green', 'base': 'blue', 'worst': 'red'}

    for scenario in scenarios:
        lcoh_val = df_scenarios.loc[scenario, 'LCOH']
        fig_scenarios.add_trace(go.Bar(
            x=[scenario],
            y=[lcoh_val],
            name=scenario.capitalize(),
            marker=dict(color=colors[scenario]),
            text=f"{lcoh_val:.4f}",
            textposition='auto'
        ))

    fig_scenarios.update_layout(
        title="Comparaison du LCOH selon les scénarios",
        xaxis_title="Scénario",
        yaxis_title="LCOH [€/kWh]",
        template="plotly_white",
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig_scenarios, width='stretch', key='mc_scenarios')

else:
    st.info("👈 Configurez les paramètres dans la barre latérale et cliquez sur '🚀 Lancer l'analyse Monte Carlo'")

    st.markdown("""
    ### À propos de l'analyse Monte Carlo

    L'analyse Monte Carlo est une technique de simulation stochastique qui permet de :

    1. **Quantifier l'incertitude** : Comprendre comment les incertitudes sur les paramètres d'entrée
       se propagent aux résultats (LCOH, production, pertes, etc.)

    2. **Identifier les paramètres critiques** : Via le diagramme tornado et les corrélations,
       identifier quels paramètres ont le plus d'impact sur les performances

    3. **Évaluer les risques** : Calculer la probabilité d'atteindre certains objectifs
       (ex: LCOH < 0.15 €/kWh)

    4. **Comparer des scénarios** : Analyser les cas optimiste, nominal et pessimiste

    ### Paramètres incertains considérés

    **Économiques:**
    - CAPEX et OPEX de l'électrolyseur PEM (±15-20%)
    - Prix de vente de l'H2 (±30%)
    - Prix de l'eau (±40%)
    - CAPEX stockage et transport (±10-20%)

    **Techniques:**
    - Efficacité de l'électrolyseur (±5%)
    - Limite du réseau électrique (±5%)

    ### Méthodes d'échantillonnage

    - **LHS (Latin Hypercube Sampling)** : Recommandé. Assure une meilleure couverture
      de l'espace des paramètres avec moins d'échantillons.
    - **Random** : Échantillonnage aléatoire simple. Nécessite plus d'échantillons pour
      une bonne convergence.
    """)
