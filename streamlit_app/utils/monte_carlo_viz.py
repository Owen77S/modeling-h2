# -*- coding: utf-8 -*-
"""
Visualisations pour l'analyse Monte Carlo
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
from utils.monte_carlo import MonteCarloResults


def create_histogram_with_stats(results: MonteCarloResults,
                                kpi_name: str = 'LCOH',
                                title: str = None) -> go.Figure:
    """
    Crée un histogramme des résultats Monte Carlo avec statistiques.

    Args:
        results: Résultats Monte Carlo
        kpi_name: Nom du KPI à visualiser
        title: Titre du graphique (optionnel)

    Returns:
        Figure Plotly
    """
    values = results.kpi_results[kpi_name].dropna()
    stats = results.statistics[kpi_name]

    # Créer l'histogramme
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=50,
        name='Distribution',
        marker=dict(color='#4472C4', line=dict(color='white', width=1)),
        opacity=0.7
    ))

    # Ajouter les lignes de statistiques
    fig.add_vline(x=stats['mean'], line=dict(color='red', width=2, dash='solid'),
                 annotation_text=f"Moyenne: {stats['mean']:.4f}",
                 annotation_position="top")

    fig.add_vline(x=stats['median'], line=dict(color='orange', width=2, dash='dash'),
                 annotation_text=f"Médiane: {stats['median']:.4f}",
                 annotation_position="top")

    fig.add_vline(x=stats['base_case'], line=dict(color='green', width=2, dash='dot'),
                 annotation_text=f"Cas nominal: {stats['base_case']:.4f}",
                 annotation_position="top")

    # Zones de percentiles
    fig.add_vrect(x0=stats['p5'], x1=stats['p95'],
                 fillcolor="green", opacity=0.1,
                 annotation_text="90% CI", annotation_position="bottom left")

    if title is None:
        title = f"Distribution de {kpi_name}"

    fig.update_layout(
        title=title,
        xaxis_title=kpi_name,
        yaxis_title="Fréquence",
        template="plotly_white",
        height=500,
        showlegend=False
    )

    return fig


def create_cdf_plot(results: MonteCarloResults,
                   kpi_name: str = 'LCOH',
                   title: str = None) -> go.Figure:
    """
    Crée une fonction de répartition cumulative (CDF).

    Args:
        results: Résultats Monte Carlo
        kpi_name: Nom du KPI à visualiser
        title: Titre du graphique (optionnel)

    Returns:
        Figure Plotly
    """
    values = results.kpi_results[kpi_name].dropna().sort_values()
    stats = results.statistics[kpi_name]

    # Calculer la CDF
    cdf = np.arange(1, len(values) + 1) / len(values)

    fig = go.Figure()

    # Trace de la CDF
    fig.add_trace(go.Scatter(
        x=values,
        y=cdf * 100,
        mode='lines',
        name='CDF',
        line=dict(color='#4472C4', width=2)
    ))

    # Ajouter les lignes de percentiles
    percentiles = [
        (0.05, 'P5', 'p5'),
        (0.25, 'P25', 'p25'),
        (0.50, 'P50', 'median'),  # P50 = médiane
        (0.75, 'P75', 'p75'),
        (0.95, 'P95', 'p95')
    ]

    for p, label, key in percentiles:
        value = stats[key]
        fig.add_trace(go.Scatter(
            x=[value, value],
            y=[0, p * 100],
            mode='lines',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False
        ))
        fig.add_annotation(
            x=value,
            y=p * 100,
            text=f"{label}: {value:.4f}",
            showarrow=False,
            yshift=10
        )

    # Ligne du cas nominal
    fig.add_vline(x=stats['base_case'], line=dict(color='green', width=2, dash='dot'),
                 annotation_text=f"Nominal: {stats['base_case']:.4f}")

    if title is None:
        title = f"Fonction de répartition cumulative - {kpi_name}"

    fig.update_layout(
        title=title,
        xaxis_title=kpi_name,
        yaxis_title="Probabilité cumulative (%)",
        template="plotly_white",
        height=500,
        yaxis=dict(range=[0, 100])
    )

    return fig


def create_tornado_chart(tornado_df: pd.DataFrame,
                        kpi_name: str = 'LCOH',
                        title: str = None) -> go.Figure:
    """
    Crée un diagramme tornado pour l'analyse de sensibilité.

    Args:
        tornado_df: DataFrame avec les données tornado (de compute_tornado_data)
        kpi_name: Nom du KPI analysé
        title: Titre du graphique (optionnel)

    Returns:
        Figure Plotly
    """
    # Prendre les 10 paramètres les plus influents
    df = tornado_df.head(10).copy()

    # Calculer les déviations par rapport au cas de base
    df['delta_low'] = df['kpi_low'] - df['kpi_base']
    df['delta_high'] = df['kpi_high'] - df['kpi_base']

    # Inverser l'ordre pour avoir le plus influent en haut
    df = df.iloc[::-1]

    fig = go.Figure()

    # Barre gauche (variation basse)
    fig.add_trace(go.Bar(
        y=df['parameter'],
        x=df['delta_low'],
        orientation='h',
        name='Variation -20%',
        marker=dict(color='#ED7D31'),
        text=[f"{v:.4f}" for v in df['delta_low']],
        textposition='auto'
    ))

    # Barre droite (variation haute)
    fig.add_trace(go.Bar(
        y=df['parameter'],
        x=df['delta_high'],
        orientation='h',
        name='Variation +20%',
        marker=dict(color='#4472C4'),
        text=[f"{v:.4f}" for v in df['delta_high']],
        textposition='auto'
    ))

    if title is None:
        title = f"Diagramme Tornado - Impact sur {kpi_name}"

    fig.update_layout(
        title=title,
        xaxis_title=f"Impact sur {kpi_name}",
        yaxis_title="Paramètre",
        template="plotly_white",
        height=600,
        barmode='overlay',
        showlegend=True,
        legend=dict(x=0.7, y=0.02)
    )

    # Ajouter une ligne verticale au centre (cas de base)
    fig.add_vline(x=0, line=dict(color='black', width=2))

    return fig


def create_scatter_matrix(results: MonteCarloResults,
                         kpi_list: List[str] = None,
                         param_list: List[str] = None) -> go.Figure:
    """
    Crée une matrice de scatter plots pour visualiser les relations.

    Args:
        results: Résultats Monte Carlo
        kpi_list: Liste des KPIs à inclure (si None, prend les principaux)
        param_list: Liste des paramètres à inclure (si None, prend les principaux)

    Returns:
        Figure Plotly
    """
    # Sélectionner les colonnes à afficher
    if kpi_list is None:
        kpi_list = ['LCOH', 'H2', 'H2_waste']

    if param_list is None:
        param_list = ['install_fee', 'OPEX_PEM', 'price_H2']

    # Créer un DataFrame combiné
    combined_df = pd.concat([
        results.parameter_samples[param_list],
        results.kpi_results[kpi_list]
    ], axis=1)

    # Créer la scatter matrix
    fig = px.scatter_matrix(
        combined_df,
        dimensions=param_list + kpi_list,
        color=combined_df['LCOH'],
        color_continuous_scale='RdYlGn_r',
        height=800,
        title="Matrice de corrélations (Scatter Matrix)"
    )

    fig.update_traces(diagonal_visible=False, showupperhalf=False)

    fig.update_layout(
        template="plotly_white",
        font=dict(size=10)
    )

    return fig


def create_correlation_heatmap(results: MonteCarloResults,
                               method: str = 'pearson',
                               kpi_list: List[str] = None) -> go.Figure:
    """
    Crée une heatmap de corrélation entre paramètres et KPIs.

    Args:
        results: Résultats Monte Carlo
        method: Méthode de corrélation ('pearson' ou 'spearman')
        kpi_list: Liste des KPIs à inclure

    Returns:
        Figure Plotly
    """
    if method == 'pearson':
        corr = results.correlations_pearson
    else:
        corr = results.correlations_spearman

    if kpi_list is None:
        kpi_list = ['LCOH', 'H2', 'H2_waste', 'power_waste', 'CAPEX_total', 'OPEX_total']

    # Extraire la sous-matrice paramètres vs KPIs
    param_names = list(results.parameter_samples.columns)
    kpi_names = [k for k in kpi_list if k in corr.columns]

    # Sous-matrice
    corr_subset = corr.loc[param_names, kpi_names]

    fig = go.Figure(data=go.Heatmap(
        z=corr_subset.values,
        x=kpi_names,
        y=param_names,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_subset.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Corrélation")
    ))

    fig.update_layout(
        title=f"Matrice de corrélation ({method.capitalize()})",
        xaxis_title="KPIs",
        yaxis_title="Paramètres",
        template="plotly_white",
        height=600
    )

    return fig


def create_sensitivity_bars(tornado_df: pd.DataFrame,
                           kpi_name: str = 'LCOH',
                           top_n: int = 10) -> go.Figure:
    """
    Crée un bar chart des sensibilités (impact total).

    Args:
        tornado_df: DataFrame avec les données tornado
        kpi_name: Nom du KPI
        top_n: Nombre de paramètres à afficher

    Returns:
        Figure Plotly
    """
    df = tornado_df.head(top_n).copy()
    df = df.sort_values('impact_total', ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df['parameter'],
        x=df['impact_total'],
        orientation='h',
        marker=dict(
            color=df['impact_total'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Impact")
        ),
        text=[f"{v:.4f} ({p:.1f}%)" for v, p in zip(df['impact_total'], df['impact_pct'])],
        textposition='auto'
    ))

    fig.update_layout(
        title=f"Sensibilité des paramètres sur {kpi_name}",
        xaxis_title=f"Impact absolu sur {kpi_name}",
        yaxis_title="Paramètre",
        template="plotly_white",
        height=500
    )

    return fig


def create_box_plots(results: MonteCarloResults,
                    kpi_list: List[str] = None) -> go.Figure:
    """
    Crée des box plots pour plusieurs KPIs.

    Args:
        results: Résultats Monte Carlo
        kpi_list: Liste des KPIs à afficher

    Returns:
        Figure Plotly
    """
    if kpi_list is None:
        kpi_list = ['LCOH', 'H2', 'H2_waste', 'power_waste']

    fig = go.Figure()

    for kpi_name in kpi_list:
        if kpi_name not in results.kpi_results.columns:
            continue

        values = results.kpi_results[kpi_name].dropna()

        fig.add_trace(go.Box(
            y=values,
            name=kpi_name,
            boxmean='sd',  # Afficher moyenne et écart-type
            marker=dict(color='#4472C4'),
            line=dict(color='#1f4788')
        ))

    fig.update_layout(
        title="Distribution des KPIs (Box Plots)",
        yaxis_title="Valeur",
        template="plotly_white",
        height=500,
        showlegend=True
    )

    return fig


def create_violin_plots(results: MonteCarloResults,
                       kpi_list: List[str] = None) -> go.Figure:
    """
    Crée des violin plots pour plusieurs KPIs.

    Args:
        results: Résultats Monte Carlo
        kpi_list: Liste des KPIs à afficher

    Returns:
        Figure Plotly
    """
    if kpi_list is None:
        kpi_list = ['LCOH', 'H2', 'H2_waste']

    fig = go.Figure()

    for kpi_name in kpi_list:
        if kpi_name not in results.kpi_results.columns:
            continue

        values = results.kpi_results[kpi_name].dropna()

        fig.add_trace(go.Violin(
            y=values,
            name=kpi_name,
            box_visible=True,
            meanline_visible=True,
            fillcolor='#4472C4',
            opacity=0.6,
            line=dict(color='#1f4788')
        ))

    fig.update_layout(
        title="Distribution des KPIs (Violin Plots)",
        yaxis_title="Valeur",
        template="plotly_white",
        height=500,
        showlegend=True
    )

    return fig


def create_parameter_impact_comparison(results: MonteCarloResults,
                                       kpi_name: str = 'LCOH') -> go.Figure:
    """
    Compare l'impact de chaque paramètre sur un KPI via les corrélations.

    Args:
        results: Résultats Monte Carlo
        kpi_name: Nom du KPI cible

    Returns:
        Figure Plotly
    """
    param_names = list(results.parameter_samples.columns)

    # Extraire les corrélations avec le KPI cible
    corr_pearson = results.correlations_pearson.loc[param_names, kpi_name]
    corr_spearman = results.correlations_spearman.loc[param_names, kpi_name]

    # Trier par corrélation absolue de Spearman
    sorted_idx = corr_spearman.abs().sort_values(ascending=True).index

    fig = go.Figure()

    # Corrélation de Pearson
    fig.add_trace(go.Bar(
        y=sorted_idx,
        x=corr_pearson[sorted_idx],
        orientation='h',
        name='Pearson (linéaire)',
        marker=dict(color='#4472C4'),
        opacity=0.7
    ))

    # Corrélation de Spearman
    fig.add_trace(go.Bar(
        y=sorted_idx,
        x=corr_spearman[sorted_idx],
        orientation='h',
        name='Spearman (monotone)',
        marker=dict(color='#ED7D31'),
        opacity=0.7
    ))

    fig.update_layout(
        title=f"Corrélation des paramètres avec {kpi_name}",
        xaxis_title="Coefficient de corrélation",
        yaxis_title="Paramètre",
        template="plotly_white",
        height=600,
        barmode='group',
        showlegend=True
    )

    # Ligne verticale à 0
    fig.add_vline(x=0, line=dict(color='black', width=1))

    return fig


def create_percentile_bands(results: MonteCarloResults,
                           kpi_name: str = 'LCOH') -> go.Figure:
    """
    Visualise les bandes de percentiles du KPI.

    Args:
        results: Résultats Monte Carlo
        kpi_name: Nom du KPI

    Returns:
        Figure Plotly
    """
    stats = results.statistics[kpi_name]

    percentiles = [
        ('p5', 'p95', 'P5-P95 (90% CI)', '#E8F4F8'),
        ('p10', 'p90', 'P10-P90 (80% CI)', '#B3D9E6'),
        ('p25', 'p75', 'P25-P75 (50% CI)', '#7FBFD4'),
    ]

    fig = go.Figure()

    # Créer les bandes
    y_pos = 0
    for low_key, high_key, label, color in percentiles:
        fig.add_trace(go.Scatter(
            x=[stats[low_key], stats[high_key]],
            y=[y_pos, y_pos],
            mode='lines',
            line=dict(color=color, width=30),
            name=label,
            showlegend=True,
            hovertemplate=f'{label}<br>Min: {stats[low_key]:.4f}<br>Max: {stats[high_key]:.4f}<extra></extra>'
        ))
        y_pos += 1

    # Ajouter la médiane
    fig.add_trace(go.Scatter(
        x=[stats['median']],
        y=[1],
        mode='markers',
        marker=dict(color='orange', size=15, symbol='diamond'),
        name='Médiane',
        hovertemplate=f"Médiane: {stats['median']:.4f}<extra></extra>"
    ))

    # Ajouter le cas nominal
    fig.add_trace(go.Scatter(
        x=[stats['base_case']],
        y=[1],
        mode='markers',
        marker=dict(color='green', size=15, symbol='star'),
        name='Cas nominal',
        hovertemplate=f"Nominal: {stats['base_case']:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"Bandes de percentiles - {kpi_name}",
        xaxis_title=kpi_name,
        yaxis=dict(showticklabels=False, showgrid=False),
        template="plotly_white",
        height=300,
        showlegend=True
    )

    return fig
