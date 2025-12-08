# -*- coding: utf-8 -*-
"""
Module de visualisations Plotly pour l'application Streamlit
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

# Palette de couleurs
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#2ca02c",
    "accent": "#ff7f0e",
    "danger": "#d62728",
    "info": "#17becf",
    "hydrogen": "#00b4d8",
    "nuclear": "#e76f51",
    "wind": "#52b788",
    "grid": "#6c757d",
    "storage": "#9b59b6",
    "sold": "#27ae60",
    "wasted": "#c0392b",
}


def create_power_chart(df: pd.DataFrame, show_all: bool = True) -> go.Figure:
    """
    Crée un graphique des productions de puissance.

    Args:
        df: DataFrame avec colonnes WP, NP, excess_power, etc.
        show_all: Afficher toutes les séries ou seulement les principales

    Returns:
        Figure Plotly
    """
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Production Éolienne", "Production Nucléaire", "Puissance Totale vs Limite Réseau"),
        vertical_spacing=0.08,
        shared_xaxes=True
    )

    # Éolien
    fig.add_trace(
        go.Scatter(
            x=df['hour'],
            y=df['WP'] / 1000,  # MW
            name="Éolien",
            line=dict(color=COLORS['wind'], width=1),
            fill='tozeroy',
            fillcolor=f"rgba(82, 183, 136, 0.3)"
        ),
        row=1, col=1
    )

    # Nucléaire
    fig.add_trace(
        go.Scatter(
            x=df['hour'],
            y=df['NP'] / 1000,  # MW
            name="Nucléaire",
            line=dict(color=COLORS['nuclear'], width=1),
            fill='tozeroy',
            fillcolor=f"rgba(231, 111, 81, 0.3)"
        ),
        row=2, col=1
    )

    # Total + Limite réseau
    fig.add_trace(
        go.Scatter(
            x=df['hour'],
            y=df['total_power'] / 1000,
            name="Production Totale",
            line=dict(color=COLORS['primary'], width=1),
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['hour'],
            y=df['grid_limit'] / 1000,
            name="Limite Réseau",
            line=dict(color=COLORS['danger'], width=2, dash='dash'),
        ),
        row=3, col=1
    )

    # Zone excédentaire
    fig.add_trace(
        go.Scatter(
            x=df['hour'],
            y=df['excess_power'] / 1000,
            name="Excédent",
            fill='tozeroy',
            fillcolor=f"rgba(255, 127, 14, 0.4)",
            line=dict(color=COLORS['accent'], width=0),
        ),
        row=3, col=1
    )

    fig.update_layout(
        height=700,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )

    fig.update_xaxes(title_text="Heure de l'année", row=3, col=1)
    fig.update_yaxes(title_text="Puissance [MW]", row=1, col=1)
    fig.update_yaxes(title_text="Puissance [MW]", row=2, col=1)
    fig.update_yaxes(title_text="Puissance [MW]", row=3, col=1)

    return fig


def create_excess_power_chart(df: pd.DataFrame) -> go.Figure:
    """Crée un graphique de la puissance excédentaire."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['hour'],
        y=df['excess_power'] / 1000,
        name="Puissance Excédentaire",
        fill='tozeroy',
        fillcolor=f"rgba(255, 127, 14, 0.5)",
        line=dict(color=COLORS['accent'], width=1),
    ))

    fig.update_layout(
        title="Puissance Excédentaire (Risque de Congestion)",
        xaxis_title="Heure de l'année",
        yaxis_title="Puissance [MW]",
        template="plotly_white",
        height=400,
    )

    return fig


def create_h2_production_chart(time_series: pd.DataFrame, capacity: float) -> go.Figure:
    """
    Crée un graphique de la production d'hydrogène.

    Args:
        time_series: DataFrame avec les séries temporelles du modèle
        capacity: Capacité de l'électrolyseur [kW]
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Puissance Fournie à l'Électrolyseur", "Production d'Hydrogène"),
        vertical_spacing=0.12,
        shared_xaxes=True
    )

    # Puissance fournie vs excédent
    fig.add_trace(
        go.Scatter(
            x=time_series['hour'],
            y=time_series['excess_power'] / 1000,
            name="Puissance Excédentaire",
            line=dict(color=COLORS['accent'], width=1),
            opacity=0.7
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=time_series['hour'],
            y=time_series['supply_power'] / 1000,
            name="Puissance Fournie",
            fill='tozeroy',
            fillcolor=f"rgba(0, 180, 216, 0.4)",
            line=dict(color=COLORS['hydrogen'], width=1),
        ),
        row=1, col=1
    )

    # Ligne capacité
    fig.add_hline(
        y=capacity / 1000,
        line_dash="dash",
        line_color=COLORS['danger'],
        annotation_text=f"Capacité: {capacity/1000:.1f} MW",
        row=1, col=1
    )

    # Production H2
    fig.add_trace(
        go.Scatter(
            x=time_series['hour'],
            y=time_series['mass_H2'],
            name="H2 Produit",
            fill='tozeroy',
            fillcolor=f"rgba(0, 180, 216, 0.5)",
            line=dict(color=COLORS['hydrogen'], width=1),
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified"
    )

    fig.update_xaxes(title_text="Heure", row=2, col=1)
    fig.update_yaxes(title_text="Puissance [MW]", row=1, col=1)
    fig.update_yaxes(title_text="H2 Produit [kg/h]", row=2, col=1)

    return fig


def create_h2_management_chart(time_series: pd.DataFrame, storage_capacity: float,
                               threshold: float) -> go.Figure:
    """
    Crée un graphique de la gestion de l'hydrogène.
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Stockage d'Hydrogène", "Hydrogène Cumulé"),
        vertical_spacing=0.12,
        shared_xaxes=True
    )

    # Stockage
    fig.add_trace(
        go.Scatter(
            x=time_series['hour'],
            y=time_series['stored'],
            name="H2 Stocké",
            fill='tozeroy',
            fillcolor=f"rgba(155, 89, 182, 0.4)",
            line=dict(color=COLORS['storage'], width=1),
        ),
        row=1, col=1
    )

    # Capacité stockage
    fig.add_hline(
        y=storage_capacity,
        line_dash="solid",
        line_color=COLORS['danger'],
        annotation_text=f"Capacité: {storage_capacity:.0f} m³",
        row=1, col=1
    )

    # Seuil de vente
    fig.add_hline(
        y=storage_capacity * threshold,
        line_dash="dash",
        line_color=COLORS['accent'],
        annotation_text=f"Seuil vente: {threshold*100:.0f}%",
        row=1, col=1
    )

    # Cumulés
    cumul_produced = np.cumsum(time_series['H2_compressed'])
    cumul_wasted = np.cumsum(time_series['wasted'])
    cumul_sold = np.cumsum(time_series['sold'])

    fig.add_trace(
        go.Scatter(
            x=time_series['hour'],
            y=cumul_produced,
            name="H2 Produit (cumulé)",
            line=dict(color=COLORS['hydrogen'], width=2),
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=time_series['hour'],
            y=cumul_sold,
            name="H2 Vendu (cumulé)",
            line=dict(color=COLORS['sold'], width=2),
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=time_series['hour'],
            y=cumul_wasted,
            name="H2 Perdu (cumulé)",
            line=dict(color=COLORS['wasted'], width=2),
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified"
    )

    fig.update_xaxes(title_text="Heure", row=2, col=1)
    fig.update_yaxes(title_text="Volume [m³]", row=1, col=1)
    fig.update_yaxes(title_text="Volume Cumulé [m³]", row=2, col=1)

    return fig


def create_ga_convergence_chart(history: Dict) -> go.Figure:
    """
    Crée un graphique de convergence de l'algorithme génétique.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Évolution du Meilleur LCOH",
            "Fitness Moyen ± Écart-type",
            "Diversité Génétique",
            "Temps d'Exécution par Génération"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    generations = history['generations']

    # Best fitness
    fig.add_trace(
        go.Scatter(
            x=generations,
            y=history['best_fitness'],
            name="Meilleur LCOH",
            mode='lines+markers',
            line=dict(color=COLORS['primary'], width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )

    # Mean ± std
    mean = np.array(history['mean_fitness'])
    std = np.array(history['std_fitness'])

    fig.add_trace(
        go.Scatter(
            x=generations,
            y=mean,
            name="Moyenne",
            mode='lines',
            line=dict(color=COLORS['secondary'], width=2),
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=generations + generations[::-1],
            y=list(mean + std) + list((mean - std)[::-1]),
            fill='toself',
            fillcolor='rgba(44, 160, 44, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name="± Écart-type",
            showlegend=False
        ),
        row=1, col=2
    )

    # Diversité
    fig.add_trace(
        go.Scatter(
            x=generations,
            y=history['population_diversity'],
            name="Diversité",
            mode='lines+markers',
            line=dict(color=COLORS['accent'], width=2),
            marker=dict(size=5)
        ),
        row=2, col=1
    )

    # Temps d'exécution
    fig.add_trace(
        go.Bar(
            x=generations,
            y=history['execution_times'],
            name="Temps (s)",
            marker_color=COLORS['info']
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=600,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05)
    )

    fig.update_xaxes(title_text="Génération", row=2, col=1)
    fig.update_xaxes(title_text="Génération", row=2, col=2)
    fig.update_yaxes(title_text="LCOH [€/kWh]", row=1, col=1)
    fig.update_yaxes(title_text="LCOH [€/kWh]", row=1, col=2)
    fig.update_yaxes(title_text="Diversité", row=2, col=1)
    fig.update_yaxes(title_text="Temps [s]", row=2, col=2)

    return fig


def create_ga_live_chart(history: Dict, current_gen: int) -> go.Figure:
    """
    Crée un graphique de progression live pour l'AG.
    """
    fig = go.Figure()

    if history['generations']:
        # Best fitness
        fig.add_trace(go.Scatter(
            x=history['generations'],
            y=history['best_fitness'],
            name="Meilleur LCOH",
            mode='lines+markers',
            line=dict(color=COLORS['primary'], width=3),
            marker=dict(size=8)
        ))

        # Mean fitness
        fig.add_trace(go.Scatter(
            x=history['generations'],
            y=history['mean_fitness'],
            name="LCOH Moyen",
            mode='lines',
            line=dict(color=COLORS['secondary'], width=2, dash='dot'),
        ))

    fig.update_layout(
        title=f"Progression de l'Optimisation - Génération {current_gen}",
        xaxis_title="Génération",
        yaxis_title="LCOH [€/kWh]",
        template="plotly_white",
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    return fig


def create_population_scatter(population: List[Dict]) -> go.Figure:
    """
    Crée un scatter plot de la population actuelle.
    """
    df = pd.DataFrame(population)

    fig = px.scatter(
        df,
        x='electrolyzer_capacity',
        y='storage_capacity',
        size='number_of_trucks',
        color='LCOH',
        color_continuous_scale='RdYlGn_r',
        hover_data=['threshold', 'LCOH'],
        title="Distribution de la Population"
    )

    fig.update_layout(
        xaxis_title="Capacité Électrolyseur [kW]",
        yaxis_title="Capacité Stockage [m³]",
        template="plotly_white",
        height=400
    )

    return fig


def create_kpi_comparison_chart(kpis_before: Dict, kpis_after: Dict) -> go.Figure:
    """
    Crée un graphique de comparaison avant/après optimisation.
    """
    categories = ['LCOH', 'H2 Vendu', 'Pertes Puissance', 'Pertes H2']

    before_values = [
        kpis_before.get('LCOH', 0),
        kpis_before.get('H2_sold', 0) / 1e6,
        kpis_before.get('wasted_power', 0) * 100,
        kpis_before.get('wasted_hydrogen', 0) * 100
    ]

    after_values = [
        kpis_after.get('LCOH', 0),
        kpis_after.get('H2_sold', 0) / 1e6,
        kpis_after.get('wasted_power', 0) * 100,
        kpis_after.get('wasted_hydrogen', 0) * 100
    ]

    fig = go.Figure(data=[
        go.Bar(name='Avant', x=categories, y=before_values, marker_color=COLORS['danger']),
        go.Bar(name='Après', x=categories, y=after_values, marker_color=COLORS['secondary'])
    ])

    fig.update_layout(
        barmode='group',
        title="Comparaison Avant/Après Optimisation",
        template="plotly_white",
        height=400
    )

    return fig


def create_sensitivity_heatmap(results: pd.DataFrame, x_param: str, y_param: str,
                               z_metric: str = 'LCOH') -> go.Figure:
    """
    Crée une heatmap pour l'analyse de sensibilité.
    """
    pivot = results.pivot_table(values=z_metric, index=y_param, columns=x_param)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn_r',
        colorbar=dict(title=z_metric)
    ))

    fig.update_layout(
        title=f"Analyse de Sensibilité: {z_metric}",
        xaxis_title=x_param,
        yaxis_title=y_param,
        template="plotly_white",
        height=500
    )

    return fig


def create_capex_breakdown_chart(kpis: Dict) -> go.Figure:
    """
    Crée un graphique de répartition du CAPEX détaillé.
    Reproduit le camembert CAPEX Breakdown de l'image de référence.
    """
    # Récupérer les valeurs CAPEX
    capex_pem = kpis.get('CAPEX_PEM', 0)
    capex_storage = kpis.get('CAPEX_storage', 0)
    capex_selling = kpis.get('CAPEX_selling', 0)

    # Détail du CAPEX de vente (approximatif basé sur les composants)
    # On suppose que CAPEX_selling inclut compresseur, tube trailer, et remplacement H2 tank
    capex_compressor = capex_selling * 0.275  # Approximation
    capex_tube_trailer = capex_selling * 0.011  # Approximation
    capex_h2_tank_replacement = capex_selling * 0.009  # Approximation
    capex_pem_replacement = capex_pem * 0.0656  # Approximation

    # Composants principaux du CAPEX
    capex_pem_construction = capex_pem - capex_pem_replacement

    labels = [
        'PEM technology construction cost',
        'Hydrogen storage tank construction cost',
        'Compressor',
        'Tube trailer',
        'PEM electrolyser replacement cost',
        'Hydrogen tank replacement cost'
    ]

    values = [
        capex_pem_construction,
        capex_storage,
        capex_compressor,
        capex_tube_trailer,
        capex_pem_replacement,
        capex_h2_tank_replacement
    ]

    # Couleurs personnalisées pour correspondre à l'image
    colors = ['#4472C4', '#ED7D31', '#FFC000', '#A5A5A5', '#5B9BD5', '#70AD47']

    # Calculer les pourcentages arrondis au centième
    total = sum(values)
    percentages = [f'{(v/total*100):.2f}%' for v in values]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        textposition='auto',
        text=percentages,
        textinfo='text',
        hovertemplate='%{label}<br>%{value:.2f} €<br>%{percent}<extra></extra>'
    )])

    fig.update_layout(
        title=dict(
            text="CAPEX Breakdown",
            font=dict(size=16, family="Arial")
        ),
        template="plotly_white",
        height=400,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=-0.3,
            xanchor="left",
            x=0.0,
            font=dict(size=10)
        )
    )

    return fig


def create_opex_breakdown_chart(kpis: Dict) -> go.Figure:
    """
    Crée un graphique de répartition de l'OPEX détaillé.
    Reproduit le camembert OPEX Breakdown de l'image de référence.
    """
    # Récupérer les valeurs OPEX
    opex_pem = kpis.get('OPEX_PEM', 0)
    opex_water = kpis.get('OPEX_water', 0)
    opex_selling = kpis.get('OPEX_selling', 0)
    opex_compressor = kpis.get('compressor_opex', 4665)  # Valeur par défaut

    # Détail de l'OPEX
    labels = [
        'PEM-electrolyser OM cost',
        'Hydrogen-tank-OM-cost',
        'Compressor OM cost',
        'Tube trailer OM cost',
        'Water cost'
    ]

    # Approximations basées sur les proportions de l'image
    opex_h2_tank = opex_selling * 0.168  # 1.80% du total
    opex_tube_trailer = opex_selling * 0.014  # 0.15% du total

    values = [
        opex_pem,
        opex_h2_tank,
        opex_compressor,
        opex_tube_trailer,
        opex_water
    ]

    # Couleurs personnalisées pour correspondre à l'image
    colors = ['#4472C4', '#ED7D31', '#FFC000', '#A5A5A5', '#5B9BD5']

    # Calculer les pourcentages arrondis au centième
    total = sum(values)
    percentages = [f'{(v/total*100):.2f}%' for v in values]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        textposition='auto',
        text=percentages,
        textinfo='text',
        hovertemplate='%{label}<br>%{value:.2f} €/year<br>%{percent}<extra></extra>'
    )])

    fig.update_layout(
        title=dict(
            text="OPEX Breakdown",
            font=dict(size=16, family="Arial")
        ),
        template="plotly_white",
        height=400,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=-0.3,
            xanchor="right",
            x=1.0,
            font=dict(size=10)
        )
    )

    return fig


def create_monthly_profile_chart(df: pd.DataFrame) -> go.Figure:
    """
    Crée un graphique des profils mensuels.
    """
    monthly = df.groupby('month').agg({
        'WP': 'mean',
        'NP': 'mean',
        'excess_power': 'mean'
    }).reset_index()

    months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=months,
        y=monthly['WP'] / 1000,
        name='Éolien',
        marker_color=COLORS['wind']
    ))

    fig.add_trace(go.Bar(
        x=months,
        y=monthly['NP'] / 1000,
        name='Nucléaire',
        marker_color=COLORS['nuclear']
    ))

    fig.add_trace(go.Scatter(
        x=months,
        y=monthly['excess_power'] / 1000,
        name='Excédent',
        mode='lines+markers',
        line=dict(color=COLORS['accent'], width=3),
        yaxis='y2'
    ))

    fig.update_layout(
        title="Profil Mensuel de Production",
        xaxis_title="Mois",
        yaxis_title="Puissance Moyenne [MW]",
        yaxis2=dict(title="Excédent [MW]", overlaying='y', side='right'),
        barmode='group',
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    return fig


def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Crée une heatmap de corrélation.
    """
    cols = ['WP', 'NP', 'total_power', 'excess_power']
    corr = df[cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=['Éolien', 'Nucléaire', 'Total', 'Excédent'],
        y=['Éolien', 'Nucléaire', 'Total', 'Excédent'],
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 12},
        colorbar=dict(title="Corrélation")
    ))

    fig.update_layout(
        title="Matrice de Corrélation",
        template="plotly_white",
        height=400
    )

    return fig


def create_faraday_efficiency_chart() -> go.Figure:
    """
    Crée un graphique de l'efficacité faradique.
    """
    ratio = np.linspace(0, 1, 100)
    eta_F_characteristic = 0.04409448818
    eta_F = 1 - np.exp(-ratio / eta_F_characteristic)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ratio * 100,
        y=eta_F * 100,
        mode='lines',
        name='η_F',
        line=dict(color=COLORS['hydrogen'], width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 180, 216, 0.3)'
    ))

    fig.update_layout(
        title="Efficacité Faradique de l'Électrolyseur",
        xaxis_title="Charge (P/C) [%]",
        yaxis_title="Efficacité Faradique η_F [%]",
        template="plotly_white",
        height=400
    )

    # Annotations
    fig.add_annotation(
        x=50, y=100 * (1 - np.exp(-0.5 / eta_F_characteristic)),
        text="Zone de fonctionnement optimal",
        showarrow=True,
        arrowhead=2
    )

    return fig


def create_3d_population_scatter(iteration_data: dict, all_iterations: list = None, bounds: dict = None) -> go.Figure:
    """
    Crée un scatter plot 3D de la population de l'algorithme génétique.

    Args:
        iteration_data: Dictionnaire contenant:
            - 'population': Liste de [C, S, N] pour chaque membre
            - 'lcoh_values': Liste des LCOH correspondants
            - 'best_member': [C, S, N] du meilleur membre
            - 'best_lcoh': LCOH du meilleur
            - 'iteration': Numéro de l'itération
        all_iterations: Liste complète de toutes les itérations pour calculer les échelles fixes
        bounds: Dictionnaire des bornes {'C': (min, max), 'S': (min, max), 'N': (min, max)}

    Returns:
        Figure Plotly 3D interactive
    """
    population = iteration_data['population']
    lcoh_values = iteration_data['lcoh_values']
    best_member = iteration_data['best_member']
    best_lcoh = iteration_data['best_lcoh']

    # Extraire C, S, N de la population
    C_values = [member[0] for member in population]
    S_values = [member[1] for member in population]
    N_values = [member[2] for member in population]

    # Définir les échelles selon les priorités : bounds > all_iterations > valeurs courantes
    if bounds:
        # Utiliser les bornes des variables d'optimisation
        C_range = list(bounds['C'])
        S_range = list(bounds['S'])
        N_range = list(bounds['N'])
        LCOH_range = [min(lcoh_values), max(lcoh_values)]
    elif all_iterations:
        # Calculer les échelles à partir de toutes les itérations
        all_C = [m[0] for it in all_iterations for m in it['population']]
        all_S = [m[1] for it in all_iterations for m in it['population']]
        all_N = [m[2] for it in all_iterations for m in it['population']]
        all_LCOH = [l for it in all_iterations for l in it['lcoh_values']]

        C_range = [min(all_C), max(all_C)]
        S_range = [min(all_S), max(all_S)]
        N_range = [min(all_N), max(all_N)]
        LCOH_range = [min(all_LCOH), max(all_LCOH)]
    else:
        # Utiliser les valeurs de l'itération courante
        C_range = [min(C_values), max(C_values)]
        S_range = [min(S_values), max(S_values)]
        N_range = [min(N_values), max(N_values)]
        LCOH_range = [min(lcoh_values), max(lcoh_values)]

    # Créer la figure
    fig = go.Figure()

    # Trace 1: Population complète
    fig.add_trace(go.Scatter3d(
        x=C_values,
        y=S_values,
        z=N_values,
        mode='markers',
        marker=dict(
            size=6,
            color=lcoh_values,
            colorscale='RdYlGn_r',  # Rouge (mauvais) -> Vert (bon)
            colorbar=dict(title="LCOH [€/kWh]", x=-0.15),
            cmin=LCOH_range[0],  # Échelle fixe pour LCOH
            cmax=LCOH_range[1],
            showscale=True,
            opacity=0.7,
            line=dict(width=0.5, color='white')
        ),
        text=[f'C={c:,.0f} kW<br>S={s:.0f} m³<br>N={n}<br>LCOH={l:.4f} €/kWh'
              for c, s, n, l in zip(C_values, S_values, N_values, lcoh_values)],
        hovertemplate='%{text}<extra></extra>',
        name='Population'
    ))

    # Trace 2: Meilleur membre (marqueur spécial)
    best_hover_text = f'<b>MEILLEUR</b><br>C={best_member[0]:,.0f} kW<br>S={best_member[1]:.0f} m³<br>N={best_member[2]}<br>LCOH={best_lcoh:.4f} €/kWh'

    fig.add_trace(go.Scatter3d(
        x=[best_member[0]],
        y=[best_member[1]],
        z=[best_member[2]],
        mode='markers',
        marker=dict(
            size=15,
            symbol='diamond',
            color='gold',
            line=dict(color='black', width=2)
        ),
        hovertext=best_hover_text,
        hoverinfo='text',
        name='Meilleur'
    ))

    # Layout avec échelles fixes et préservation de la position caméra
    fig.update_layout(
        title=f"Population de l'itération {1+iteration_data['iteration']}",
        scene=dict(
            xaxis=dict(
                title='Capacité électrolyseur [kW]',
                range=C_range
            ),
            yaxis=dict(
                title='Capacité stockage [m³]',
                range=S_range
            ),
            zaxis=dict(
                title='Nombre de camions',
                range=N_range
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        height=700,
        showlegend=True,
        template='plotly_white',
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=0, r=0, t=50, b=0),
        uirevision='constant'  # Préserve la position de la caméra
    )

    return fig
