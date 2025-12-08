# -*- coding: utf-8 -*-
"""
Page 2: Modèle et équations
Explication du modèle physique et mathématique
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_power_data, get_statistics, get_monthly_stats, get_hourly_profile
from utils.image_loader import display_image, get_section_images, load_image
from utils.visualizations import create_faraday_efficiency_chart
from config import SYSTEM_PARAMS, GAS_MODEL

st.set_page_config(
    page_title="Modèle - Centrale H2",
    page_icon="📐",
    layout="wide"
)

st.title("Modélisation physique et économique")
st.markdown("---")

# Vue d'ensemble
st.header("Vue d'ensemble du modèle")

col1, col2 = st.columns([3, 2])

st.markdown("""
    Le modèle simule **heure par heure** le fonctionnement d'une centrale de production
    d'hydrogène alimentée par les surplus d'électricité d'un système éolien-nucléaire.

    ### Flux d'énergie et de matière

    ```
    ┌─────────────┐     ┌─────────────┐
    │   Éolien    │────►│             │
    │  (104×3.3MW)│     │   Réseau    │──► Consommation
    └─────────────┘     │ Électrique  │
                        │             │
    ┌─────────────┐     │  Limite:    │     ┌──────────────┐
    │  Nucléaire  │────►│  1.32 GW    │────►│ Électrolyseur│
    │  (1450 MW)  │     │             │     │     PEM      │
    └─────────────┘     └─────────────┘     └──────┬───────┘
                                                   │
                                                   ▼ H2
                                             ┌──────────────┐
                                             │   Stockage   │
                                             │  (250 bar)   │
                                             └──────┬───────┘
                                                    │
                                                    ▼
                                             ┌──────────────┐
                                             │   Camions    │──► Vente
                                             └──────────────┘
    ```
    """)


st.markdown("---")

# Gestion de la puissance
st.header("Modélisation technique")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Centrale éolienne", "Centrale nucléaire", "Centrale hydrogène", "Stokage de l'hydrogène", "Transport de l'hydrogène", "Stratégie de gestion de l'H2"])

# Charger les données
try:
    data = load_power_data()
    hours = np.arange(len(data)) if data is not None else np.array([])
except:
    data = None
    hours = np.array([])


with tab1:
    st.subheader("Centrale éolienne")

    # Caractéristiques en tableau 2x2
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Nombre d'éoliennes", "104")
        st.metric("Capacité unitaire", "3.3 MW")
    with col2:
        st.metric("Type", "Nordex N131 3300")
        st.metric("Capacité totale", "343.2 MW")

    st.markdown("---")

    # Images de l'éolienne et power curve
    st.subheader("Éolienne Nordex N131 3300")

    if not display_image("eolien.png", caption="Éolienne Nordex N131", use_column_width=False, max_width=400):
        st.info("Image de l'éolienne non disponible")

    if not display_image("power_curve_eolien.png", caption="Courbe de puissance de l'éolienne Nordex N131", use_column_width=False, max_width=1000):
        st.info("Courbe de puissance non disponible")

    st.markdown("---")

    # Graphique de production horaire
    st.subheader("Production Horaire du Parc Éolien (8760h)")

    if data is not None:
        wind_mw = data["WP"] / 1000

        # Stats rapides
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Moyenne", f"{wind_mw.mean():.1f} MW")
        with col2:
            st.metric("Max", f"{wind_mw.max():.1f} MW")
        with col3:
            st.metric("Min", f"{wind_mw.min():.1f} MW")

        # Graphique
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours,
            y=wind_mw,
            mode='lines',
            name='Production éolienne',
            line=dict(color='#52b788', width=1),
            fill='tozeroy',
            fillcolor='rgba(82, 183, 136, 0.3)'
        ))

        fig.add_hline(
            y=wind_mw.mean(),
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Moyenne: {wind_mw.mean():.1f} MW"
        )

        fig.update_layout(
            xaxis_title="Heure de l'année",
            yaxis_title="Puissance [MW]",
            hovermode='x unified',
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Données de production éolienne non disponibles")

with tab2:
    st.subheader("Centrale nucléaire")

    # Caractéristiques succinctes
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Capacité", "1450 MW", delta="Oskarshamn 3")
    with col2:
        st.metric("Facteur de charge", "~92%", delta="Très stable")

    st.markdown("---")

    # Graphique de production horaire
    st.subheader("Production horaire de la centrale nucléaire")

    if data is not None:
        nuclear_mw = data["NP"] / 1000

        # Stats rapides
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Moyenne", f"{nuclear_mw.mean():.1f} MW")
        with col2:
            st.metric("Max", f"{nuclear_mw.max():.1f} MW")
        with col3:
            st.metric("Min", f"{nuclear_mw.min():.1f} MW")

        # Graphique
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours,
            y=nuclear_mw,
            mode='lines',
            name='Production Nucléaire',
            line=dict(color='#e76f51', width=1),
            fill='tozeroy',
            fillcolor='rgba(231, 111, 81, 0.3)'
        ))

        fig.add_hline(
            y=nuclear_mw.mean(),
            line_dash="dash",
            line_color="blue",
            annotation_text=f"Moyenne: {nuclear_mw.mean():.1f} MW"
        )

        fig.update_layout(
            xaxis_title="Heure de l'année",
            yaxis_title="Puissance [MW]",
            hovermode='x unified',
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Données de production nucléaire non disponibles")

with tab3:
    st.subheader("Centrale hydrogène")
    
    display_image("methodology-Electrolyseur.png", caption="Modélisation de l'électrolyseur", use_column_width=True)

    st.markdown("""
    ##### 1. Limiteur de puissance 
    La puissance fournie à l'électrolyseur sera toujours inférieure à sa capacité.

    ##### 2. Auxilliaires
    Les auxilliaires consomment 3\% de la puissance fournie.

    ##### 3. Efficacité faradique""")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        L'efficacité faradique $\\eta_F$ représente le rendement de conversion
        électricité → hydrogène. Elle dépend du taux de charge de l'électrolyseur:

        $$\\eta_F = 1 - \\exp\\left(-\\frac{P_{supply}/C}{0.04409}\\right)$$

        Où:
        - $P_{supply}$ : Puissance fournie [kW]
        - $C$ : Capacité nominale [kW]
        - $0.04409$ : Constante caractéristique

        **Interprétation:**
        - À faible charge: efficacité réduite
        - À pleine charge: efficacité ~100%
        """)

    with col2:
        fig = create_faraday_efficiency_chart()
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ##### 4. Coefficient d'amélioration
    Un coefficient appelé coefficient d'amélioration sera utilisé pour rapprocher le modèle 
    de la réalité. Sans ce coefficient, un électrolyseur de 120 kW produirait 3,49 kg d'H2 en 
    une heure, au lieu de 2 kg comme certains électrolyseurs commerciaux. Il est fixé à 2/3,49 , 
    afin de correspondre à la production de ces derniers.
    """)

with tab4:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### Modèle
        Le système de stockage est considéré comme un réservoir parfait (sans fuite), à 250 bars.
        
        """)

    with col2:
        display_image("hydrogentank.png", caption="Réservoir de stockage d'hydrogène", use_column_width=False, max_width=100)

    st.markdown("""
        #### Hypothèses
        - Le rendement du compresseur est fixé à 100 %.
        - La consommation électrique du compresseur est déjà prise en compte dans la perte de puissance due à la consommation des auxiliaires,
        définie dans la partie modélisation PEM,
        - Le calcul est effectué à l'aide de l'équation des gaz parfaits,
        - La pression de l'hydrogène stocké est fixée à 250 bars.
        Données techniques Le système de stockage est considéré comme un réservoir parfait (sans fuite), à 250 bars.
        """)

with tab5:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### Modèle
        - Distance aller simple : 35 km,
        - Temps de déchargement et de repos : 2 heures,
        - Vitesse de conduite : 70 km/h,
        - Temps nécessaire pour un aller-retour : 3 heures,
        - Pression d'hydrogène à la livraison : 250 bars,
        - Densité de l'hydrogène : 17,6 kg/m3,
        - Capacité totale de transport en volume : 29,36 m3,
        - Capacité en masse : 500 kg.
        """)

    with col2:
        display_image("truck.jpg", caption="Remorques-citernes de transport d'hydrogène", use_column_width=False, max_width=100)

    st.markdown("""
        #### Hypothèses
        - L'efficacité de chargement du réservoir vers la remorque-citerne est fixée à 100 %.
        - L'efficacité de déchargement de la remorque-citerne est fixée à 100 %.
        - On suppose que la remorque-citerne est immédiatement disponible pour le trajet suivant dès son retour à l'usine d'hydrogène.
        """)

with tab6:
    display_image("Dispatch_strategy.png", caption="Stratégie de gestion de l'hydrogène", use_column_width=True)

    st.markdown("""
- Initialisation : une liste des indisponibilités et le nombre de camions disponibles sont configurés. Le système suppose qu'aucun hydrogène n'est vendu pendant la première heure de fonctionnement.
- Boucle horaire de simulation : pour chaque heure, la fonction détermine d'abord la quantité d'hydrogène pouvant être stockée selon la capacité disponible. Si l'hydrogène comprimé produit dépasse cette capacité, l'excédent est rejeté et le reste est stocké.
- Vente d'hydrogène : lorsque le stockage atteint un seuil spécifique et que des camions sont disponibles, le système vend de l'hydrogène en remplissant le nombre maximum de camions possible. Les camions utilisés sont marqués comme indisponibles pour un certain nombre d'heures.
- Vérification des camions : la fonction vérifie si des camions actuellement indisponibles redeviennent disponibles à l'heure suivante et met à jour le nombre de camions disponibles.
- Mise à jour continue : la fonction parcourt toutes les heures et gère le stockage et la vente en fonction de la capacité disponible et de la disponibilité des camions. Le système met à jour les quantités d'hydrogène rejeté, stocké et vendu pour chaque heure.
  """)

st.markdown("---")

# Gestion de la puissance
st.header("Modélisation économique")

# Tabulation
tab1, tab2, tab3 = st.tabs(["LCOH", "NPV", "PBP"])

with tab1:
    st.subheader("Coût actualisé de l'hydrogène (LCOH)")

    st.markdown("""
    Le **LCOH** (Levelized Cost of Hydrogen) est le principal indicateur économique utilisé dans le modèle d'optimisation.
    Il représente le coût total actualisé de production de l'hydrogène sur toute la durée de vie du projet.
    """)

    # Formules
    st.markdown("### Formules de calcul")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Structure des coûts:**
        """)
        st.latex(r"CAPEX = CAPEX_{direct} + CAPEX_{indirect}")
        st.latex(r"OPEX = OPEX_{O\&M} + OPEX_{feedstock}")

        st.markdown("""
        - **CAPEX direct**: Coûts de construction initiaux
        - **CAPEX indirect**: Coûts de remplacement des équipements
        - **OPEX**: Maintenance et coûts d'exploitation
        """)

    with col2:
        st.markdown("""
        **Calcul du LCOH:**
        """)
        st.latex(r"LCOH = \frac{CAPEX + \sum_{n=1}^{N} \frac{OPEX}{(1+WACC)^n}}{\sum_{n=1}^{N} \frac{E \cdot (1-\varepsilon)^n}{(1+WACC)^n}}")

        st.markdown("""
        - **N**: Durée de vie du projet (30 ans)
        - **WACC**: 5% (taux d'actualisation)
        - **ε**: 0.3% (dégradation annuelle)
        - **E**: Énergie hydrogène produite (kWh)
        """)

    st.markdown("---")

    # Tableau des coûts
    st.markdown("### Composition des coûts")

    cost_data = {
        "Catégorie": [
            "CAPEX direct",
            "",
            "",
            "",
            "CAPEX indirect",
            "",
            "OPEX O&M",
            "",
            "",
            "",
            "OPEX Feedstock"
        ],
        "Composant": [
            "Électrolyseur PEM",
            "Réservoir de stockage H₂",
            "Compresseur",
            "Camion citerne",
            "Remplacement électrolyseur",
            "Remplacement réservoir",
            "Maintenance électrolyseur",
            "Maintenance réservoir",
            "Maintenance compresseur",
            "Maintenance camion",
            "Eau"
        ],
        "Unité": [
            "€/kW",
            "€/kg",
            "€",
            "€/unité",
            "€/kW/an",
            "€/kg/an",
            "€/kW/an",
            "€/kg/an",
            "€/an",
            "€/unité/an",
            "€/kg"
        ],
        "Valeur": [
            "1,800",
            "490",
            "93,296",
            "610,000",
            "82.9",
            "18.8",
            "54",
            "9.8",
            "4,665",
            "30,500",
            "0.003"
        ]
    }

    df_costs = pd.DataFrame(cost_data)

    st.dataframe(
        df_costs,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("""
    **Notes:**
    - Les coûts de remplacement sont répartis uniformément sur 30 ans
    - Taux de change USD-EUR 2023 : 0.924
    - Les coûts d'O&M incluent la main-d'œuvre
    """)

with tab2:
    st.subheader("Valeur actualisée nette")

    st.markdown("""
    La **VAN** (valeur actualisée nette) ou **NPV** (net present value) est un indicateur économique qui actualise
    les revenus et coûts futurs du projet à la valeur présente. Elle permet d'évaluer la rentabilité du projet
    sur toute sa durée de vie.
    """)

    # Critère de décision
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### ✅ NPV > 0
        - Le taux de rendement dépasse le WACC
        - Le projet est **rentable**
        - Revenus > Coûts sur 33 ans
        - Projet viable financièrement
        """)

    with col2:
        st.markdown("""
        ### ❌ NPV < 0
        - Le taux de rendement est inférieur au WACC
        - Le projet est **non rentable**
        - Revenus < Coûts sur 33 ans
        - Risque d'échec financier
        """)

    st.markdown("---")

    # Formule NPV
    st.markdown("### Formule de calcul")

    st.latex(r"""
    NPV = -\sum_{t=0}^{n_{con}-1} \frac{CAPEX}{n_{con}(1+IRR)^t}
    + \sum_{t=n_{con}}^{n_{con}+n_{op}-1} \frac{\sum_{h=1}^{8760} E_{net,H2} \cdot p_{H2} - OPEX}{(1+IRR)^t}
    + \sum_{t=n_{con}+n_{op}}^{n_{con}+n_{op}+n_{res}-1} \frac{residual\ fee}{n_{res}(1+IRR)^t}
    """)

    st.markdown("---")

    # Paramètres
    st.markdown("### Paramètres du calcul")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Durées:**
        - **n_con**: 2 ans (construction)
        - **n_op**: 30 ans (exploitation)
        - **n_res**: 1 an (démantèlement)
        - **Total**: 33 ans
        """)

    with col2:
        st.markdown("""
        **Revenus:**
        - **E_net,H2**: Énergie H₂ annuelle (kWh)
        - **p_H2**: Prix de vente H₂ = 2.7 €/kg
        - Calculé sur 8760 heures/an
        """)

    with col3:
        st.markdown("""
        **Actualisation:**
        - **IRR**: Taux de rendement interne
        - **WACC**: 5% (référence)
        - Actualisation des flux annuels
        """)

    st.markdown("---")

    # Timeline du projet
    st.markdown("### Chronologie du projet (33 ans)")

    timeline_data = {
        "Phase": ["Construction", "Exploitation", "Démantèlement"],
        "Durée": ["2 ans", "30 ans", "1 an"],
        "Flux financiers": [
            "Sortie: CAPEX réparti",
            "Entrée: Ventes H₂ - OPEX",
            "Neutre (coûts résiduels ignorés)"
        ],
        "Années": ["t = 0-1", "t = 2-31", "t = 32"]
    }

    df_timeline = pd.DataFrame(timeline_data)

    st.dataframe(
        df_timeline,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("""
    **Note importante:**
    - Les coûts de démantèlement des équipements sont **ignorés** dans ce projet
    - En pratique, le propriétaire paie souvent pour la mise au rebut des équipements
    - Ces coûts sont difficiles à estimer avec précision
    """)

    st.info("La VAN est calculée **après** l'optimisation du LCOH pour évaluer la viabilité économique du projet optimal.")


with tab3:
    st.subheader("Période de Retour sur Investissement (PBP)")

    st.markdown("""
    La **PBP** (Payback Period) ou **Période de Retour sur Investissement** est un paramètre économique simple
    qui indique le temps nécessaire pour que les revenus du projet égalisent l'investissement initial.
    """)

    # Avantages et limites
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### ✅ Avantages
        - **Simple à calculer** et à comprendre
        - **Indicateur direct** de l'équilibre financier
        - **Utile pour les décisions** rapides
        - Permet de vérifier la **faisabilité** du projet
        """)

    with col2:
        st.markdown("""
        ### ⚠️ Limites
        - **N'actualise pas** la valeur de l'argent
        - Ignore les flux après le retour
        - Ne considère pas le **WACC**
        - Complément à la VAN et au LCOH
        """)

    st.markdown("---")

    # Formule PBP
    st.markdown("### Formule de calcul")

    st.latex(r"""
    PBP = \frac{CAPEX}{\sum_{h=1}^{8760} E_{net,H2} \cdot p_{H2} - OPEX}
    """)

    st.markdown("""
    - **Numérateur**: Investissement initial total (CAPEX)
    - **Dénominateur**: Revenus annuels nets (Ventes H₂ - OPEX)
    - **Résultat**: Nombre d'années pour récupérer l'investissement
    """)

    st.markdown("---")

    # Interprétation
    st.markdown("### Interprétation des résultats")

    interpretation_data = {
        "PBP": ["< 5 ans", "5-10 ans", "10-20 ans", "> 20 ans"],
        "Évaluation": ["Excellent", "Bon", "Acceptable", "Risqué"],
        "Signification": [
            "Retour très rapide, projet très attractif",
            "Retour rapide, projet rentable",
            "Retour modéré, projet viable si stable",
            "Retour lent, nécessite analyse approfondie (VAN)"
        ]
    }

    df_interpretation = pd.DataFrame(interpretation_data)

    st.dataframe(
        df_interpretation,
        use_container_width=True,
        hide_index=True
    )