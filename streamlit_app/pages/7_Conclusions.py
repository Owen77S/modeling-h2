# -*- coding: utf-8 -*-
"""
Page 7: Conclusions et Perspectives
Synthèse du projet et prochaines étapes
"""

import streamlit as st
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_loader import display_image, get_available_images
from utils.visualizations import COLORS
from config import OPTIMAL_DESIGN, ECONOMICS

st.set_page_config(
    page_title="Conclusions - Centrale H2",
    page_icon="🎯",
    layout="wide"
)

st.title("Conclusions et perspectives")
st.markdown("---")

# Résumé des résultats
st.header("Synthèse des résultats")


st.subheader("Performances atteintes")

st.markdown(f"""
    | KPI | Valeur | Objectif |
    |-----|--------|----------|
    | **LCOH** | 0.145 €/kWh | Minimiser |
    | **H2 Annuel** | 3.0 kt | Maximiser |
    | **Pertes Puissance** | {OPTIMAL_DESIGN['wasted_power']*100:.1f}% | < 80% ✅ |
    | **Pertes H2** | {OPTIMAL_DESIGN['wasted_hydrogen']*100:.1f}% | < 80% ✅ |
    """)

st.markdown("---")

# Discussion
st.header("Discussion")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    ### Point de vue de la centrale hydrogène

    Du point de vue de la centrale hydrogène, il peut être **bénéfique d'installer une centrale de production d'hydrogène**
    lorsque la différence entre la capacité des centrales électriques et la limite du réseau est élevée.

    **Cependant**, d'un point de vue plus large, il peut être **plus pertinent de dimensionner correctement les centrales
    électriques** en fonction de la limite du réseau, car l'ajout d'une centrale hydrogène devrait être considéré comme
    **un moyen d'améliorer l'efficacité** des centrales électriques, plutôt que comme un moyen de résoudre des problèmes
    de surdimensionnement.
    """)

with col2:
    st.markdown("""
    ### Cas de l'hybridation nucléaire-éolien

    Dans le cas de l'hybridation d'une centrale nucléaire avec un parc éolien, **collecter la puissance excédentaire
    avec une centrale hydrogène peut valoir le coup**, en particulier si la centrale est de **grande taille**.

    **Avantages :**
    - Valorisation de l'électricité excédentaire
    - Production d'hydrogène vert
    - Amélioration de l'efficacité globale du système
    - Réduction des risques de congestion du réseau
    """)

st.markdown("---")

# Aspects de durabilité
st.header("Aspects de durabilité")

st.markdown("""
L'hydrogène est de plus en plus utilisé dans la transition énergétique. Cependant, l'expansion de la production
d'hydrogène a également des **conséquences environnementales et sociales**.

Le projet est situé près de la centrale nucléaire d'Oskarshamn en Suède, à proximité de la mer et de la forêt.
Il est nécessaire d'explorer les **impacts du projet** sur l'environnement, les ressources en eau, les terres et
les objectifs de développement durable (ODD).
""")

st.subheader("Contribution aux Objectifs de Développement Durable (ODD)")

# Créer des cartes pour les ODD
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='background: #4C9FEB; padding: 15px; border-radius: 10px; color: white;'>
        <h4>🚰 ODD 6 - Eau propre et assainissement</h4>
        <p style='font-size: 0.9em;'>
        <b>Impact :</b> La production d'hydrogène nécessite la consommation de grandes quantités d'eau.
        Bien que la Suède dispose de ressources en eau abondantes, <b>la menace potentielle</b> pour les ressources
        en eau de la production d'hydrogène doit être prise en compte.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    st.markdown("""
    <div style='background: #56C02B; padding: 15px; border-radius: 10px; color: white;'>
        <h4>⚡ ODD 13 - Lutte contre les changements climatiques</h4>
        <p style='font-size: 0.9em;'>
        <b>Impact positif :</b> En tant que ressource propre, la combustion de l'hydrogène ne produit
        <b>aucun gaz nocif pour l'environnement</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background: #FCC30B; padding: 15px; border-radius: 10px; color: white;'>
        <h4>💡 ODD 7 - Énergie propre et abordable</h4>
        <p style='font-size: 0.9em;'>
        <b>Double avantage :</b>
        <br>1. Vendre l'hydrogène augmente les revenus des producteurs d'électricité et réduit le gaspillage
        <br>2. L'hydrogène peut être utilisé pour produire de l'électricité en cas de manque, facilitant l'accès à plus d'électricité
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='background: #FD6925; padding: 15px; border-radius: 10px; color: white;'>
        <h4>🏭 ODD 9 - Industrie, innovation et infrastructure</h4>
        <p style='font-size: 0.9em;'>
        <b>Impact positif :</b> L'expansion des centrales hydrogène pourrait faciliter le développement
        d'<b>infrastructures industrielles scientifiques et technologiques</b> pour la production d'hydrogène.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    st.markdown("""
    <div style='background: #3F7E44; padding: 15px; border-radius: 10px; color: white;'>
        <h4>🌲 ODD 15 - Vie terrestre</h4>
        <p style='font-size: 0.9em;'>
        <b>Préoccupation :</b> L'expansion des centrales de production d'hydrogène peut entrer en conflit avec
        l'utilisation des terres. La centrale est entourée de forêts, et <b>les dommages potentiels</b> aux terres
        forestières et l'impact sur les animaux vivant dans les forêts doivent être pris en compte.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Points forts du projet

    **1. Modélisation Complète**
    - Simulation horaire réaliste (8760 h)
    - Prise en compte de tous les flux physiques
    - Intégration des contraintes économiques

    **2. Optimisation Efficace**
    - Algorithme génétique robuste
    - Convergence rapide vers l'optimum
    - Gestion de la stagnation

    **3. Analyse Approfondie**
    - Sensibilité multi-paramètres
    - Visualisations interactives
    - Données exploitables
    """)

with col2:
    st.markdown("""
    ### Résultats majeurs

    **1. LCOH compétitif**
    - 0.145 €/kWh
    - Proche des objectifs EU 2030
    - Viable économiquement

    **2. Valorisation efficace**
    - 98% de l'H2 produit est vendu
    - Gestion optimale du stockage
    - Logistique de transport adaptée

    **3. Dimensionnement équilibré**
    - Électrolyseur adapté aux excédents
    - Stockage suffisant sans surdimensionnement
    - Flotte de camions optimisée
    """)

st.markdown("---")

# Limitations
st.header("Limitations et hypothèses")

st.markdown("""
### Simplifications du modèle

| Aspect | Simplification | Impact potentiel |
|--------|----------------|------------------|
| **Dégradation** | Non modélisée dynamiquement | Sous-estimation LCOH à long terme |
| **Prix H2** | Fixe à 2.7 €/kg | Sensible aux fluctuations du marché |
| **Réseau** | Limite constante | Réalité plus variable |
| **Météo** | Données d'une année | Variabilité inter-annuelle |
| **Maintenance** | OPEX simplifié | Arrêts non planifiés ignorés |
| **Compression** | Polytropique idéale | Pertes réelles supérieures |

### Hypothèses fortes

1. **Disponibilité 100%** de l'électrolyseur (hors maintenance planifiée)
2. **Pas de contrainte de raccordement** pour l'électrolyseur
3. **Marché H2 garanti** - toute production vendue
""")

st.markdown("---")

# Perspectives
st.header("🔮 Perspectives et Améliorations")

tab1, tab2 = st.tabs(["Améliorations techniques", "Extensions du modèle"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        - **Optimisation multi-objectif**
          - Pareto LCOH vs Production
          - Trade-offs visuels

        - **Parallélisation**
          - Évaluation multi-process pour réduction de temps de calcul
        """)

    with col2:
        st.markdown("""
        - **Intégration données réelles**
          - API météo temps réel
          - Prix marché dynamiques
          - Profils de demande

        - **Machine Learning**
          - Surrogate models
          - Prédiction de performance
        """)

with tab2:
    st.markdown("""
    ### Extensions possibles du modèle

    **1. Modélisation plus fine de l'électrolyseur** 
    - **Courbe de rendement complète** : La puissance fournie à l'électrolyseur n'est pas constante,
      ce qui aura nécessairement un impact sur son efficacité et sa durée de vie
    - **Dégradation dynamique** : Le modèle actuel ne considère pas les principales pertes du dispositif
    - **Temps de démarrage/arrêt** : Impact sur la production réelle
    - **Modes de fonctionnement** : standby, hot standby

    **2. Stockage avancé**
    - **Développer un modèle propre** pour le stockage d'hydrogène
    - Différentes technologies (réservoirs, cavernes)
    - Pertes de stockage (boil-off)
    - Coûts différenciés selon la technologie

    **3. Transport multi-modal**
    - **Pipelines** : Pour livrer l'hydrogène gazeux si la production et la demande sont élevées
    - Les pipelines d'hydrogène sont très courants dans les régions à forte demande (comme le Gulf Coast)
    - Méthode rentable pour la livraison à grande échelle s'il existe des pipelines
    - Différents types de camions et optimisation des routes

    **4. Analyse de sensibilité étendue**
    - **Analyse de sensibilité sur les deux contraintes** (électricité et hydrogène gaspillés)
      pour voir comment elles impactent les KPIs
    - Contraintes variables dans le temps

    **5. Loi empirique**
    - **Obtenir une loi empirique** pour calculer les différentes variables optimisées
    - Permettrait un dimensionnement rapide sans optimisation complète


    """)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h3>Pour toute question technique ou collaboration, n'hésitez pas à me contacter.</h3>
    <p>Analyse techno-économique d'une centrale à hydrogène optimisée pour réduire les risques de congestion</p>
</div>
""", unsafe_allow_html=True)
