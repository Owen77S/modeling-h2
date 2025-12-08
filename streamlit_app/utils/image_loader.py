# -*- coding: utf-8 -*-
"""
Module de chargement des images pour l'application Streamlit
"""

import streamlit as st
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple
import os


# Chemins des dossiers d'images
IMAGE_DIRS = [
    Path(__file__).parent.parent.parent.parent / "images",
    Path(__file__).parent.parent.parent.parent / "images_ia",
    Path(__file__).parent.parent / "assets",
]

# Extensions supportées
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.svg', '.gif', '.webp'}


@st.cache_data
def get_available_images() -> Dict[str, Path]:
    """
    Scanne les dossiers d'images et retourne un dictionnaire des images disponibles.

    Returns:
        Dict mapping nom_image -> chemin complet
    """
    images = {}

    for img_dir in IMAGE_DIRS:
        if img_dir.exists():
            # Scanner récursivement
            for path in img_dir.rglob("*"):
                if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    # Utiliser le nom relatif comme clé
                    try:
                        rel_path = path.relative_to(img_dir)
                        key = str(rel_path).replace("\\", "/")
                        images[key] = path

                        # Aussi ajouter juste le nom du fichier
                        images[path.name] = path
                    except ValueError:
                        pass

    return images


def load_image(name: str, max_width: Optional[int] = None) -> Optional[Image.Image]:
    """
    Charge une image par son nom.

    Args:
        name: Nom ou chemin de l'image
        max_width: Largeur maximale (redimensionne si nécessaire)

    Returns:
        Image PIL ou None si non trouvée
    """
    images = get_available_images()
    # Chercher l'image
    path = None
    if name in images:
        path = images[name]
    else:
        # Essayer de trouver une correspondance partielle
        for key, p in images.items():
            if name.lower() in key.lower():
                path = p
                break

    if path is None or not path.exists():
        return None

    try:
        img = Image.open(path)

        # Redimensionner si nécessaire
        if max_width and img.width > max_width:
            ratio = max_width / img.width
            new_size = (max_width, int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        return img

    except Exception as e:
        st.warning(f"Impossible de charger l'image {name}: {e}")
        return None


def display_image(name: str, caption: Optional[str] = None,
                  max_width: Optional[int] = None,
                  use_column_width: bool = False) -> bool:
    """
    Affiche une image dans Streamlit.

    Args:
        name: Nom ou chemin de l'image
        caption: Légende optionnelle
        max_width: Largeur maximale
        use_column_width: Utiliser la largeur de la colonne

    Returns:
        True si l'image a été affichée, False sinon
    """
    img = load_image(name, max_width)

    if img is not None:
        st.image(img, caption=caption, use_container_width=use_column_width)
        return True
    return False


def get_images_in_folder(folder: str) -> List[str]:
    """
    Retourne la liste des images dans un sous-dossier spécifique.

    Args:
        folder: Nom du sous-dossier

    Returns:
        Liste des noms d'images
    """
    images = get_available_images()
    return [name for name in images.keys() if folder.lower() in name.lower()]


def create_image_gallery(names: List[str], cols: int = 3,
                        captions: Optional[List[str]] = None) -> None:
    """
    Crée une galerie d'images.

    Args:
        names: Liste des noms d'images
        cols: Nombre de colonnes
        captions: Légendes optionnelles
    """
    if captions is None:
        captions = names

    columns = st.columns(cols)

    for i, (name, caption) in enumerate(zip(names, captions)):
        with columns[i % cols]:
            display_image(name, caption=caption, use_column_width=True)


# Mapping des images par section
IMAGE_MAPPING = {
    "introduction": [
        "layout.png",
        "system_layout.png",
        "presentation.png",
    ],
    "model": [
        "electrolyseur_model_1.png",
        "schema_PEM.jpg",
        "eta_F.png",
        "eta_F.svg",
    ],
    "methodology": [
        "methodlogy chart.png",
        "methodology_chart.png",
        "methodlogy chart without benchmarking.png",
    ],
    "power_production": [
        "power prod/WP.png",
        "power prod/NP.png",
        "power prod/EP.png",
        "power prod/GL.png",
        "renewable_ninja_spec.png",
    ],
    "optimization": [
        "optimisation best design/hydrogen_prod.png",
        "optimisation best design/hydrogen_managemetn.png",
        "optimisation best design/step1.png",
        "optimisation best design/step2.png",
        "optimisation best design/step3.png",
    ],
    "sensitivity_grid": [
        "grid limit/1.png",
        "grid limit/2.png",
    ],
    "sensitivity_wind": [
        "wind capacity/",
    ],
    "h2_plant": [
        "H2_plant_modelisation.png",
    ],
}


def get_section_images(section: str) -> List[str]:
    """
    Retourne les images associées à une section.

    Args:
        section: Nom de la section

    Returns:
        Liste des noms d'images disponibles pour cette section
    """
    if section not in IMAGE_MAPPING:
        return []

    available = get_available_images()
    result = []

    for img_name in IMAGE_MAPPING[section]:
        # Chercher l'image exacte ou une correspondance partielle
        if img_name in available:
            result.append(img_name)
        else:
            # Chercher correspondance partielle
            for key in available.keys():
                if img_name.lower() in key.lower():
                    result.append(key)
                    break

    return result


def display_section_images(section: str, cols: int = 2) -> None:
    """
    Affiche toutes les images d'une section.
    """
    images = get_section_images(section)

    if not images:
        return

    with st.expander("📸 Images associées", expanded=False):
        create_image_gallery(images, cols=cols)
