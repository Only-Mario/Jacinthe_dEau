import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import os
from datetime import datetime
import pandas as pd
import base64
from io import BytesIO
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Pour éviter les problèmes de thread avec Streamlit

# Fonction pour détecter la jacinthe d'eau (gardée inchangée)
def detecter_jacinthe(image, visualiser=True, sauvegarder=True, dossier_sortie="resultats", resolution_m2_px=None):
    """
    Détecte la jacinthe d'eau avec des paramètres optimisés pour des images de drone.
    
    Paramètres:
    - image: Image d'entrée au format OpenCV (BGR)
    - visualiser: Afficher ou non les résultats
    - sauvegarder: Sauvegarder ou non les résultats
    - dossier_sortie: Dossier où sauvegarder les résultats
    - resolution_m2_px: Résolution en mètres carrés par pixel (si None, pas de calcul de surface)
    
    Retourne:
    - resultats: Dictionnaire contenant les statistiques de détection
    - masque_dilate: Masque binaire des jacinthes détectées
    - overlay: Image originale avec superposition des détections
    - contours_filtres: Liste des contours détectés
    """
    # Enregistrement des dimensions originales
    hauteur, largeur = image.shape[:2]

    # Redimensionnement si l'image est trop grande pour accélérer le traitement
    max_dimension = 1500
    scale = 1.0
    if max(hauteur, largeur) > max_dimension:
        scale = max_dimension / max(hauteur, largeur)
        width = int(largeur * scale)
        height = int(hauteur * scale)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    # Conversion BGR vers RGB pour l'affichage
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Réduction du bruit
    image_debruitee = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Conversion en LAB pour meilleure détection des verts
    lab = cv2.cvtColor(image_debruitee, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Amélioration du contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Reconstruction de l'image LAB
    lab_clahe = cv2.merge((l, a, b))
    image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Conversion en HSV
    hsv = cv2.cvtColor(image_clahe, cv2.COLOR_BGR2HSV)

    # Plages HSV pour la jacinthe d'eau
    lower_hyacinth1 = np.array([35, 50, 50])  # Jeune jacinthe (vert clair)
    upper_hyacinth1 = np.array([85, 255, 255])

    lower_hyacinth2 = np.array([85, 30, 50])  # Jacinthe mature (vert foncé)
    upper_hyacinth2 = np.array([95, 255, 255])

    lower_hyacinth_flower = np.array([130, 40, 50])  # Fleurs violettes
    upper_hyacinth_flower = np.array([170, 255, 255])

    # Création des masques
    masque_hsv1 = cv2.inRange(hsv, lower_hyacinth1, upper_hyacinth1)
    masque_hsv2 = cv2.inRange(hsv, lower_hyacinth2, upper_hyacinth2)
    masque_hsv_fleur = cv2.inRange(hsv, lower_hyacinth_flower, upper_hyacinth_flower)

    # Combinaison des masques de végétation
    masque_hsv_vegetation = cv2.bitwise_or(masque_hsv1, masque_hsv2)

    # Dilatation du masque des fleurs pour identifier les zones proches
    kernel_fleur = np.ones((15, 15), np.uint8)
    masque_hsv_fleur_dilate = cv2.dilate(masque_hsv_fleur, kernel_fleur, iterations=2)

    # Zones où la végétation verte est proche des fleurs
    masque_vegetation_pres_fleurs = cv2.bitwise_and(masque_hsv_vegetation, masque_hsv_fleur_dilate)

    # Combinaison pondérée des masques
    masque_hsv = cv2.addWeighted(masque_hsv_vegetation, 0.7, masque_vegetation_pres_fleurs, 0.3, 0)

    # Suppression des petits éléments (bruit)
    kernel_erosion = np.ones((3, 3), np.uint8)
    masque_erode = cv2.erode(masque_hsv, kernel_erosion, iterations=1)

    # Suppression du bruit et amélioration des formes
    kernel_clean = np.ones((7, 7), np.uint8)
    masque_clean = cv2.morphologyEx(masque_erode, cv2.MORPH_CLOSE, kernel_clean)
    masque_clean = cv2.morphologyEx(masque_clean, cv2.MORPH_OPEN, kernel_clean)

    # Détection de la structure en rosette
    kernel_size = 9
    kernel_circle = np.zeros((kernel_size, kernel_size), np.uint8)
    cv2.circle(kernel_circle, (kernel_size // 2, kernel_size // 2), kernel_size // 2, 1, -1)
    masque_rosette = cv2.morphologyEx(masque_clean, cv2.MORPH_CLOSE, kernel_circle)

    # Dilatation pour relier les zones proches
    kernel_dilation = np.ones((5, 5), np.uint8)
    masque_dilate = cv2.dilate(masque_rosette, kernel_dilation, iterations=1)

    # Détection des contours
    contours, _ = cv2.findContours(masque_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrage des contours
    contours_filtres = []
    contours_notes = []
    min_area = 50 * scale * scale
    max_area = 50000 * scale * scale

    # Conversion en niveaux de gris pour analyse de texture
    gray = cv2.cvtColor(image_clahe, cv2.COLOR_BGR2GRAY)

    # Classificateurs par taille
    petites_colonies = 0
    moyennes_colonies = 0
    grandes_colonies = 0
    
    # Seuils de classification par taille (en pixels²)
    seuil_petite = 200 * scale * scale
    seuil_moyenne = 1000 * scale * scale

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            # Classification par taille
            if area < seuil_petite:
                petites_colonies += 1
            elif area < seuil_moyenne:
                moyennes_colonies += 1
            else:
                grandes_colonies += 1
                
            # Calcul de circularité
            perimetre = cv2.arcLength(cnt, True)
            if perimetre > 0:
                circularite = 4 * np.pi * area / (perimetre * perimetre)

                # Calcul du rapport d'aspect
                rect = cv2.minAreaRect(cnt)
                largeur, hauteur = rect[1]
                if min(largeur, hauteur) > 0:
                    aspect_ratio = max(largeur, hauteur) / min(largeur, hauteur)
                else:
                    aspect_ratio = 1

                # Analyse de texture
                mask_roi = np.zeros_like(gray)
                cv2.drawContours(mask_roi, [cnt], -1, 255, -1)
                roi = cv2.bitwise_and(gray, gray, mask=mask_roi)
                non_zero_coords = cv2.findNonZero(mask_roi)

                if non_zero_coords is not None and len(non_zero_coords) > 100:
                    x, y, w, h = cv2.boundingRect(non_zero_coords)
                    roi_cropped = roi[y:y + h, x:x + w]
                    roi_cropped_no_zeros = roi_cropped[roi_cropped > 0]

                    if len(roi_cropped_no_zeros) > 0:
                        moyenne = np.mean(roi_cropped_no_zeros)
                        ecart_type = np.std(roi_cropped_no_zeros)
                        texture_score = 1.0
                        if 15 < ecart_type < 50:  # Plage typique pour la jacinthe
                            texture_score = 1.5
                    else:
                        texture_score = 0.8
                else:
                    texture_score = 0.8

                # Score final
                score_final = (circularite * 0.4) + (aspect_ratio * 0.3) + (texture_score * 0.3)

                # Filtrage par score
                if score_final > 0.7:  # Seuil de confiance
                    contours_filtres.append(cnt)
                    contours_notes.append(score_final)

    # Calcul des métriques finales
    pixels_totaux = image.shape[0] * image.shape[1]
    pixels_jacinthe = sum([cv2.contourArea(cnt) for cnt in contours_filtres])
    pourcentage = (pixels_jacinthe / pixels_totaux) * 100
    
    # Calcul de la surface réelle si résolution fournie
    surface_totale_m2 = pixels_totaux * resolution_m2_px if resolution_m2_px else 0
    surface_jacinthe_m2 = pixels_jacinthe * resolution_m2_px if resolution_m2_px else 0
    
    # Calcul de la fiabilité de détection (basé sur la qualité des contours et des scores)
    scores_moyens = np.mean(contours_notes) if contours_notes else 0
    fiabilite = int(min(scores_moyens * 66.7, 95))  # Conversion des scores en pourcentage (max 95%)
    
    # Statistiques détaillées
    nb_colonies = len(contours_filtres)
    taille_moyenne = pixels_jacinthe / nb_colonies if nb_colonies > 0 else 0

    # Création de l'image résultat
    image_resultat = image_rgb.copy()
    masque_colore = np.zeros_like(image_rgb)

    # Visualisation avec code couleur selon le niveau de confiance
    for i, (cnt, score) in enumerate(zip(contours_filtres, contours_notes)):
        normalized_score = min(1.0, score / 1.5)
        r = int(255 * (1 - normalized_score))
        g = 150 + int(105 * normalized_score)
        b = int(50 * (1 - normalized_score))
        color = (r, g, b)

        # Dessin des contours
        cv2.drawContours(image_resultat, [cnt], -1, color, 2)
        cv2.drawContours(masque_colore, [cnt], -1, color, -1)

        # Ajout d'un indice de confiance pour les grandes colonies
        area = cv2.contourArea(cnt)
        if area > 500 * scale * scale:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                confidence = int(score * 100 / 1.5)
                confidence = min(99, confidence)
                cv2.putText(image_resultat, f"{confidence}%", (cx, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Création de l'image overlay (superposition semi-transparente)
    overlay = cv2.addWeighted(image_rgb, 0.7, masque_colore, 0.3, 0)
    
    # Compilation des résultats
    resultats = {
        "pourcentage_couverture": pourcentage,
        "nombre_colonies": nb_colonies,
        "petites_colonies": petites_colonies,
        "moyennes_colonies": moyennes_colonies,
        "grandes_colonies": grandes_colonies,
        "taille_moyenne_colonie_px": taille_moyenne,
        "surface_totale_m2": surface_totale_m2,
        "surface_jacinthe_m2": surface_jacinthe_m2,
        "fiabilite_detection": fiabilite,
    }
    
    # Sauvegarde des résultats si demandé
    if sauvegarder:
        if not os.path.exists(dossier_sortie):
            os.makedirs(dossier_sortie)
            
        # Nom de fichier basé sur l'horodatage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarde des images
        cv2.imwrite(f"{dossier_sortie}/jacinthe_detection_{timestamp}.jpg", 
                   cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{dossier_sortie}/jacinthe_masque_{timestamp}.png", masque_dilate)
        
        # Sauvegarde des résultats en JSON
        with open(f"{dossier_sortie}/jacinthe_resultats_{timestamp}.json", 'w') as f:
            json.dump(resultats, f, indent=4)
    
    # Affichage si demandé
    if visualiser:
        # Utiliser matplotlib pour l'affichage
        try:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            plt.imshow(image_rgb)
            plt.title('Image originale')
            
            plt.subplot(2, 2, 2)
            plt.imshow(masque_dilate, cmap='gray')
            plt.title('Masque de détection')
            
            plt.subplot(2, 2, 3)
            plt.imshow(masque_colore)
            plt.title('Classification par confiance')
            
            plt.subplot(2, 2, 4)
            plt.imshow(overlay)
            plt.title('Résultat final')
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            # Fallback vers OpenCV si matplotlib n'est pas disponible
            cv2.imshow('Détection Jacinthe', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return resultats, masque_dilate, overlay, contours_filtres, image_rgb

# Fonctions utilitaires pour l'UI
def get_image_download_link(img, filename, text):
    """Génère un lien de téléchargement pour une image"""
    buffered = BytesIO()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
    img_pil = Image.fromarray(img_rgb)
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def get_json_download_link(dict_data, filename, text):
    """Génère un lien de téléchargement pour des données JSON"""
    json_str = json.dumps(dict_data, indent=4)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="{filename}">{text}</a>'
    return href

def create_comparison_chart(resultats):
    """Crée un graphique de comparaison pour les colonies"""
    labels = ['Petites', 'Moyennes', 'Grandes']
    values = [resultats['petites_colonies'], resultats['moyennes_colonies'], resultats['grandes_colonies']]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=['#8BC34A', '#4CAF50', '#2E7D32'])
    
    ax.set_title('Distribution des colonies par taille')
    ax.set_ylabel('Nombre de colonies')
    
    # Ajout des valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def create_coverage_gauge(pourcentage):
    """Crée un graphique de jauge pour le pourcentage de couverture"""
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'polar': True})
    
    # Paramètres de la jauge
    theta = np.linspace(0, 1, 100) * np.pi
    rad = 0.8
    width = 0.2
    
    # Dessin de la base de la jauge (vert clair à rouge)
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, 100))
    bars = ax.bar(theta, rad, width=width, bottom=0.2, color=colors, alpha=0.8)
    
    # Dessin de l'indicateur
    pos = (pourcentage / 100) * np.pi
    ax.bar([pos], [1], width=0.03, bottom=0, color='black')
    
    # Personnalisation
    ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_yticks([])
    ax.set_title(f'Couverture: {pourcentage:.1f}%', pad=15)
    
    # Masquer la moitié inférieure
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    
    plt.tight_layout()
    return fig

# Configuration de Streamlit avec un thème moderne
st.set_page_config(
    page_title="Détection de Jacinthe d'Eau | Analyse d'Images",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import de PIL pour la manipulation d'images
try:
    from PIL import Image
except ImportError:
    from PIL import Image

# CSS personnalisé pour un design moderne
st.markdown("""
<style>
    /* Couleurs principales */
    :root {
        --primary: #2E7D32;
        --primary-light: #4CAF50;
        --primary-dark: #1B5E20;
        --accent: #8BC34A;
        --text: #212121;
        --text-light: #757575;
        --background: #F5F7FA;
        --card: #FFFFFF;
        --border: #E0E0E0;
    }
    
    /* Style général */
    body {
        background-color: var(--background);
        color: var(--text);
        font-family: 'Roboto', sans-serif;
    }
    
    /* En-tête */
    .main-header {
        background: linear-gradient(135deg, var(--primary), var(--primary-light));
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    /* Cartes */
    .card {
        background-color: var(--card);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border-left: 4px solid var(--primary);
    }
    
    .card-header {
        color: var(--primary);
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.5rem;
    }
    
    /* Boutons */
    .stButton > button {
        background-color: var(--primary) !important;
        color: white !important;
        border-radius: 50px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 500 !important;
        border: none !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    
    .stButton > button:hover {
        background-color: var(--primary-dark) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
    }
    
    /* File uploader */
    .st-bc {
        background-color: var(--card) !important;
        border-radius: 10px !important;
        border: 2px dashed var(--primary-light) !important;
        padding: 1rem !important;
    }
    
    /* Tableau de résultats */
    .dataframe {
        width: 100%;
        border-collapse: collapse;
    }
    
    .dataframe th {
        background-color: var(--primary-light);
        color: white;
        padding: 12px;
        text-align: left;
    }
    
    .dataframe td {
        padding: 10px;
        border-bottom: 1px solid var(--border);
    }
    
    .dataframe tr:hover {
        background-color: #f5f5f5;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        font-size: 0.875rem;
        font-weight: 500;
        text-align: center;
    }
    
    .badge-success {
        background-color: #4CAF50;
        color: white;
    }
    
    .badge-warning {
        background-color: #FFC107;
        color: #212121;
    }
    
    .badge-danger {
        background-color: #F44336;
        color: white;
    }
    
    /* Animation de chargement */
    .loading-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
    
    /* Onglets stylisés */
    .tab-container {
        display: flex;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1rem;
    }
    
    .tab {
        padding: 0.75rem 1.5rem;
        cursor: pointer;
        border-bottom: 3px solid transparent;
        transition: all 0.3s;
    }
    
    .tab.active {
        border-bottom: 3px solid var(--primary);
        color: var(--primary);
        font-weight: 500;
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted var(--primary-light);
    }
    
    /* Légende de couleur */
    .color-legend {
        display: flex;
        align-items: center;
        margin-top: 1rem;
    }
    
    .color-box {
        width: 20px;
        height: 20px;
        margin-right: 8px;
        border-radius: 4px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: var(--text-light);
        font-size: 0.9rem;
        border-top: 1px solid var(--border);
        margin-top: 2rem;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header {
            padding: 1.5rem;
        }
        
        .card {
            padding: 1rem;
        }
    }
    
    /* Progress bar personnalisé */
    .stProgress > div > div {
        background-color: var(--primary-light) !important;
    }
    
    /* Images avec coins arrondis */
    img {
        border-radius: 8px;
    }

    /* Section de téléchargement */
    .download-section {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 3px solid var(--primary);
        margin-top: 20px;
    }
    
    .download-button {
        text-decoration: none;
        color: white;
        background-color: var(--primary);
        padding: 10px 15px;
        border-radius: 5px;
        display: inline-block;
        margin: 5px 0;
        transition: background-color 0.3s;
    }
    
    .download-button:hover {
        background-color: var(--primary-dark);
    }
    
    /* Indicateur de fiabilité */
    .reliability-meter {
        width: 100%;
        height: 10px;
        background-color: #e0e0e0;
        border-radius: 5px;
        margin-top: 5px;
        position: relative;
    }
    
    .reliability-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)

# En-tête de l'application
st.markdown("""
<div class="main-header">
    <div class="header-icon">🌿</div>
    <h1 style="margin:0;">Détection de Jacinthe d'Eau</h1>
    <p style="margin-top:10px;opacity:0.8;">Analyse avancée d'images pour la surveillance écologique des plans d'eau</p>
</div>
""", unsafe_allow_html=True)

# Création d'une barre latérale pour les paramètres
with st.sidebar:
    st.markdown("### ⚙️ Paramètres")
    
    # Paramètres d'analyse
    st.markdown("#### Paramètres d'analyse")
    resolution_input = st.number_input(
        "Résolution (m² par pixel)",
        min_value=0.0001,
        max_value=1.0,
        value=0.01,
        format="%.4f",
        help="Définit la résolution spatiale de l'image pour calculer la surface réelle"
    )
    
    # Mode d'affichage
    st.markdown("#### Options d'affichage")
    affichage_mode = st.radio(
        "Mode d'affichage des résultats",
        ["Standard", "Comparaison", "Détaillé"],
        help="Choisissez comment afficher les résultats d'analyse"
    )
    
    # Options d'exportation
    st.markdown("#### Options d'exportation")
    format_export = st.selectbox(
        "Format d'exportation",
        ["PNG", "JPG", "JSON"],
        help="Format pour l'exportation des résultats"
    )
    
    # À propos
    st.markdown("---")
    st.markdown("### À propos")
    st.markdown("""
    Cette application utilise des techniques avancées de traitement d'image pour détecter et quantifier la présence de jacinthe d'eau dans les écosystèmes aquatiques.
    
    """)

# Corps principal
st.markdown("""
<div class="card">
    <h3 class="card-header">📸 Importer une image</h3>
    <p>Téléchargez une image aérienne ou de drone pour analyse</p>
</div>
""", unsafe_allow_html=True)

# Section de téléchargement d'image avec preview
uploaded_file = st.file_uploader("Sélectionnez une image JPG ou PNG", type=["jpg", "jpeg", "png"])

# Variables de session pour stocker les résultats d'analyse
if 'analyse_effectuee' not in st.session_state:
    st.session_state.analyse_effectuee = False

if 'resultats' not in st.session_state:
    st.session_state.resultats = None
    
if 'images' not in st.session_state:
    st.session_state.images = None

# Interface principale
if uploaded_file is not None:
    # Lecture et affichage de l'image téléchargée
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Image téléchargée", use_container_width=True)

    # Bouton pour lancer l'analyse
    if st.button("🔍 Détecter la jacinthe d'eau"):
        with st.spinner("Analyse en cours..."):
            # Exécution de l'algorithme de détection
            resultats, masque, overlay, contours, image_originale = detecter_jacinthe(
                image, 
                visualiser=False, 
                sauvegarder=False,
                resolution_m2_px=resolution_input
            )
            
            # Stockage des résultats dans la session
            st.session_state.resultats = resultats
            st.session_state.images = {
                "masque": masque,
                "overlay": overlay,
                "originale": image_originale
            }
            st.session_state.analyse_effectuee = True
            
            # Notification de succès
            st.success("Analyse terminée avec succès!")
            
            # Forcer le rechargement pour afficher les résultats
            st.query_params.get_all(all)



# Affichage des résultats si l'analyse a été effectuée
if st.session_state.analyse_effectuee and st.session_state.resultats is not None:
    resultats = st.session_state.resultats
    images = st.session_state.images
    
    # Création des onglets
    tab1, tab2, tab3 = st.tabs(["Résultats", "Visualisation", "Données détaillées"])
    
    with tab1:
        st.markdown("""
        <div class="card">
            <h3 class="card-header">📊 Résultats de l'analyse</h3>
            <p>Résumé des résultats de détection de jacinthe d'eau</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Layout en colonnes pour les métriques principales
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Surface couverte", 
                f"{resultats['pourcentage_couverture']:.1f}%",
                help="Pourcentage de la surface de l'image couverte par la jacinthe d'eau"
            )
            
            # Graphique de jauge pour le pourcentage de couverture
            fig_gauge = create_coverage_gauge(resultats['pourcentage_couverture'])
            st.pyplot(fig_gauge)
        
        with col2:
            st.metric(
                "Nombre de colonies", 
                f"{resultats['nombre_colonies']}",
                help="Nombre total de colonies de jacinthe détectées"
            )
            
            # Graphique de distribution des colonies
            fig_dist = create_comparison_chart(resultats)
            st.pyplot(fig_dist)
        
        with col3:
            if resultats['surface_jacinthe_m2'] > 0:
                st.metric(
                    "Surface estimée", 
                    f"{resultats['surface_jacinthe_m2']:.2f} m²",
                    help="Surface estimée couverte par la jacinthe (basée sur la résolution définie)"
                )
            
            # Indicateur de fiabilité
            st.markdown(f"""
            <div style="margin-top: 20px;">
                <h4>Fiabilité de la détection: {resultats['fiabilite_detection']}%</h4>
                <div class="reliability-meter">
                    <div class="reliability-fill" style="width: {resultats['fiabilite_detection']}%; 
                         background-color: {'#4CAF50' if resultats['fiabilite_detection'] > 70 
                                           else '#FFC107' if resultats['fiabilite_detection'] > 50 
                                           else '#F44336'};">
                    </div>
                </div>
                <p style="font-size: 0.8rem; margin-top: 5px;">
                    {'Excellente' if resultats['fiabilite_detection'] > 80 
                     else 'Bonne' if resultats['fiabilite_detection'] > 70 
                     else 'Moyenne' if resultats['fiabilite_detection'] > 50 
                     else 'Faible'} qualité de détection
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="card">
            <h3 class="card-header">👁️ Visualisation</h3>
            <p>Vue détaillée des résultats de la détection</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Options de visualisation
        viz_option = st.radio(
            "Mode de visualisation",
            ["Résultat final", "Masque de détection", "Comparaison"],
            horizontal=True
        )
        
        if viz_option == "Résultat final":
            st.image(images["overlay"], caption="Détection superposée", use_container_width=True)
            
            # Légende explicative
            st.markdown("""
            <div class="color-legend">
                <div style="margin-bottom: 10px;"><b>Légende:</b></div>
                <div style="display: flex; align-items: center; margin-right: 20px;">
                    <div class="color-box" style="background-color: rgb(50, 200, 50);"></div>
                    <span>Forte confiance</span>
                </div>
                <div style="display: flex; align-items: center; margin-right: 20px;">
                    <div class="color-box" style="background-color: rgb(150, 180, 50);"></div>
                    <span>Confiance moyenne</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div class="color-box" style="background-color: rgb(200, 150, 50);"></div>
                    <span>De l'eau</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        elif viz_option == "Masque de détection":
            st.image(images["masque"], caption="Masque binaire", use_container_width=True)
            
        else:  # Comparaison
            col1, col2 = st.columns(2)
            with col1:
                st.image(images["originale"], caption="Image originale", use_container_width=True)
            with col2:
                st.image(images["overlay"], caption="Détection", use_container_width=True)
    
    with tab3:
        st.markdown("""
        <div class="card">
            <h3 class="card-header">📋 Données détaillées</h3>
            <p>Informations complètes sur les résultats de l'analyse</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tableau des résultats détaillés
        df_resultats = pd.DataFrame({
            "Métrique": [
                "Couverture (%)",
                "Nombre total de colonies",
                "Petites colonies",
                "Moyennes colonies", 
                "Grandes colonies",
                "Taille moyenne des colonies (px²)",
                "Surface totale analysée (m²)",
                "Surface couverte par la jacinthe (m²)",
                "Fiabilité de la détection (%)"
            ],
            "Valeur": [
                f"{resultats['pourcentage_couverture']:.2f}",
                resultats['nombre_colonies'],
                resultats['petites_colonies'],
                resultats['moyennes_colonies'],
                resultats['grandes_colonies'],
                f"{resultats['taille_moyenne_colonie_px']:.2f}",
                f"{resultats['surface_totale_m2']:.2f}" if resultats['surface_totale_m2'] > 0 else "N/A",
                f"{resultats['surface_jacinthe_m2']:.2f}" if resultats['surface_jacinthe_m2'] > 0 else "N/A",
                f"{resultats['fiabilite_detection']}"
            ]
        })
        
        st.dataframe(df_resultats, use_container_width=True)
        
        # Section d'exportation
        st.markdown("""
        <div class="download-section">
            <h4>Exporter les résultats</h4>
            <p>Téléchargez les résultats sous différents formats pour vos analyses</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Téléchargement de l'image résultat
            st.markdown(
                get_image_download_link(
                    cv2.cvtColor(images["overlay"], cv2.COLOR_RGB2BGR), 
                    "detection_jacinthe.png", 
                    "📥 Télécharger l'image de détection"
                ), 
                unsafe_allow_html=True
            )
            
            # Téléchargement du masque
            st.markdown(
                get_image_download_link(
                    images["masque"], 
                    "masque_jacinthe.png", 
                    "📥 Télécharger le masque binaire"
                ), 
                unsafe_allow_html=True
            )
        
        with col2:
            # Téléchargement des données JSON
            st.markdown(
                get_json_download_link(
                    resultats, 
                    "resultats_jacinthe.json", 
                    "📥 Télécharger les données (JSON)"
                ), 
                unsafe_allow_html=True
            )
            

# Pied de page
st.markdown("""
<div class="footer">
    <p>Outil de détection de jacinthe d'eau | Développé pour la surveillance écologique</p>
    <p style="font-size: 0.8rem;">© 2025 - Tous droits réservés</p>
</div>
""", unsafe_allow_html=True)
    