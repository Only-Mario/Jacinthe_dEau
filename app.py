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

# Deep Learning
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import Image
import tempfile

def create_cnn_model(input_shape):
    """
    Crée un modèle CNN simple pour la classification binaire.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification (jacinthe or not)
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_data(image_dir, batch_size=32, img_size=(150, 150)):
    """
    Charge les images à partir d'un dossier et les prétraite pour l'entraînement.
    """
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_generator = datagen.flow_from_directory(
        image_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )
    
    validation_generator = datagen.flow_from_directory(
        image_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    return train_generator, validation_generator

def train_model(model, train_generator, validation_generator, epochs=10):
    """
    Entraîne le modèle CNN avec les données fournies.
    """
    checkpoint = ModelCheckpoint('jacinthe_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=epochs,
        callbacks=[checkpoint]
    )
    return history

def save_model(model, filename='jacinthe_model.h5'):
    """
    Sauvegarde le modèle dans un fichier.
    """
    model.save(filename)

def load_saved_model(filename='jacinthe_model.h5'):
    """
    Charge un modèle pré-entraîné à partir d'un fichier.
    """
    if os.path.exists(filename):
        return load_model(filename)
    return None

# Configuration de Streamlit
st.set_page_config(
    page_title="Détection de Jacinthe d'Eau avec CNN",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# En-tête de l'application
st.markdown("""
<div class="main-header">
    <div class="header-icon">🌿</div>
    <h1 style="margin:0;">Détection de Jacinthe d'Eau avec CNN</h1>
    <p style="margin-top:10px;opacity:0.8;">Analyse avancée d'images pour la surveillance écologique des plans d'eau</p>
</div>
""", unsafe_allow_html=True)

# Créer un dossier temporaire pour stocker les images téléchargées
temp_dir = tempfile.mkdtemp()

# Fonction pour sauvegarder les images téléchargées
def save_uploaded_image(uploaded_file, label):
    """
    Sauvegarde l'image téléchargée dans un dossier spécifique en fonction du label.
    """
    try:
        image = Image.open(uploaded_file)
        label_dir = os.path.join(temp_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        image_path = os.path.join(label_dir, uploaded_file.name)
        image.save(image_path)
        return image_path
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde de l'image : {e}")
        return None

# Interface Streamlit
uploaded_file = st.file_uploader("Téléchargez une image de jacinthe d'eau", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Afficher l'image téléchargée
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image téléchargée", use_container_width=True)
        
        # Demander à l'utilisateur de confirmer si c'est de la jacinthe d'eau
        label = st.radio("Cette image contient-elle de la jacinthe d'eau ?", ("Oui", "Non"))
        label = 1 if label == "Oui" else 0
        
        # Sauvegarder l'image avec le label
        image_path = save_uploaded_image(uploaded_file, label)
        if image_path:
            st.success(f"Image sauvegardée avec le label : {'Jacinthe d\'eau' if label == 1 else 'Non jacinthe'}")
        
        # Charger ou créer le modèle
        model = load_saved_model()
        if model is None:
            st.write("Création d'un nouveau modèle...")
            model = create_cnn_model(input_shape=(150, 150, 3))
        
        # Entraîner le modèle avec les nouvelles données
        if st.button("Entraîner le modèle avec cette image"):
            try:
                train_generator, validation_generator = load_data(temp_dir)
                history = train_model(model, train_generator, validation_generator, epochs=5)
                st.success("Modèle entraîné avec succès !")
                
                # Sauvegarder le modèle
                save_model(model)
            except Exception as e:
                st.error(f"Erreur lors de l'entraînement du modèle : {e}")
        
        # Faire une prédiction sur l'image téléchargée
        if st.button("Prédire si l'image contient de la jacinthe d'eau"):
            try:
                img = image.resize((150, 150))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                prediction = model.predict(img_array)
                
                st.write(f"Prédiction : {'Jacinthe d\'eau' if prediction > 0.5 else 'Non jacinthe'}")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
    except Exception as e:
        st.error(f"Erreur lors du traitement de l'image : {e}")