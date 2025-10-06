"""
===========================================
 PRÉAMBULE : Entraînement d’un modèle YOLOv8
===========================================

Ce script illustre l’utilisation du modèle YOLOv8 développé par Ultralytics pour la détection d’objets. 
YOLO est un algorithme de vision par ordinateur capable d’identifier et de localiser 
plusieurs objets dans une image.

Le code se déroule en plusieurs étapes :

1. Importation de la bibliothèque Ultralytics.
2. Chargement d’un modèle YOLOv8 pré-entraîné (ici la version "nano" : yolov8n.pt).
3. Test du modèle sur une image du dataset et visualisation des prédictions.
4. Test du modèle sur une video et visualisation des prédictions.

"""
from ultralytics import YOLO

# Modèle pré-entraîné YOLOv8n sur le dataset COCO entier
model = YOLO("yolov8n.pt")

# Tester le modèle sur une image du dataset
test_results = model.predict(
    source="./datasets/coco128/images/train2017/000000000109.jpg",  
    conf=0.25,               # seuil de confiance
    show=True
)

video_results = model.predict(
    source="./video/traffic.mp4",  
    conf=0.25,              # seuil de confiance minimal
    show=True               # affichage 
)
