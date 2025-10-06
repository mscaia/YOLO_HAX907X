# YOLO HAX907X

Ce dépôt contient des scripts pour **tester, évaluer et comparer plusieurs modèles YOLO** sur le dataset COCO128. Les scripts permettent :

1. La **détection d’objets sur des images ou des vidéos** en utilisant des modèles pré-entraînés.  
2. La **comparaison de plusieurs modèles YOLO** (v3, v5, v8, YOLO11) sur un petit dataset d’exemple, avec calcul des métriques et visualisation des résultats.


## Contenu

- `YOLO.py` : Script pour tester un modèle YOLO pré-entraîné sur une image et une vidéo.  
- `comparaisonmodelesyolo.py` : Script pour comparer plusieurs modèles YOLO sur le dataset COCO128, calculer les métriques et le temps d’inférence, et générer des graphiques de comparaison.  
- `datasets/` : Contient le dataset COCO128 utilisé pour les tests.  
- `runs/` : Dossier où sont sauvegardés les résultats, prédictions et graphiques.
- `video/` : Dossier qui contient la vidéo.


## Modèles pré-entraînés

Aucun modèle n’est entraîné dans ce dépôt. Tous les modèles utilisés ont été pré-entraînés sur le dataset COCO complet, comprenant environ 118 000 images.  

Ce choix présente plusieurs avantages :  
- Éviter d’utiliser des modèles insuffisamment entraînés et donc peu performants.  
- Réduire le temps nécessaire au lancement et à l’exécution des scripts.

## Exécution des scripts

- Le script `YOLO.py` s’exécute en quelques secondes pour tester un modèle sur une image et une vidéo.  
- Le script `comparaisonmodelesyolo.py` s’exécute en environ 5 minutes sur un Mac avec puce M1 pour comparer tous les modèles sur le dataset COCO128.


## À propos du dataset COCO128

COCO128 est une version miniature du dataset COCO (Common Objects in Context).  
- Il contient 128 images d’exemples annotés pour 80 classes d’objets courants (personnes, véhicules, animaux, objets du quotidien…).  
- Il est utilisé ici pour tester et démontrer les modèles YOLO sans nécessiter un dataset complet, ce qui permet des tests rapides.  
- Bien que petit, il conserve la diversité des objets et des contextes pour valider les pipelines de détection.


## Installation

1. Cloner le dépôt :

```bash
git clone git@github.com:mscaia/YOLO_HAX907X.git
```

2. Installer les dépendances Python :
 
```bash
pip install ultralytics pandas matplotlib
```