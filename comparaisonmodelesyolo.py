"""
===========================================
PRÉAMBULE : Comparaison de modèles YOLO
===========================================

Ce script permet d'évaluer et de comparer plusieurs modèles YOLO sur le dataset COCO128. 
Il combine des tests de performance, des mesures de précision, et des visualisations des résultats.

Fonctionnalités principales :

1. Définition des modèles à comparer
2. Préparation des images de test
3. Organisation des résultats
4. Évaluation des modèles
5. Compilation des résultats
6. Visualisation
   - Graphique « Vitesse vs Précision » : temps moyen par image vs mAP.
   - Histogrammes comparatifs des métriques (mAP, précision, rappel) pour chaque modèle.
   - Sauvegarde des figures dans le dossier des résultats.


"""




from ultralytics import YOLO
from pathlib import Path
import shutil
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg") 

# %%

# Liste des modèles à comparer
models_list = ["yolov3u.pt","yolov5nu.pt", "yolov8n.pt","yolo11n.pt","yolo11s.pt"]

# Dossier contenant les images de test
test_images_dir = Path("./datasets/coco128/images/train2017")
test_images = list(test_images_dir.glob("*.jpg"))

# Dossier pour sauvegarder les résultats
save_root = Path("./runs/compare_models")

# Supprimer l'ancien dossier s'il existe
if save_root.exists():
    shutil.rmtree(save_root)
save_root.mkdir(exist_ok=True)

# Stockage des résultats pour comparaison
results_summary = []

for model_name in models_list:
    print(f"\n=== Évaluation du modèle {model_name} ===")
    model = YOLO(model_name)
    
    # Créer un dossier unique pour ce modèle
    model_dir = save_root / model_name.split(".")[0]
    model_dir.mkdir(exist_ok=True, parents=True)

    start = time.time()
    all_results = model.predict(
        source=test_images,
        conf=0.25,
        save=True,
        project=str(model_dir),   # dossier du modèle
        name="predict"            # sous-dossier pour les prédictions
    )
    end = time.time()
    avg_time = (end - start) / len(test_images)  # temps moyen par image

    metrics = model.val(
        data="coco128.yaml",
        project=str(model_dir),    
        name="val"                
    )
    precision = metrics.box.p.mean()
    recall = metrics.box.r.mean()

    results_summary.append({
        "Modèle": model_name,
        "mAP@0.5": metrics.box.map50,
        "mAP@0.5:0.95": metrics.box.map,
        "Précision": precision,
        "Rappel": recall,
        "Temps (s/img)": avg_time
    })

# %%


# Comparaison : tableau et graphique
df = pd.DataFrame(results_summary)
print("\n=== Résumé comparatif ===")
print(df)
# Sauvegarder tableau
df.to_csv(save_root / "comparaison.csv", index=False)


# Affichage compromis Vitesse vs Précision (mAP@0.5 et mAP@0.5:0.95)
plt.figure(figsize=(8,6))
# Points mAP@0.5
plt.scatter(df["Temps (s/img)"], df["mAP@0.5"], s=100, c='blue', label='mAP@0.5')
# Points mAP@0.5:0.95
plt.scatter(df["Temps (s/img)"], df["mAP@0.5:0.95"], s=100, c='red', label='mAP@0.5:0.95')

# Ajouter les noms des modèles pour les deux séries
for i, row in df.iterrows():
    plt.text(row["Temps (s/img)"] + 0.002, row["mAP@0.5"] + 0.002, row["Modèle"], fontsize=9, color='blue')
    plt.text(row["Temps (s/img)"] + 0.002, row["mAP@0.5:0.95"] + 0.002, row["Modèle"], fontsize=9, color='red')

plt.xlabel("Temps moyen par image (s)")
plt.ylabel("mAP")
plt.title("Compromis Vitesse vs Précision des modèles YOLO")
plt.grid(True)
plt.legend()
plt.savefig(save_root / "vitesse_vs_precision.png", dpi=300)
plt.show()

df.plot(x="Modèle", y=["mAP@0.5", "mAP@0.5:0.95", "Précision", "Rappel"], kind="bar")
plt.title("Comparaison YOLO sur COCO128")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.savefig(save_root / "bar_comparaison.png", dpi=300)
plt.show()


# %%
