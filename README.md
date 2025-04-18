# 🧠 Reconnaissance Faciale avec OpenCV et Python

Projet d'initiation à la **reconnaissance faciale** utilisant **Python** et **OpenCV**, avec plusieurs modules pour :
- 📸 détecter les visages sur des **images statiques**
- 🎥 effectuer une détection **en temps réel** via webcam
- 🧪 entraîner un jeu de données local à partir de visages

## 🧰 Technologies utilisées

- Python 3
- OpenCV (cv2)
- Haarcascade classifiers
- VS Code

## 📂 Arborescence

```
ressource-projet-ia/
│
├── face-detection.py             # Détection de visages sur image statique
├── realtime.py                   # Détection de visages en temps réel (webcam)
├── dataset.py                    # Génération des datasets d'entraînement
├── app.py / apps.py              # Variantes du lanceur principal
├── haarcascade_frontalface.xml   # Classifieurs pré-entraînés
├── images/                       # Exemples (brad, obama, etc.)
└── visages/                      # Dossiers de visages capturés
```
## 🚀 Lancer les scripts

### 1. Détection sur image
```bash
python face-detection.py
```

### 2. Détection en temps réel
```bash
python realtime.py
```

### 3. Génération du dataset
```bash
python dataset.py
```

## 🔍 Exemple d'utilisation

- Détection de visages sur une photo d’un groupe de personnes
- Extraction de visages à partir d’une webcam pour entraînement
- Conversion du modèle pour une reconnaissance simplifiée

## 📌 Ce que j’ai appris

- Utilisation d’OpenCV pour la vision par ordinateur
- Détection de visages via Haarcascade
- Travail avec des flux vidéo (temps réel)
- Manipulation de fichiers image et XML pour les classifieurs

## 📸 Démonstration

Bientôt...

## ✅ Statut

Projet fonctionnel – terminé ✅
