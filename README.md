
# 👾 Pac-Man RL – Deep Reinforcement Learning

Un projet d’intelligence artificielle où une IA apprend à jouer à Pac-Man toute seule grâce à l’apprentissage par renforcement profond (**Deep Reinforcement Learning**) avec Stable-Baselines3 et PyTorch.

---

## 📂 Contenu du projet

- `train.py` → Script principal pour entraîner l’IA sur Pac-Man
- `play.py` → Lancer une partie avec le modèle déjà entraîné
- `models/` → Dossier contenant les modèles sauvegardés (`best_model`, checkpoints…)
- `logs/` → Dossier de logs TensorBoard pour suivre l’entraînement
- `utils.py` → Fonctions utilitaires (ex: wrappers, preprocessing)
- `requirements.txt` → Dépendances à installer

---

## 🎯 Objectif

Faire apprendre à une IA à jouer à Pac-Man, à éviter les fantômes et à manger un max de pac-gommes.  
L’agent explore, apprend par l’erreur, et optimise ses mouvements pour maximiser son score.

---

## 🛠️ Technologies utilisées

- Python 3.10+
- Stable-Baselines3 (algorithmes RL)
- PyTorch (backend)
- Gymnasium + Environnement Pac-Man (via roms Atari)
- TensorBoard (suivi visuel de l'entraînement)
- NumPy

---

## ⚙️ Installation

```bash
# 1. Ouvrir un terminal dans le dossier du projet
# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Installer les roms Atari (si nécessaire)
AutoROM --accept-license
```

---

## 🚀 Utilisation

### Entraîner l’IA depuis zéro

```bash
python train.py
```

### Faire jouer l’IA avec le meilleur modèle

```bash
python play.py
```

---

## 📈 Suivi de l’apprentissage

L'entraînement est loggé automatiquement dans le dossier `logs/`.  
Tu peux le visualiser avec TensorBoard :

```bash
tensorboard --logdir logs/
```

---

## 🧠 Fonctionnement

L’agent apprend à jouer à Pac-Man en interagissant avec l’environnement, en recevant des récompenses et en s’adaptant. Il est entraîné avec un algorithme de type PPO, A2C ou DQN selon la config.  
Il devient progressivement meilleur à éviter les fantômes et à prendre les bonnes décisions.

---

