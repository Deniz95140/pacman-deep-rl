Pacman RL — MsPacman avec DQN (Gymnasium 1.x / SB3 2.7)
======================================================
Cette version est corrigée pour Gymnasium ≥ 1.1 et ale-py ≥ 0.11 (Windows 11, Python 3.12 OK).
Plus besoin d'AutoROM. Atari est géré par ale-py.

Étapes d'installation (sans venv, Windows 11)
--------------------------------------------
1) Mettre pip à jour
   python -m pip install -U pip setuptools wheel

2) Installer PyTorch d'abord (choisis UNE des deux lignes)
   # CPU (simple) :
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   # GPU (CUDA 12.1) :
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

3) Installer les libs du projet
   pip install -r requirements.txt

4) Test rapide (une seule ligne PowerShell)
   python -c "import gymnasium as gym, ale_py; gym.register_envs(ale_py); env=gym.make('ALE/MsPacman-v5'); print('OK:', env.observation_space, env.action_space); env.close()"

Entraînement
------------
   python train.py --timesteps 200000

Jouer avec le modèle
--------------------
   python play.py --model models/best_model.zip --episodes 3

Notes techniques
----------------
- FrameStack a été remplacé par FrameStackObservation (Gymnasium 1.x).
- VecTransposeImage est supprimé (le stacking sort déjà (C, H, W)).
- Sous Windows, les blocs << 'PY' ne fonctionnent pas; utilisez python -c "..." sur une seule ligne.
