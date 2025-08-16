import argparse
import os
import time

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

import ale_py
import gymnasium as gym
# Enregistre les envs ALE pour Gymnasium 1.x
gym.register_envs(ale_py)
from utils import make_env

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best_model.zip", help="Chemin vers le modèle .zip")
    parser.add_argument("--episodes", type=int, default=5, help="Nombre d'épisodes à jouer")
    parser.add_argument("--seed", type=int, default=123, help="Seed de l'env")
    return parser.parse_args()

def build_vec_env(render_mode="human", seed: int = 0):
    def _make():
        return make_env(render_mode=render_mode, seed=seed)
    env = DummyVecEnv([_make])
    return env

if __name__ == "__main__":
    args = parse_args()
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Modèle introuvable: {args.model}. Lance d'abord l'entraînement.")

    env = build_vec_env(render_mode="human", seed=args.seed)
    model = DQN.load(args.model, env=env, print_system_info=True)

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += float(reward)
            time.sleep(0.01)
        print(f"[PLAY] Épisode {ep} — Récompense: {ep_reward}")
    env.close()
