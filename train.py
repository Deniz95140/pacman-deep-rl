import argparse
import os
import time

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

import ale_py
import gymnasium as gym
# Enregistre les envs ALE pour Gymnasium 1.x
gym.register_envs(ale_py)
from utils import make_env

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000, help="Nombre total de steps d'entraînement")
    parser.add_argument("--seed", type=int, default=0, help="Seed pour la reproductibilité")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, cuda:0, mps, auto")
    parser.add_argument("--save_dir", type=str, default="models", help="Dossier de sauvegarde des modèles")
    parser.add_argument("--log_dir", type=str, default="logs", help="Dossier des logs TensorBoard")
    parser.add_argument("--load", type=str, default=None, help="Chemin d'un modèle .zip à charger pour reprendre l'entraînement")
    return parser.parse_args()

def build_vec_env(render_mode=None, seed: int = 0):
    def _make():
        return make_env(render_mode=render_mode, seed=seed)
    vec_env = DummyVecEnv([_make])

    return vec_env

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    env = build_vec_env(render_mode=None, seed=args.seed)
    eval_env = build_vec_env(render_mode=None, seed=args.seed + 42)

    run_name = f"mspacman_dqn_{int(time.time())}"
    new_logger = configure(args.log_dir, ["tensorboard", "stdout"])

    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=20_000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        target_update_interval=10_000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=args.log_dir,
        device=args.device,
    )

    if args.load is not None and os.path.isfile(args.load):
        print(f"[INFO] Chargement du modèle: {args.load}")
        model = DQN.load(args.load, env=env, device=args.device, print_system_info=True)

    model.set_logger(new_logger)

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=args.save_dir,
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.save_dir,
        log_path=args.log_dir,
        eval_freq=25_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=args.timesteps, tb_log_name=run_name, callback=[checkpoint_cb, eval_cb])

    last_model_path = os.path.join(args.save_dir, "last_model.zip")
    model.save(last_model_path)
    print(f"[OK] Entraînement terminé. Modèle sauvegardé: {last_model_path}")
    print(f"[TIP] Meilleur modèle (selon EvalCallback) : {os.path.join(args.save_dir, 'best_model.zip')} (si créé)")
